"""Prefix-cached prompt tokenization for the ingress router's vLLM renderer.

Each turn of an agent session re-renders the whole conversation, so the
router tokenizes a prompt that is mostly an earlier turn's prompt. This cache
reuses the token IDs of the longest previously seen prefix of the prompt that
ends right after an added (special) token, and encodes only the rest.

Why this is exact: HF ``tokenizers`` first extracts added tokens from the
text (leftmost-longest), then normalizes, pre-tokenizes and encodes each
piece between them on its own. A prefix that ends right after an extracted
added token therefore tokenizes to exactly the prefix of the full token IDs,
and the remainder tokenizes on its own to the rest. ``_unsupported_reason``
turns the cache off for tokenizers where that does not hold.

The keys are content hashes of prompt prefixes, not session IDs, so prompts
that share a prefix (e.g. a system prompt) share its tokens too, and the next
turn hits even though it diverges from the previous prompt shortly before its
end (the generation prompt, or the previous answer's dropped reasoning).

Finding those prefixes means scanning the whole prompt for added tokens and
hashing it. With a session ID (``RequestScope``), the cache instead compares
the prompt byte for byte with the session's latest one, takes the scan and
hashes of the shared part from it, and scans and hashes only the rest; a
wrong or stale session ID only costs that shortcut.

The lists of token IDs the cache returns hold shared ints, one int object
per token ID (``PromptTokenCache._ints``), and each stored prompt keeps its IDs
as tuples of them too, in chunks that a prompt extending it shares. A prompt's
list is then a copy of its prefix's pointers plus the new IDs, rather than a
new int object per token, which costs several times as much to create and
again to free; storing it creates (and the garbage collector scans) only the
chunks of its new tokens.
"""

import asyncio
import contextvars
import functools
import hashlib
import json
import re
import sys
import threading
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware import trace as kv_trace

logger = get_logger(__name__)

# Normalizers, pre-tokenizers and models that process each piece between added
# tokens independently of its position in the text.
_PER_PIECE_PRE_TOKENIZERS = {
    "BertPreTokenizer",
    "ByteLevel",
    "CharDelimiterSplit",
    "Digits",
    "Punctuation",
    "Sequence",
    "Split",
    "UnicodeScripts",
    "Whitespace",
    "WhitespaceSplit",
}
_DETERMINISTIC_MODELS = {"BPE", "Unigram", "WordLevel", "WordPiece"}
# With add_special_tokens=False these leave the token IDs alone; with True,
# they wrap them in fixed special tokens (checked by ``_special_affixes``).
_POST_PROCESSORS = {
    "BertProcessing",
    "ByteLevel",
    "RobertaProcessing",
    "Sequence",
    "TemplateProcessing",
}
# The only encode kwargs the cache reproduces exactly (see ``encode``).
_SUPPORTED_ENCODE_KWARGS = {"add_special_tokens", "max_length", "truncation"}
# Estimated bytes of index bookkeeping per stored prefix (key, value, dict slot).
_INDEX_BYTES_PER_KEY = 200
# A stored prompt that a newer one extends up to its last few tokens (usually
# the generation prompt the next turn replaces) is dropped, so that a session
# keeps one stored prompt rather than one per turn.
_SUPERSEDED_TAIL_TOKENS = 32
_DIGEST_SIZE = 16
_MAX_LOGGED_MISMATCHES = 20
# Per session, the hash states kept at the last boundaries of its latest
# prompt: its next prompt resumes scanning and hashing at the last of these
# that it shares. The next prompt usually differs from the latest only in its
# last one or two boundaries (the generation prompt).
_RECENT_STATES = 16
# Estimated bytes per boundary (end offset, token ID, digest) and per hash
# state of a session's latest prompt.
_RECENT_BYTES_PER_BOUNDARY = 120
_RECENT_BYTES_PER_STATE = 500
# Share of ``max_bytes`` that sessions' latest prompts may take in addition.
_RECENT_BYTES_FRACTION = 0.25
# Tokens per chunk of a stored prompt's token ID tuples (see ``_Entry``).
_ID_CHUNK = 4096


class RequestScope:
    """The prompt token cache's view of one request, entered (as a context
    manager) in the task that tokenizes the request.

    In: the request's session ID, which lets the cache compare the prompt
    with the session's latest one and scan and hash only what it adds.
    Exactness never depends on it.

    Out: the uint32 array the cache built the prompt token IDs from, so the
    caller can serialize the IDs (``tobytes``) without converting the list.
    """

    __slots__ = ("session_id", "_ids", "_array", "_token")

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self._ids: Optional[List[int]] = None
        self._array: Optional[np.ndarray] = None

    def __enter__(self) -> "RequestScope":
        self._token = _request_scope.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _request_scope.reset(self._token)

    def put(self, ids: List[int], array: np.ndarray) -> None:
        self._ids, self._array = ids, array

    def token_array(self, ids: Optional[List[int]]) -> Optional[np.ndarray]:
        """The uint32 array of ``ids``, if they are the very list the cache
        returned (not a copy or a truncated or concatenated list)."""
        return self._array if ids is not None and ids is self._ids else None


_request_scope: contextvars.ContextVar[Optional[RequestScope]] = contextvars.ContextVar(
    "prompt_token_cache_request", default=None
)


def _is_identity_normalizer(normalizer: Optional[Dict[str, Any]]) -> bool:
    if normalizer is None:
        return True
    return normalizer.get("type") == "Sequence" and all(
        _is_identity_normalizer(n) for n in normalizer.get("normalizers", ())
    )


def _pre_tokenizer_reason(pre_tokenizer: Optional[Dict[str, Any]]) -> Optional[str]:
    if pre_tokenizer is None:
        return None
    kind = pre_tokenizer.get("type")
    if kind not in _PER_PIECE_PRE_TOKENIZERS:
        return f"pre-tokenizer {kind}"
    if kind == "ByteLevel" and pre_tokenizer.get("add_prefix_space"):
        return "ByteLevel pre-tokenizer with add_prefix_space"
    for child in pre_tokenizer.get("pretokenizers") or ():
        reason = _pre_tokenizer_reason(child)
        if reason:
            return reason
    return None


def _unsupported_reason(tokenizer: Any, spec: Dict[str, Any]) -> Optional[str]:
    """Why splitting at added tokens may change this tokenizer's output, if so."""
    if getattr(tokenizer, "split_special_tokens", False):
        return "split_special_tokens is set"
    if getattr(tokenizer.backend_tokenizer, "encode_special_tokens", False):
        return "encode_special_tokens is set"
    for token in spec.get("added_tokens") or ():
        if token.get("lstrip") or token.get("rstrip") or token.get("single_word"):
            return f"added token {token.get('content')!r} strips or is single-word"
    reason = _pre_tokenizer_reason(spec.get("pre_tokenizer"))
    if reason:
        return reason
    model = spec.get("model") or {}
    if model.get("type") not in _DETERMINISTIC_MODELS:
        return f"model {model.get('type')}"
    if model.get("dropout"):
        return "BPE dropout"
    post_processor = spec.get("post_processor")
    if post_processor is not None:
        kinds = [post_processor] + list(post_processor.get("processors") or ())
        for p in kinds:
            if p.get("type") not in _POST_PROCESSORS:
                return f"post-processor {p.get('type')}"
    return None


def _trie_pattern(tokens: Iterable[bytes]) -> "re.Pattern[bytes]":
    """A regex matching ``tokens`` leftmost-longest, like ``tokenizers``'
    Aho-Corasick: a trie of bytes whose optional (greedy) branches prefer the
    longer token at the same start."""
    trie: Dict[Optional[int], Any] = {}
    for token in tokens:
        node = trie
        for byte in token:
            node = node.setdefault(byte, {})
        node[None] = True

    def build(node: Dict[Optional[int], Any]) -> bytes:
        branches = [
            re.escape(bytes([byte])) + build(child)
            for byte, child in sorted((k, v) for k, v in node.items() if k is not None)
        ]
        if not branches:
            return b""
        body = (
            branches[0] if len(branches) == 1 else b"(?:" + b"|".join(branches) + b")"
        )
        return b"(?:" + body + b")?" if None in node else body

    return re.compile(build(trie), re.DOTALL)


def _non_overlapping(matches: List[Tuple[int, int, int]]) -> bool:
    """Whether sorted (start, end, ID) matches are disjoint."""
    previous_end = 0
    for start, end, _ in matches:
        if start < previous_end:
            return False
        previous_end = end
    return True


class _TokenMatcher:
    """Leftmost-longest, non-overlapping matches of a set of tokens, as
    ``tokenizers``' Aho-Corasick finds them."""

    def __init__(self, token_ids: Dict[bytes, int]):
        self._ids = token_ids
        by_first_byte: Dict[int, List[bytes]] = {}
        for token in token_ids:
            by_first_byte.setdefault(token[0], []).append(token)
        # One pattern per first byte: re finds a pattern's literal first byte
        # quickly, but tries a pattern of alternatives at every byte.
        self._patterns = [_trie_pattern(group) for group in by_first_byte.values()]
        self._combined = _trie_pattern(token_ids)

    def _scan(self, pattern: "re.Pattern[bytes]", data: bytes, start: int, end: int):
        ids = self._ids
        return [
            (m.start(), m.end(), ids[m.group()])
            for m in pattern.finditer(data, start, end)
        ]

    def find(self, data: bytes, start: int, end: int) -> List[Tuple[int, int, int]]:
        """(start, end, token ID) of each match in ``data[start:end]``, in order."""
        if len(self._patterns) == 1:
            return self._scan(self._patterns[0], data, start, end)
        found = sorted(
            match
            for pattern in self._patterns
            for match in self._scan(pattern, data, start, end)
        )
        # Without overlaps, the union of the patterns' matches is what one scan
        # for all tokens finds; with them, a match may hide another.
        if _non_overlapping(found):
            return found
        return self._scan(self._combined, data, start, end)


class _BoundaryScanner:
    """Finds where the tokenizer extracts the boundary added tokens in a text.

    Mirrors ``tokenizers``' ``AddedVocabulary``: tokens with ``normalized=False``
    are extracted from the raw text first, then ``normalized=True`` tokens from
    the (normalized) pieces between them. Normalized tokens are boundaries only
    when the normalizer is the identity, since otherwise their raw-text
    positions are unknown.
    """

    def __init__(self, added_tokens: List[Dict[str, Any]], identity_normalizer: bool):
        raw = {
            t["content"].encode(): t["id"] for t in added_tokens if not t["normalized"]
        }
        normalized = {
            t["content"].encode(): t["id"] for t in added_tokens if t["normalized"]
        }
        self.token_ids = dict(raw)
        self._raw = _TokenMatcher(raw) if raw else None
        self._normalized = None
        if identity_normalizer and normalized:
            self.token_ids.update(normalized)
            self._normalized = _TokenMatcher(normalized)
        # Whether the scan matches a token at a position depends on at most
        # this many bytes from there.
        self.max_token_bytes = max(map(len, self.token_ids), default=0)

    def scan(self, data: bytes, start: int = 0) -> Tuple[List[int], List[int]]:
        """Returns the end offsets and token IDs of the boundary tokens in
        ``data[start:]``.

        ``start`` must be 0 or the end of a boundary token of ``data``, where
        the scan of all of ``data`` resumes too.
        """
        end = len(data)
        raw = self._raw.find(data, start, end) if self._raw is not None else []
        normalized = self._normalized
        if normalized is None:
            matches = raw
        else:
            # Normalized tokens are extracted from the pieces between raw ones.
            # Scanning the whole text finds the same unless a match overlaps a
            # raw token.
            matches = sorted(raw + normalized.find(data, start, end))
            if not _non_overlapping(matches):
                matches = []
                gap_start = start
                for match in raw:
                    if match[0] > gap_start:
                        matches += normalized.find(data, gap_start, match[0])
                    matches.append(match)
                    gap_start = match[1]
                if gap_start < end:
                    matches += normalized.find(data, gap_start, end)
        return [m[1] for m in matches], [m[2] for m in matches]


class _Entry:
    """The token IDs of one stored prompt, and the index keys pointing to it.

    Holds the IDs twice: as a uint32 array (``tokens``), and as tuples of
    the cache's shared ints (``chunks``) that a list of a prefix's IDs
    copies. Every chunk but the last has ``_ID_CHUNK`` IDs, so a prompt that
    extends this one shares the chunks within its reused prefix.
    ``counts[i]`` is the number of tokens up to and including the prompt's
    ``i``-th boundary token.
    """

    __slots__ = ("tokens", "chunks", "keys", "counts", "live", "nbytes")

    def __init__(
        self,
        tokens: np.ndarray,
        chunks: Tuple[Tuple[int, ...], ...],
        keys: List[bytes],
        counts: np.ndarray,
    ):
        self.tokens = tokens
        self.chunks = chunks
        self.keys = keys
        self.counts = counts
        self.live = 0
        # Counts shared chunks in full, as an older prompt holding them may
        # be dropped.
        self.nbytes = (
            tokens.nbytes
            + sys.getsizeof(chunks)
            + sum(map(sys.getsizeof, chunks))
            + counts.nbytes
            + len(keys) * _INDEX_BYTES_PER_KEY
        )


class _RecentPrompt:
    """A session's latest stored prompt, to scan and hash only what the
    session's next prompt adds to it.

    Holds the prompt's UTF-8 bytes, its boundaries' end offsets, token IDs
    and digests, and the hash states at its last ``_RECENT_STATES``
    boundaries, keyed by boundary index.
    """

    __slots__ = ("data", "ends", "ids", "digests", "states", "nbytes")

    def __init__(
        self,
        data: bytes,
        ends: List[int],
        ids: List[int],
        digests: List[bytes],
        states: List[Tuple[int, Any]],
    ):
        self.data = data
        self.ends = ends
        self.ids = ids
        self.digests = digests
        self.states = states
        self.nbytes = (
            len(data)
            + len(ends) * _RECENT_BYTES_PER_BOUNDARY
            + len(states) * _RECENT_BYTES_PER_STATE
        )


class PromptTokenCache:
    """Reuses the token IDs of earlier prompts' prefixes when tokenizing a prompt.

    Every prefix of a stored prompt that ends right after a boundary added
    token is indexed by the hash of its UTF-8 bytes, so a new prompt reuses its
    longest stored prefix and encodes only the rest. Stored prompts are evicted
    least recently used beyond ``max_bytes`` of tokens and index. Each
    session's latest prompt (its text and scan) is kept too, least recently
    used beyond ``max_bytes * _RECENT_BYTES_FRACTION``.

    Thread-safe: the renderer encodes on a thread pool. The lock covers only
    the index and LRU, never an encode.

    Args:
        tokenizer: The renderer's HF fast tokenizer.
        spec: The parsed ``tokenizer.json`` of its backend.
        max_bytes: Memory budget for stored token IDs and their index.
        verify: Also encode every prompt in full, return that, and log
            any mismatch with the cached encoding.
    """

    def __init__(
        self,
        tokenizer: Any,
        spec: Dict[str, Any],
        *,
        max_bytes: int,
        verify: bool = False,
    ):
        self._tokenizer = tokenizer
        self._max_bytes = max_bytes
        self._verify = verify
        self._scanner = _BoundaryScanner(
            spec.get("added_tokens") or [],
            _is_identity_normalizer(spec.get("normalizer")),
        )
        backend = tokenizer.backend_tokenizer
        # Above every token ID the tokenizer can return.
        size = max(
            [backend.get_vocab_size(with_added_tokens=True)]
            + [t["id"] + 1 for t in spec.get("added_tokens") or ()]
        )
        self._is_boundary_id = np.zeros(size, dtype=bool)
        self._is_boundary_id[list(self._scanner.token_ids.values())] = True
        # One int object per token ID, shared by all the token ID lists and
        # tuples the cache builds (``_id_list``, ``_id_chunks``).
        self._ints = np.arange(size).astype(object)
        self._special_affixes = self._probe_special_affixes()
        self._fast_backend = self._make_fast_backend()
        self._lock = threading.Lock()
        self._index: Dict[bytes, Tuple[_Entry, int]] = {}
        self._lru: "OrderedDict[_Entry, None]" = OrderedDict()
        self._bytes = 0
        # Each session's latest stored prompt, least recently stored first.
        self._recent: "OrderedDict[str, _RecentPrompt]" = OrderedDict()
        self._recent_bytes = 0
        self._max_recent_bytes = int(max_bytes * _RECENT_BYTES_FRACTION)
        self._mismatches = 0
        self._scan_mismatches = 0

    @classmethod
    def create(
        cls, tokenizer: Any, *, max_bytes: int, verify: bool = False
    ) -> Optional["PromptTokenCache"]:
        """Returns a cache for ``tokenizer``, or None if it cannot be exact."""
        from tokenizers import Tokenizer as HfBackendTokenizer

        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not isinstance(backend, HfBackendTokenizer):
            reason = "not an HF fast tokenizer"
        else:
            spec = json.loads(backend.to_str())
            reason = _unsupported_reason(tokenizer, spec)
        if reason:
            logger.info("Prompt token cache disabled: %s.", reason)
            return None
        cache = cls(tokenizer, spec, max_bytes=max_bytes, verify=verify)
        if not cache._splits_exactly():
            logger.info(
                "Prompt token cache disabled: splitting at added tokens changes "
                "%s's token IDs.",
                type(tokenizer).__name__,
            )
            return None
        return cache

    def _splits_exactly(self) -> bool:
        """Checks that text split after an added token encodes as the whole,
        e.g. that the tokenizer's Python wrapper does not alter each text."""
        if not self._scanner.token_ids:
            return True
        token = next(iter(self._scanner.token_ids)).decode()
        for left, right in (("Hello", " world"), (" a", "b\n c"), ("x\n", "  1")):
            whole = self._tokenizer(left + token + right, add_special_tokens=False)
            first = self._tokenizer(left + token, add_special_tokens=False)
            rest = self._tokenizer(right, add_special_tokens=False)
            if whole["input_ids"] != first["input_ids"] + rest["input_ids"]:
                return False
        return True

    def _make_fast_backend(self) -> Optional[Any]:
        """A private copy of the backend tokenizer that encodes with
        ``encode_batch_fast`` (no offsets: ~15% faster), or None.

        Nothing changes its truncation or padding, so threads can share it.
        Like ``encode_batch`` (which the tokenizer calls), and unlike
        ``encode``, it releases the GIL. Used only if it encodes probe texts
        exactly as the tokenizer does.
        """
        backend = self._tokenizer.backend_tokenizer
        if not hasattr(backend, "encode_batch_fast"):
            return None
        try:
            fast = type(backend).from_str(backend.to_str())
            fast.no_truncation()
            fast.no_padding()
            fast.encode_special_tokens = False
        except Exception:
            return None
        tokens = [t.decode() for t in self._scanner.token_ids][:4]
        probes = [
            "Hello, world! 123 4567",
            " a\n\n  b\t\r\n c  ",
            "你好世界 こんにちは 🚀 é ﬁ １２３ ½",
            "def f(x):\n    return x <y> && z\n",
        ] + [f"x {t}y\n{t}{t} {t}" for t in tokens]
        for probe in probes:
            if (
                fast.encode_batch_fast([probe], add_special_tokens=False)[0].ids
                != self._tokenizer(probe, add_special_tokens=False)["input_ids"]
            ):
                return None
        return fast

    def _encode_rest(self, rest: str, kwargs: Dict[str, Any]) -> List[int]:
        """``tokenizer(rest, **kwargs)["input_ids"]`` without special tokens:
        the IDs of the text after the reused prefix."""
        if self._fast_backend is not None:
            ids = self._fast_backend.encode_batch_fast(
                [rest], add_special_tokens=False
            )[0].ids
            max_length = kwargs.get("max_length")
            if not (
                kwargs.get("truncation")
                and max_length is not None
                and len(ids) > max_length
            ):
                return ids
            # The tokenizer would truncate them.
        # The same truncation kwargs keep the tokenizer's shared state unchanged.
        return self._tokenizer(rest, **{**kwargs, "add_special_tokens": False})[
            "input_ids"
        ]

    def _probe_special_affixes(self) -> Optional[Tuple[List[int], List[int]]]:
        """The IDs add_special_tokens=True puts before and after the text's, if
        the same for any text (as for TemplateProcessing's single sequence)."""
        affixes = None
        for probe in ("a", "Hello, world! 123"):
            with_special = self._tokenizer(probe, add_special_tokens=True)["input_ids"]
            plain = self._tokenizer(probe, add_special_tokens=False)["input_ids"]
            found = None
            for i in range(len(with_special) - len(plain) + 1):
                if with_special[i : i + len(plain)] == plain:
                    found = (with_special[:i], with_special[i + len(plain) :])
                    break
            if found is None or (affixes is not None and found != affixes):
                return None
            affixes = found
        return affixes

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "prompts": len(self._lru),
                "keys": len(self._index),
                "bytes": self._bytes,
                "sessions": len(self._recent),
                "session_bytes": self._recent_bytes,
                "mismatches": self._mismatches,
            }

    def encode(
        self,
        text: str,
        kwargs: Dict[str, Any],
        record: Optional[Dict] = None,
        session: Optional[str] = None,
    ) -> Optional[List[int]]:
        """``tokenizer(text, **kwargs)["input_ids"]`` from cached prefixes.

        Returns None when it cannot be reproduced exactly; the caller then
        encodes as usual. Sets ``tok_reused`` and ``tok_encoded`` (and
        ``tok_mismatch`` when verifying) on the trace ``record``.

        ``session`` (e.g. a session ID) names the conversation that ``text``
        continues: the cache then compares ``text`` with the session's latest
        prompt, and scans and hashes only what it adds. Any value is exact.
        """
        result = self.encode_array(text, kwargs, record, session)
        return None if result is None else result[0]

    def encode_array(
        self,
        text: str,
        kwargs: Dict[str, Any],
        record: Optional[Dict] = None,
        session: Optional[str] = None,
    ) -> Optional[Tuple[List[int], Optional[np.ndarray]]]:
        """``encode``, also returning the IDs as a uint32 array (or None)."""
        result = self._encode(text, kwargs, session)
        if result is None:
            return None
        input_ids, array, reused, encoded, store = result
        if record is not None:
            record["tok_reused"] = reused
            record["tok_encoded"] = encoded
        if self._verify:
            expected = self._tokenizer(text, **kwargs)["input_ids"]
            # The list and the array come from separate copies of the prefix.
            got = input_ids
            if got == expected and array is not None:
                got = array.tolist()
            mismatch = got != expected
            if record is not None:
                record["tok_mismatch"] = mismatch
            if mismatch:
                self._log_mismatch(got, expected, reused)
                return expected, None
        if store is not None:
            store()
        return input_ids, array

    def _encode(self, text: str, kwargs: Dict[str, Any], session: Optional[str] = None):
        """Returns the input IDs as a list and as a uint32 array, the numbers
        of reused and newly encoded tokens, and a callable that stores the
        prompt (or None)."""
        if not kwargs.keys() <= _SUPPORTED_ENCODE_KWARGS:
            return None
        add_special = kwargs.get("add_special_tokens", True)
        if add_special and self._special_affixes is None:
            return None
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:  # Lone surrogates.
            return None

        scanned = None
        if session is not None:
            with self._lock:
                recent = self._recent.get(session)
            if recent is not None:
                scanned = self._resume(recent, data)
        if scanned is None:
            ends, boundary_ids = self._scanner.scan(data)
            digests, full_digest, states = self._digests(data, ends)
        else:
            ends, boundary_ids, digests, full_digest, states = scanned

        # The longest stored prefix: the whole text, else the last boundary.
        # The first ``shared`` boundaries end within it.
        with self._lock:
            index = self._index
            hit, cut, shared = index.get(full_digest), len(data), len(ends)
            if hit is None:
                for i in range(len(digests) - 1, -1, -1):
                    hit = index.get(digests[i])
                    if hit is not None:
                        cut, shared = ends[i], i + 1
                        break
            if hit is not None:
                self._lru.move_to_end(hit[0])

        if hit is None:
            prefix = np.empty(0, dtype=np.uint32)
            rest = text
            shared = 0
        else:
            prefix = hit[0].tokens[: hit[1]]
            rest = data[cut:].decode("utf-8")
        # The rest's pieces tokenize as they would within the whole text.
        encoded = self._encode_rest(rest, kwargs) if rest else []
        new = np.asarray(encoded, dtype=np.uint32)
        tokens = np.concatenate((prefix, new)) if len(prefix) else new
        ids = self._id_list(hit, new)
        input_ids = ids
        array = tokens
        if add_special:
            before, after = self._special_affixes
            if before or after:
                input_ids = before + input_ids + after
                array = np.concatenate(
                    (
                        np.asarray(before, dtype=np.uint32),
                        tokens,
                        np.asarray(after, dtype=np.uint32),
                    )
                )
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(input_ids) >= max_length
        ):
            # Possibly truncated: exact only as the tokenizer's own encode.
            if len(prefix) or len(input_ids) > max_length:
                return None
            return input_ids, array, 0, len(encoded), None
        if not rest:
            return input_ids, array, len(prefix), 0, None  # Already stored.

        # The scan must agree with the tokenizer on every boundary token, or
        # the prefixes this prompt reused or would store may be wrong. Those
        # in the reused prefix agreed when its prompt was stored: up to the
        # end of the prefix, right after a boundary token in both prompts,
        # the scans of both find the same tokens.
        positions = np.flatnonzero(self._is_boundary_id[new])
        if len(positions) != len(ends) - shared or not np.array_equal(
            new[positions], boundary_ids[shared:]
        ):
            self._scan_mismatches += 1
            if self._scan_mismatches <= _MAX_LOGGED_MISMATCHES:
                logger.warning(
                    "Prompt token cache: added tokens found in the text (%d) "
                    "differ from those in its token IDs (%d); not caching it.",
                    len(ends) - shared,
                    len(positions),
                )
            if len(prefix):
                return None
            return input_ids, array, 0, len(encoded), None

        counts = positions + (len(prefix) + 1)
        if shared:
            counts = np.concatenate((hit[0].counts[:shared], counts))
        keys = digests + ([full_digest] if not ends or ends[-1] != len(data) else [])
        key_counts = counts.tolist() + [len(tokens)] * (len(keys) - len(ends))
        superseded = None
        if hit is not None and len(hit[0].tokens) - len(prefix) <= (
            _SUPERSEDED_TAIL_TOKENS
        ):
            superseded = hit[0]
        recent = None
        if session is not None:
            recent = _RecentPrompt(data, ends, boundary_ids, digests, states)
        # Tuples, which only the cache holds (the caller may change ``ids``)
        # and the garbage collector stops tracking once it looks at them.
        store = functools.partial(
            self._store,
            _Entry(tokens, self._id_chunks(hit, ids), keys, counts),
            key_counts,
            superseded,
            session,
            recent,
        )
        return input_ids, array, len(prefix), len(encoded), store

    def _id_list(self, hit: Optional[Tuple[_Entry, int]], new: np.ndarray) -> List[int]:
        """The IDs of the ``hit`` prefix (if any) followed by ``new``, as a new
        list of the shared ints: the prefix's copied from its stored chunks,
        ``new``'s looked up by ID."""
        ids: List[int] = []
        if hit is not None:
            chunks = hit[0].chunks
            full, part = divmod(hit[1], _ID_CHUNK)
            for chunk in chunks[:full]:
                ids += chunk
            if part:
                ids += chunks[full][:part]
        if len(new):
            ids += self._ints.take(new).tolist()
        return ids

    @staticmethod
    def _id_chunks(
        hit: Optional[Tuple[_Entry, int]], ids: List[int]
    ) -> Tuple[Tuple[int, ...], ...]:
        """``ids`` (``_id_list``'s) in chunks, sharing the full chunks of the
        ``hit`` prefix."""
        shared = hit[0].chunks[: hit[1] // _ID_CHUNK] if hit is not None else ()
        return shared + tuple(
            tuple(ids[i : i + _ID_CHUNK])
            for i in range(len(shared) * _ID_CHUNK, len(ids), _ID_CHUNK)
        )

    def _resume(self, recent: _RecentPrompt, data: bytes):
        """The boundaries, digests and kept hash states of ``data``, scanning
        and hashing it only from the last of ``recent``'s kept boundaries that
        it shares, and taking the earlier ones from ``recent``. None if it
        shares none of them.

        ``data`` shares a boundary of ``recent`` that ends at ``end`` if both
        texts agree on their first ``end + max_token_bytes`` bytes. Their scans
        then find the same tokens up to ``end`` (each match depends on at most
        the next ``max_token_bytes`` bytes), and the scan of ``data`` resumes
        at ``end``.
        """
        lookahead = self._scanner.max_token_bytes
        old = memoryview(recent.data)
        limit = min(len(data), len(old)) - lookahead
        for i, state in reversed(recent.states):
            start = recent.ends[i]
            if start <= limit and data.startswith(old[: start + lookahead]):
                break
        else:
            return None
        ends, ids = self._scanner.scan(data, start)
        digests, full_digest, states = self._digests(
            data, ends, start, state.copy(), i + 1
        )
        n = i + 1
        kept = [s for s in recent.states if s[0] < n]
        return (
            recent.ends[:n] + ends,
            recent.ids[:n] + ids,
            recent.digests[:n] + digests,
            full_digest,
            (kept + states)[-_RECENT_STATES:],
        )

    @staticmethod
    def _digests(
        data: bytes,
        ends: List[int],
        start: int = 0,
        hasher: Optional[Any] = None,
        first: int = 0,
    ) -> Tuple[List[bytes], bytes, List[Tuple[int, Any]]]:
        """Hashes of ``data[:end]`` for each end, and of all of ``data``.

        ``hasher`` holds the hash of ``data[:start]``, if ``start`` is not 0.
        Also returns the hash states at the last ``_RECENT_STATES`` ends, with
        their boundary index (counting the first end as ``first``).
        """
        view = memoryview(data)
        h = hasher if hasher is not None else hashlib.blake2b(digest_size=_DIGEST_SIZE)
        digests = []
        states: "deque[Tuple[int, Any]]" = deque(maxlen=_RECENT_STATES)
        for i, end in enumerate(ends, first):
            h.update(view[start:end])
            state = h.copy()
            digests.append(state.digest())
            states.append((i, state))
            start = end
        h.update(view[start:])
        return digests, h.digest(), list(states)

    def _store(
        self,
        entry: _Entry,
        counts: List[int],
        superseded: Optional[_Entry],
        session: Optional[str] = None,
        recent: Optional[_RecentPrompt] = None,
    ) -> None:
        if entry.nbytes > self._max_bytes:
            return
        with self._lock:
            if session is not None:
                previous = self._recent.pop(session, None)
                if previous is not None:
                    self._recent_bytes -= previous.nbytes
                if recent is not None and recent.nbytes <= self._max_recent_bytes:
                    self._recent[session] = recent
                    self._recent_bytes += recent.nbytes
                    while self._recent_bytes > self._max_recent_bytes:
                        _, dropped = self._recent.popitem(last=False)
                        self._recent_bytes -= dropped.nbytes
            index = self._index
            for key, count in zip(entry.keys, counts):
                old = index.get(key)
                if old is not None:
                    old_entry = old[0]
                    old_entry.live -= 1
                    if old_entry.live == 0 and old_entry in self._lru:
                        # A newer prompt now holds every prefix it held.
                        del self._lru[old_entry]
                        self._bytes -= old_entry.nbytes
                index[key] = (entry, count)
                entry.live += 1
            if superseded is not None and superseded in self._lru:
                self._evict(superseded)
            self._lru[entry] = None
            self._bytes += entry.nbytes
            while self._bytes > self._max_bytes:
                self._evict(next(iter(self._lru)))

    def _evict(self, entry: _Entry) -> None:
        del self._lru[entry]
        self._bytes -= entry.nbytes
        index = self._index
        for key in entry.keys:
            value = index.get(key)
            if value is not None and value[0] is entry:
                del index[key]

    def _log_mismatch(self, got: List[int], expected: List[int], reused: int) -> None:
        self._mismatches += 1
        if self._mismatches > _MAX_LOGGED_MISMATCHES and self._mismatches % 1000:
            return
        first = next(
            (i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
            min(len(got), len(expected)),
        )
        logger.warning(
            "Prompt token cache mismatch #%d: %d cached vs %d expected tokens, "
            "%d reused, first difference at token %d (cached %s, expected %s).",
            self._mismatches,
            len(got),
            len(expected),
            reused,
            first,
            got[first : first + 8],
            expected[first : first + 8],
        )


class CachedPromptTokenizer:
    """Routes a vLLM renderer's prompt tokenization through a PromptTokenCache.

    Replaces ``_tokenize_prompt`` and ``_tokenize_prompt_async`` on one renderer
    instance only, so other renderers (e.g. an engine's) are unaffected.
    """

    def __init__(self, renderer: Any, cache: PromptTokenCache):
        self._renderer = renderer
        self._tokenizer = renderer.get_tokenizer()
        self.cache = cache
        self._original = renderer._tokenize_prompt
        self._executor = renderer._executor
        renderer._tokenize_prompt = self._tokenize_prompt
        renderer._tokenize_prompt_async = self._tokenize_prompt_async

    def _tokenize_prompt(self, prompt: Dict[str, Any], params: Any) -> Dict[str, Any]:
        return self.tokenize_prompt(
            prompt, params, kv_trace.current.get(), _request_scope.get()
        )

    def _tokenize_prompt_async(self, prompt: Dict[str, Any], params: Any):
        # The pool thread has no context: capture the trace record and the
        # request scope here.
        record = kv_trace.current.get()
        scope = _request_scope.get()
        return asyncio.get_running_loop().run_in_executor(
            self._executor,
            functools.partial(self.tokenize_prompt, prompt, params, record, scope),
        )

    def tokenize_prompt(
        self,
        prompt: Dict[str, Any],
        params: Any,
        record: Optional[Dict] = None,
        scope: Optional[RequestScope] = None,
    ) -> Dict[str, Any]:
        """``BaseRenderer._tokenize_prompt`` with the prompt token cache."""
        renderer = self._renderer
        result = None
        if renderer.get_tokenizer() is self._tokenizer and not (
            renderer._wants_offsets(prompt, params)
        ):
            result = self.cache.encode_array(
                prompt["prompt"],
                params.get_encode_kwargs(),
                record,
                scope.session_id if scope is not None else None,
            )
        if result is None:
            result = self._original(prompt, params)
            if record is not None:
                record["tok_reused"] = 0
                record["tok_encoded"] = len(result["prompt_token_ids"])
            return result
        input_ids, array = result
        # _build_tokens_prompt copies the IDs into a new list; the cache's
        # list is already new, so put it in place of the copy. The scope can
        # then recognize it by identity.
        tokens_prompt = renderer._build_tokens_prompt((), prompt)
        tokens_prompt["prompt_token_ids"] = input_ids
        if scope is not None and array is not None:
            scope.put(input_ids, array)
        return tokens_prompt


def install_prompt_token_cache(
    renderer: Any, *, max_bytes: int, verify: bool = False
) -> Optional[CachedPromptTokenizer]:
    """Caches prompt tokenization on ``renderer`` if it can be done exactly."""
    from vllm.renderers.base import BaseRenderer

    if type(renderer)._tokenize_prompt is not BaseRenderer._tokenize_prompt:
        logger.info(
            "Prompt token cache disabled: %s overrides _tokenize_prompt.",
            type(renderer).__name__,
        )
        return None
    try:
        tokenizer = renderer.get_tokenizer()
    except ValueError:  # skip_tokenizer_init
        return None
    cache = PromptTokenCache.create(tokenizer, max_bytes=max_bytes, verify=verify)
    if cache is None:
        return None
    logger.info(
        "Prompt token cache enabled (max %d bytes%s).",
        max_bytes,
        ", verifying every prompt" if verify else "",
    )
    return CachedPromptTokenizer(renderer, cache)
