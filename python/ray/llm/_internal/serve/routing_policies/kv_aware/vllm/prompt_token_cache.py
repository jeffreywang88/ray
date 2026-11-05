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
"""

import asyncio
import functools
import hashlib
import json
import re
import threading
from collections import OrderedDict
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

    def scan(self, data: bytes) -> Tuple[List[int], List[int]]:
        """Returns the end offsets and token IDs of the boundary tokens in ``data``."""
        raw = self._raw.find(data, 0, len(data)) if self._raw is not None else []
        normalized = self._normalized
        if normalized is None:
            matches = raw
        else:
            # Normalized tokens are extracted from the pieces between raw ones.
            # Scanning the whole text finds the same unless a match overlaps a
            # raw token.
            matches = sorted(raw + normalized.find(data, 0, len(data)))
            if not _non_overlapping(matches):
                matches = []
                gap_start = 0
                for match in raw:
                    if match[0] > gap_start:
                        matches += normalized.find(data, gap_start, match[0])
                    matches.append(match)
                    gap_start = match[1]
                if gap_start < len(data):
                    matches += normalized.find(data, gap_start, len(data))
        return [m[1] for m in matches], [m[2] for m in matches]


class _Entry:
    """The token IDs of one stored prompt, and the index keys pointing to it."""

    __slots__ = ("tokens", "keys", "live", "nbytes")

    def __init__(self, tokens: np.ndarray, keys: List[bytes]):
        self.tokens = tokens
        self.keys = keys
        self.live = 0
        self.nbytes = tokens.nbytes + len(keys) * _INDEX_BYTES_PER_KEY


class PromptTokenCache:
    """Reuses the token IDs of earlier prompts' prefixes when tokenizing a prompt.

    Every prefix of a stored prompt that ends right after a boundary added
    token is indexed by the hash of its UTF-8 bytes, so a new prompt reuses its
    longest stored prefix and encodes only the rest. Stored prompts are evicted
    least recently used beyond ``max_bytes`` of tokens and index.

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
        size = max(
            [backend.get_vocab_size(with_added_tokens=True)]
            + [i + 1 for i in self._scanner.token_ids.values()]
        )
        self._is_boundary_id = np.zeros(size, dtype=bool)
        self._is_boundary_id[list(self._scanner.token_ids.values())] = True
        self._special_affixes = self._probe_special_affixes()
        self._lock = threading.Lock()
        self._index: Dict[bytes, Tuple[_Entry, int]] = {}
        self._lru: "OrderedDict[_Entry, None]" = OrderedDict()
        self._bytes = 0
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
                "mismatches": self._mismatches,
            }

    def encode(
        self, text: str, kwargs: Dict[str, Any], record: Optional[Dict] = None
    ) -> Optional[List[int]]:
        """``tokenizer(text, **kwargs)["input_ids"]`` from cached prefixes.

        Returns None when it cannot be reproduced exactly; the caller then
        encodes as usual. Sets ``tok_reused`` and ``tok_encoded`` (and
        ``tok_mismatch`` when verifying) on the trace ``record``.
        """
        result = self._encode(text, kwargs)
        if result is None:
            return None
        input_ids, reused, encoded, store = result
        if record is not None:
            record["tok_reused"] = reused
            record["tok_encoded"] = encoded
        if self._verify:
            expected = self._tokenizer(text, **kwargs)["input_ids"]
            mismatch = input_ids != expected
            if record is not None:
                record["tok_mismatch"] = mismatch
            if mismatch:
                self._log_mismatch(input_ids, expected, reused)
                return expected
        if store is not None:
            store()
        return input_ids

    def _encode(self, text: str, kwargs: Dict[str, Any]):
        """Returns the input IDs, the numbers of reused and newly encoded
        tokens, and a callable that stores the prompt (or None)."""
        if not kwargs.keys() <= _SUPPORTED_ENCODE_KWARGS:
            return None
        add_special = kwargs.get("add_special_tokens", True)
        if add_special and self._special_affixes is None:
            return None
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError:  # Lone surrogates.
            return None

        ends, boundary_ids = self._scanner.scan(data)
        digests, full_digest = self._digests(data, ends)

        # The longest stored prefix: the whole text, else the last boundary.
        with self._lock:
            index = self._index
            hit, cut = index.get(full_digest), len(data)
            if hit is None:
                for i in range(len(digests) - 1, -1, -1):
                    hit = index.get(digests[i])
                    if hit is not None:
                        cut = ends[i]
                        break
            if hit is not None:
                self._lru.move_to_end(hit[0])

        if hit is None:
            prefix = np.empty(0, dtype=np.uint32)
            rest = text
        else:
            prefix = hit[0].tokens[: hit[1]]
            rest = data[cut:].decode("utf-8")
        # The rest's pieces tokenize as they would within the whole text. The
        # same truncation kwargs keep the tokenizer's shared state unchanged.
        encoded = (
            self._tokenizer(rest, **{**kwargs, "add_special_tokens": False})[
                "input_ids"
            ]
            if rest
            else []
        )
        tokens = (
            np.concatenate((prefix, np.asarray(encoded, dtype=np.uint32)))
            if len(prefix)
            else np.asarray(encoded, dtype=np.uint32)
        )
        input_ids = prefix.tolist() + encoded if len(prefix) else encoded
        if add_special:
            before, after = self._special_affixes
            input_ids = before + input_ids + after
        max_length = kwargs.get("max_length")
        if (
            kwargs.get("truncation")
            and max_length is not None
            and len(input_ids) >= max_length
        ):
            # Possibly truncated: exact only as the tokenizer's own encode.
            if len(prefix) or len(input_ids) > max_length:
                return None
            return input_ids, 0, len(encoded), None
        if not rest:
            return input_ids, len(prefix), 0, None  # Already stored.

        # The scan must agree with the tokenizer on every boundary token, or
        # the prefixes this prompt reused or would store may be wrong.
        positions = np.flatnonzero(self._is_boundary_id[tokens])
        if len(positions) != len(ends) or not np.array_equal(
            tokens[positions], boundary_ids
        ):
            self._scan_mismatches += 1
            if self._scan_mismatches <= _MAX_LOGGED_MISMATCHES:
                logger.warning(
                    "Prompt token cache: added tokens found in the text (%d) "
                    "differ from those in its token IDs (%d); not caching it.",
                    len(ends),
                    len(positions),
                )
            if len(prefix):
                return None
            return input_ids, 0, len(encoded), None

        keys = digests + ([full_digest] if not ends or ends[-1] != len(data) else [])
        counts = (positions + 1).tolist() + [len(tokens)] * (len(keys) - len(ends))
        superseded = None
        if hit is not None and len(hit[0].tokens) - len(prefix) <= (
            _SUPERSEDED_TAIL_TOKENS
        ):
            superseded = hit[0]
        store = functools.partial(self._store, _Entry(tokens, keys), counts, superseded)
        return input_ids, len(prefix), len(encoded), store

    @staticmethod
    def _digests(data: bytes, ends: List[int]) -> Tuple[List[bytes], bytes]:
        """Hashes of ``data[:end]`` for each end, and of all of ``data``."""
        view = memoryview(data)
        h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        digests = []
        start = 0
        for end in ends:
            h.update(view[start:end])
            digests.append(h.copy().digest())
            start = end
        h.update(view[start:])
        return digests, h.digest()

    def _store(
        self, entry: _Entry, counts: List[int], superseded: Optional[_Entry]
    ) -> None:
        if entry.nbytes > self._max_bytes:
            return
        with self._lock:
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
        return self.tokenize_prompt(prompt, params, kv_trace.current.get())

    def _tokenize_prompt_async(self, prompt: Dict[str, Any], params: Any):
        # The pool thread has no context: capture the trace record here.
        record = kv_trace.current.get()
        return asyncio.get_running_loop().run_in_executor(
            self._executor,
            functools.partial(self.tokenize_prompt, prompt, params, record),
        )

    def tokenize_prompt(
        self, prompt: Dict[str, Any], params: Any, record: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """``BaseRenderer._tokenize_prompt`` with the prompt token cache."""
        renderer = self._renderer
        input_ids = None
        if renderer.get_tokenizer() is self._tokenizer and not (
            renderer._wants_offsets(prompt, params)
        ):
            input_ids = self.cache.encode(
                prompt["prompt"], params.get_encode_kwargs(), record
            )
        if input_ids is None:
            result = self._original(prompt, params)
            if record is not None:
                record["tok_reused"] = 0
                record["tok_encoded"] = len(result["prompt_token_ids"])
            return result
        return renderer._build_tokens_prompt(input_ids, prompt)


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
