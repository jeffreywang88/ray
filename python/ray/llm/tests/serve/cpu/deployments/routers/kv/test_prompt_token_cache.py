"""Tests for the router's prefix-cached prompt tokenization.

The cache must return exactly the token IDs of a full encode: KV-aware routing
scores replicas on prompt prefix overlap, so a divergence silently mis-routes.
"""

import gc
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from transformers import AutoTokenizer

from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.routing_policies.kv_aware import trace as kv_trace
from ray.llm._internal.serve.routing_policies.kv_aware.vllm import prompt_token_cache
from ray.llm._internal.serve.routing_policies.kv_aware.vllm.prompt_token_cache import (
    PromptTokenCache,
    RequestScope,
    _BoundaryScanner,
    _TokenMatcher,
    _unsupported_reason,
)
from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import Tokenizer

MODEL = "Qwen/Qwen3-0.6B"
CHAT = {"add_special_tokens": False, "truncation": True, "max_length": 40961}

PIECES = [
    "hello world",
    " the quick brown fox",
    "def f(x):\n    return x + 1\n",
    '{"a": [1, 2, 3]}',
    "<div>a < b > c</div>",
    "你好世界",
    "こんにちは",
    "🚀🔥",
    "é",
    "ﬁ",
    "１２３",
    "12345678",
    " ",
    "\n",
    "\n\n",
    "\t",
    "  \n ",
    "",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "<|endoftext|>",
    "<|im_",
    "<think",
    "|>",
    "user\n",
    "assistant\n",
]


def _random_text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(PIECES) for _ in range(n))


@pytest.fixture(scope="module")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


def _cache(hf_tokenizer, **kwargs) -> PromptTokenCache:
    cache = PromptTokenCache.create(
        hf_tokenizer, max_bytes=kwargs.pop("max_bytes", 1 << 30), **kwargs
    )
    assert cache is not None
    return cache


def _reference_matches(data: bytes, tokens: dict):
    """Leftmost-longest non-overlapping matches, by brute force."""
    out = []
    i = 0
    while i < len(data):
        for start in range(i, len(data)):
            found = [t for t in tokens if data.startswith(t, start)]
            if found:
                token = max(found, key=len)
                out.append((start, start + len(token), tokens[token]))
                i = start + len(token)
                break
        else:
            break
    return out


class TestBoundaryScanner:
    # Overlapping tokens with shared prefixes and different first bytes.
    TOKENS = [b"ab", b"abc", b"b", b"bca", b"ca", b"<x>", b"<xy>", b"x>"]

    @pytest.mark.parametrize("seed", range(20))
    def test_matcher_is_leftmost_longest(self, seed):
        rng = random.Random(seed)
        tokens = {t: i for i, t in enumerate(rng.sample(self.TOKENS, 5))}
        matcher = _TokenMatcher(tokens)
        for _ in range(50):
            data = bytes(rng.choice(b"abcxy<> ") for _ in range(rng.randint(0, 40)))
            assert matcher.find(data, 0, len(data)) == _reference_matches(data, tokens)

    @pytest.mark.parametrize("seed", range(20))
    def test_normalized_tokens_match_between_raw_ones(self, seed):
        rng = random.Random(seed)
        names = rng.sample(self.TOKENS, 6)
        added = [
            {"id": i, "content": t.decode(), "normalized": i % 2 == 1}
            for i, t in enumerate(names)
        ]
        raw = {t["content"].encode(): t["id"] for t in added if not t["normalized"]}
        normalized = {t["content"].encode(): t["id"] for t in added if t["normalized"]}
        scanner = _BoundaryScanner(added, identity_normalizer=True)
        for _ in range(50):
            data = bytes(rng.choice(b"abcxy<> ") for _ in range(rng.randint(0, 40)))
            expected = []
            gap_start = 0
            for match in _reference_matches(data, raw) + [(len(data), len(data), None)]:
                gap = data[gap_start : match[0]]
                expected += [
                    (s + gap_start, e + gap_start, i)
                    for s, e, i in _reference_matches(gap, normalized)
                ]
                if match[2] is not None:
                    expected.append(match)
                gap_start = match[1]
            assert scanner.scan(data) == (
                [e for _, e, _ in expected],
                [i for _, _, i in expected],
            )

    @pytest.mark.parametrize("seed", range(20))
    def test_scan_resumes_at_any_boundary(self, seed):
        """Scanning from the end of a boundary token finds the rest of the
        full scan, which lets a session's next prompt skip its shared prefix."""
        rng = random.Random(seed)
        names = rng.sample(self.TOKENS, 6)
        added = [
            {"id": i, "content": t.decode(), "normalized": i % 3 == 1}
            for i, t in enumerate(names)
        ]
        scanner = _BoundaryScanner(added, identity_normalizer=True)
        assert scanner.max_token_bytes == max(len(t) for t in names)
        for _ in range(50):
            data = bytes(rng.choice(b"abcxy<> ") for _ in range(rng.randint(0, 40)))
            ends, ids = scanner.scan(data)
            for k, end in enumerate(ends):
                assert scanner.scan(data, end) == (ends[k + 1 :], ids[k + 1 :])


class TestExactness:
    @pytest.mark.parametrize("seed", range(5))
    def test_growing_prompts(self, hf_tokenizer, seed):
        """Each prompt extends a cut of an earlier one, like the turns of
        agent sessions that share a system prompt."""
        rng = random.Random(seed)
        cache = _cache(hf_tokenizer)
        prompts = [_random_text(rng, 40)]
        reused = 0
        for _ in range(60):
            base = rng.choice(prompts)
            prompt = base[: rng.randint(0, len(base))] + _random_text(rng, 15)
            prompts.append(prompt)
            for kwargs in (CHAT, {"add_special_tokens": True}):
                record = {}
                ids = cache.encode(prompt, kwargs, record)
                assert ids == hf_tokenizer(prompt, **kwargs)["input_ids"]
                reused += record["tok_reused"]
                assert record["tok_reused"] + record["tok_encoded"] == len(ids)
        assert reused > 0

    @pytest.mark.parametrize("seed", range(5))
    def test_sessions_resume_their_latest_prompt(self, hf_tokenizer, seed):
        """Turns of interleaved sessions with session hints, including wrong
        hints (sessions sharing one), edits of earlier turns, shorter prompts
        and prompts ending in part of an added token, encode exactly. Most
        turns scan and hash only what they add to the session's latest one."""
        rng = random.Random(seed)
        cache = _cache(hf_tokenizer)
        resumed = []
        resume = cache._resume
        cache._resume = lambda *a: resumed.append(resume(*a)) or resumed[-1]
        system = "<|im_start|>system\n" + _random_text(rng, 30) + "<|im_end|>\n"
        sessions = {f"s{i}": system for i in range(4)}
        hints = {name: rng.choice([name, name, "shared"]) for name in sessions}
        for _ in range(80):
            name = rng.choice(list(sessions))
            history = sessions[name]
            turn = "<|im_start|>user\n" + _random_text(rng, 8) + "<|im_end|>\n"
            prompt = history + turn + "<|im_start|>assistant\n"
            r = rng.random()
            if r < 0.1:
                cut = rng.randint(0, len(prompt))
                prompt = prompt[:cut] + rng.choice(PIECES) + prompt[cut:]
            elif r < 0.2:
                prompt = prompt[: rng.randint(0, len(prompt))]
            elif r < 0.3:
                prompt += rng.choice(["<|im_", "<", "<think", "|>"])
            for kwargs in (CHAT, {"add_special_tokens": True}):
                record = {}
                result = cache.encode_array(prompt, kwargs, record, hints[name])
                expected = hf_tokenizer(prompt, **kwargs)["input_ids"]
                assert result[0] == expected
                assert result[1].dtype == np.uint32
                assert result[1].tolist() == expected
                assert record["tok_reused"] + record["tok_encoded"] == len(expected)
            sessions[name] = (
                history
                + turn
                + "<|im_start|>assistant\n"
                + (_random_text(rng, 8) + "<|im_end|>\n")
            )
        assert sum(r is not None for r in resumed) > len(resumed) // 2
        assert 0 < cache.stats()["sessions"] <= len(sessions)

    @pytest.mark.parametrize("chunk", [1, 7, 4096])
    @pytest.mark.parametrize("seed", range(3))
    def test_id_lists_share_ints_and_chunks(
        self, hf_tokenizer, monkeypatch, seed, chunk
    ):
        """The returned lists hold the cache's shared int objects, copied
        from a stored prompt's chunks whether the reused prefix is most of it
        or a small part of it; a session's next prompt shares its full chunks;
        changing a returned list changes no later result."""
        monkeypatch.setattr(prompt_token_cache, "_ID_CHUNK", chunk)
        rng = random.Random(seed)
        cache = _cache(hf_tokenizer)
        system = "<|im_start|>system\n" + _random_text(rng, 30) + "<|im_end|>\n"
        history = system
        previous = None
        for turn in range(12):
            history += "<|im_start|>user\n" + _random_text(rng, 10) + "<|im_end|>\n"
            # Every third prompt shares only the system prompt with the
            # session's history, so it reuses a short prefix of a long one.
            prompt = (system if turn % 3 == 2 else history) + "<|im_start|>assistant\n"
            for kwargs in (CHAT, {"add_special_tokens": True}):
                record = {}
                ids, array = cache.encode_array(prompt, kwargs, record, "s")
                expected = hf_tokenizer(prompt, **kwargs)["input_ids"]
                assert ids == expected
                assert array.tolist() == expected
                if kwargs is CHAT:
                    assert all(i is cache._ints[i] for i in ids)
                if turn:
                    assert record["tok_reused"] > 0
                # The caller owns the list.
                ids.reverse()
                ids[: len(ids) // 2] = [0] * (len(ids) // 2)
                ids.append(1)
            entry = next(reversed(cache._lru))
            assert [len(c) for c in entry.chunks[:-1]] == [chunk] * (
                len(entry.chunks) - 1
            )
            assert sum(map(len, entry.chunks)) == len(entry.tokens)
            if chunk < 4096 and turn % 3 == 1:
                # It extends the previous history prompt.
                assert entry.chunks[0] is previous.chunks[0]
            previous = entry
            history += "<|im_start|>assistant\n" + _random_text(rng, 8) + "<|im_end|>\n"
        # Untracks the chunks, then the tuples of them.
        gc.collect()
        gc.collect()
        for entry in cache._lru:
            assert not any(map(gc.is_tracked, (entry.chunks,) + entry.chunks))

    def test_session_prompts_memory_bounded(self, hf_tokenizer):
        rng = random.Random(2)
        cache = _cache(hf_tokenizer, max_bytes=40_000)
        for i in range(100):
            prompt = _random_text(rng, rng.randint(1, 120))
            ids = cache.encode(prompt, CHAT, session=f"s{i % 7}")
            assert ids == hf_tokenizer(prompt, **CHAT)["input_ids"]
            assert cache.stats()["session_bytes"] <= 10_000

    def test_fast_backend_matches_tokenizer(self, hf_tokenizer):
        cache = _cache(hf_tokenizer)
        assert cache._fast_backend is not None
        rng = random.Random(3)
        for _ in range(20):
            text = _random_text(rng, rng.randint(0, 80))
            assert (
                cache._encode_rest(text, CHAT)
                == hf_tokenizer(text, add_special_tokens=False)["input_ids"]
            )

    def test_concurrent_encodes(self, hf_tokenizer):
        rng = random.Random(0)
        cache = _cache(hf_tokenizer)
        system = _random_text(rng, 200)
        prompts = [system + _random_text(rng, 50) for _ in range(64)]
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda p: cache.encode(p, CHAT), prompts * 2))
        for prompt, ids in zip(prompts * 2, results):
            assert ids == hf_tokenizer(prompt, **CHAT)["input_ids"]

    def test_eviction_keeps_memory_bounded(self, hf_tokenizer):
        rng = random.Random(1)
        cache = _cache(hf_tokenizer, max_bytes=20_000)
        for _ in range(200):
            prompt = _random_text(rng, rng.randint(1, 120))
            assert (
                cache.encode(prompt, CHAT) == hf_tokenizer(prompt, **CHAT)["input_ids"]
            )
            assert cache.stats()["bytes"] <= 20_000

    def test_superseded_prompt_is_dropped(self, hf_tokenizer):
        """A session keeps one stored prompt, not one per turn."""
        cache = _cache(hf_tokenizer)
        prompt = "<|im_start|>system\nBe brief.<|im_end|>\n"
        for turn in range(10):
            prompt += f"<|im_start|>user\nquestion {turn}<|im_end|>\n"
            # The generation prompt the next turn replaces with the answer.
            assert cache.encode(prompt + "<|im_start|>assistant\n", CHAT)
            prompt += f"<|im_start|>assistant\nanswer {turn}<|im_end|>\n"
        assert cache.stats()["prompts"] == 1

    def test_truncation_falls_back(self, hf_tokenizer):
        cache = _cache(hf_tokenizer)
        prompt = "<|im_start|>user\n" + "word " * 50 + "<|im_end|>"
        kwargs = {"add_special_tokens": False, "truncation": True, "max_length": 20}
        # Cold, the cache encodes exactly as the tokenizer, truncation included.
        assert (
            cache.encode(prompt, kwargs) == hf_tokenizer(prompt, **kwargs)["input_ids"]
        )
        cache.encode(prompt, CHAT)
        # Reusing a prefix, it cannot truncate, so it leaves it to the tokenizer.
        assert cache.encode(prompt + "<|im_end|>more", kwargs) is None

    def test_unsupported_kwargs_fall_back(self, hf_tokenizer):
        cache = _cache(hf_tokenizer)
        assert (
            cache.encode("hi", {"add_special_tokens": False, "padding": True}) is None
        )

    @pytest.mark.parametrize("corrupt", ["tokens", "chunks"])
    def test_verify_returns_full_encode_on_mismatch(self, hf_tokenizer, corrupt):
        cache = _cache(hf_tokenizer, verify=True)
        prompt = "<|im_start|>user\nhello world<|im_end|>\n"
        record = {}
        assert (
            cache.encode(prompt, CHAT, record)
            == hf_tokenizer(prompt, **CHAT)["input_ids"]
        )
        assert record["tok_mismatch"] is False
        # Corrupt the stored prompt's array or chunks: the next prompt reuses
        # its prefix, as an array (sent to the engine) and as a list.
        (entry,) = cache._lru
        if corrupt == "tokens":
            entry.tokens[1] += 1
        else:
            (chunk,) = entry.chunks
            entry.chunks = (chunk[:1] + (chunk[1] + 1,) + chunk[2:],)
        longer = prompt + "<|im_start|>assistant\n"
        record = {}
        ids, array = cache.encode_array(longer, CHAT, record)
        assert ids == hf_tokenizer(longer, **CHAT)["input_ids"]
        assert array is None
        assert record["tok_mismatch"] is True
        assert record["tok_reused"] > 0
        assert cache.stats()["mismatches"] == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ({}, None),
        ({"added_tokens": [{"content": "<m>", "lstrip": True}]}, "strips"),
        ({"pre_tokenizer": {"type": "Metaspace"}}, "Metaspace"),
        (
            {"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True}},
            "add_prefix_space",
        ),
        ({"model": {"type": "BPE", "dropout": 0.1}}, "dropout"),
        ({"post_processor": {"type": "Custom"}}, "Custom"),
    ],
)
def test_unsupported_tokenizers(hf_tokenizer, change, reason):
    import json

    spec = {**json.loads(hf_tokenizer.backend_tokenizer.to_str()), **change}
    result = _unsupported_reason(hf_tokenizer, spec)
    if reason is None:
        assert result is None
    else:
        assert reason in result


# The renderer binds its async tokenizer to the running loop at first use, so
# the async tests share one loop (as the LLMRouter replica does).
@pytest.mark.asyncio(loop_scope="module")
async def test_router_tokenizer_sessions(hf_tokenizer):
    """Multi-turn chats through the router's Tokenizer match vLLM's
    uncached tokenization, and later turns reuse the earlier turns' tokens."""
    from vllm.platforms import current_platform

    if not current_platform.device_type:
        current_platform.device_type = "cpu"
    config = LLMConfig(
        model_loading_config=dict(model_id="test-model", model_source=MODEL),
        engine_kwargs=dict(max_model_len=40960, enforce_eager=True),
    )
    cached = Tokenizer(config, prompt_token_cache=True)
    plain = Tokenizer(config)
    assert cached._prompt_token_cache is not None
    assert plain._prompt_token_cache is None

    rng = random.Random(0)
    system = {"role": "system", "content": _random_text(rng, 30)}
    for session in range(3):
        messages = [system]
        for turn in range(6):
            messages.append({"role": "user", "content": _random_text(rng, 10)})
            payload = {"model": "test-model", "messages": messages}
            record = {}
            kv_trace.current.set(record)
            with RequestScope(session_id=f"session-{session}") as scope:
                ids = await cached.tokenize(
                    payload, request_path="/v1/chat/completions"
                )
            kv_trace.current.set(None)
            assert ids == await plain.tokenize(
                payload, request_path="/v1/chat/completions"
            )
            assert record["tok_reused"] + record["tok_encoded"] == len(ids)
            if turn or session:
                assert record["tok_reused"] > 0
            # The IDs come with the array the cache built them from.
            array = scope.token_array(ids)
            assert array is not None and array.tolist() == ids
            assert scope.token_array(list(ids)) is None
            messages.append({"role": "assistant", "content": _random_text(rng, 10)})


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
