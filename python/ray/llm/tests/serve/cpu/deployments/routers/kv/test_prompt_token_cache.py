"""Tests for the router's prefix-cached prompt tokenization.

The cache must return exactly the token IDs of a full encode: KV-aware routing
scores replicas on prompt prefix overlap, so a divergence silently mis-routes.
"""

import random
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from transformers import AutoTokenizer

from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.routing_policies.kv_aware import trace as kv_trace
from ray.llm._internal.serve.routing_policies.kv_aware.vllm.prompt_token_cache import (
    PromptTokenCache,
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

    def test_verify_returns_full_encode_on_mismatch(self, hf_tokenizer):
        cache = _cache(hf_tokenizer, verify=True)
        prompt = "<|im_start|>user\nhello world<|im_end|>\n"
        record = {}
        assert (
            cache.encode(prompt, CHAT, record)
            == hf_tokenizer(prompt, **CHAT)["input_ids"]
        )
        assert record["tok_mismatch"] is False
        # Corrupt the stored prompt: the next prompt reuses its prefix.
        (entry,) = cache._lru
        entry.tokens[1] += 1
        longer = prompt + "<|im_start|>assistant\n"
        record = {}
        assert (
            cache.encode(longer, CHAT, record)
            == hf_tokenizer(longer, **CHAT)["input_ids"]
        )
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
            ids = await cached.tokenize(payload, request_path="/v1/chat/completions")
            kv_trace.current.set(None)
            assert ids == await plain.tokenize(
                payload, request_path="/v1/chat/completions"
            )
            assert record["tok_reused"] + record["tok_encoded"] == len(ids)
            if turn or session:
                assert record["tok_reused"] > 0
            messages.append({"role": "assistant", "content": _random_text(rng, 10)})


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
