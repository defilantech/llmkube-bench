import json

import httpx
import pytest

from harness import ladder, oai


def sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def fake_server(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        n = body["max_tokens"]
        events = [{"choices": [{"delta": {"content": "x"}}]} for _ in range(n)]
        events.append({"choices": [], "usage": {"prompt_tokens": 1234, "completion_tokens": n}})
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def fake_server_no_usage():
    """A server that streams content but never sends a usage block (dropped final frame)."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = body["max_tokens"]
        events = [{"choices": [{"delta": {"content": "x"}}]} for _ in range(n)]
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_uses_server_usage_as_truth():
    seen = []
    res = oai.chat(fake_server(seen), "http://x", "m", [{"role": "user", "content": "hi"}], max_tokens=3)
    assert res.prompt_tokens == 1234 and res.completion_tokens == 3
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}
    assert res.ttft_s <= res.total_s


def test_chat_without_usage_reports_none():
    def handler(request):
        return httpx.Response(200, text=sse([{"choices": [{"delta": {"content": "x"}}]}]),
                              headers={"content-type": "text/event-stream"})
    res = oai.chat(httpx.Client(transport=httpx.MockTransport(handler)), "http://x", "m", [], max_tokens=1)
    assert res.prompt_tokens is None


def test_chat_reads_delta_reasoning_field():
    """Newer vLLM streams the reasoning trace as delta.reasoning, not delta.reasoning_content."""
    events = [{"choices": [{"delta": {"reasoning": "thinking "}}]},
              {"choices": [{"delta": {"reasoning": "more"}}]},
              {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}]

    def handler(request):
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    res = oai.chat(httpx.Client(transport=httpx.MockTransport(handler)), "http://x", "m", [], max_tokens=2)
    assert res.text == "thinking more"
    assert res.completion_tokens == 2


class CountTok:
    """Whitespace tokenizer standing in for tokenizers.Tokenizer."""
    def encode(self, text):
        class E:
            ids = text.split()
        return E()

    def decode(self, ids):
        return " ".join(ids)


def test_build_prompt_hits_target_and_starts_with_nonce():
    p = ladder.build_prompt(CountTok(), 50, "nonce-1", corpus="a b c d e f g")
    assert p.startswith("nonce-1")
    assert len(p.split()) == 50


def test_cold_prompts_are_unique_and_cached_prompts_repeat():
    seen = []
    client = fake_server(seen)
    out = ladder.run_ladder(client, "http://x", "m", CountTok(), corpus="w " * 100, sizes=[20],
                            repeats=2, decode_prompt_tokens=10, decode_max_tokens=4, concurrencies=[1, 2])
    user_texts = [b["messages"][-1]["content"] for b in seen]
    cold = [r for r in out["prefill"] if r["mode"] == "cold"]
    cached = [r for r in out["prefill"] if r["mode"] == "cached"]
    assert len(cold) == 2 and len(cached) == 2
    cold_texts = [t for t in user_texts if t.startswith("cold-")]
    cached_texts = [t for t in user_texts if t.startswith("cached-")]
    assert len(set(cold_texts)) == 2
    # each cached rung sends the same text twice (prime, then timed) in order
    assert len(cached_texts) == 4
    assert cached_texts[0] == cached_texts[1]
    assert cached_texts[2] == cached_texts[3]
    assert set(cached_texts).isdisjoint(cold_texts)
    assert all(r["prompt_tokens"] == 1234 and r["tok_s"] > 0 and r["valid"] is True for r in out["prefill"])


def test_missing_usage_marks_rows_invalid():
    out = ladder.run_ladder(fake_server_no_usage(), "http://x", "m", CountTok(), corpus="w " * 100, sizes=[20],
                            repeats=1, decode_prompt_tokens=10, decode_max_tokens=4, concurrencies=[1])
    assert out["prefill"] and all(r["valid"] is False and r["tok_s"] is None for r in out["prefill"])
    assert all(r["target_error"] is None for r in out["prefill"])
    d = out["decode"][0]
    assert all(rate is None for rate in d["per_stream_tok_s"])
    assert d["mean_per_stream_tok_s"] is None
    assert d["invalid_streams"] == len(d["per_stream_tok_s"])


def test_build_prompt_rejects_empty_corpus():
    with pytest.raises(ValueError):
        ladder.build_prompt(CountTok(), 50, "nonce-1", corpus="")


def test_every_rung_and_decode_run_is_preceded_by_a_warmup():
    seen = []
    ladder.run_ladder(fake_server(seen), "http://x", "m", CountTok(), corpus="w " * 100, sizes=[20, 30],
                      repeats=1, decode_prompt_tokens=10, decode_max_tokens=4, concurrencies=[1])
    kinds = ["warmup" if b["messages"][-1]["content"].startswith("warmup") else "work" for b in seen]
    # sizes [20, 30] plus one decode concurrency = 3 blocks, each opening with a warmup
    assert kinds.count("warmup") == 3 and kinds[0] == "warmup"


def test_decode_reports_per_stream_rates():
    out = ladder.run_ladder(fake_server([]), "http://x", "m", CountTok(), corpus="w " * 100, sizes=[],
                            repeats=1, decode_prompt_tokens=10, decode_max_tokens=8, concurrencies=[2])
    d = out["decode"][0]
    assert d["concurrency"] == 2 and len(d["per_stream_tok_s"]) == 2


def test_decode_rate_counts_the_first_token_at_ttft():
    r = oai.ChatResult(text="x" * 5, prompt_tokens=10, completion_tokens=5, ttft_s=0.1, total_s=0.5)
    invalid, rate = ladder._decode_row(r)
    assert invalid is False
    assert rate == pytest.approx((5 - 1) / (0.5 - 0.1))


def test_decode_row_invalid_when_completion_tokens_under_two():
    r = oai.ChatResult(text="x", prompt_tokens=10, completion_tokens=1, ttft_s=0.1, total_s=0.5)
    invalid, rate = ladder._decode_row(r)
    assert invalid is True and rate is None


def test_decode_row_invalid_when_ttft_never_arrived():
    # No token was ever seen: ttft falls back to total_s (see oai.chat), so ttft_s >= total_s.
    r = oai.ChatResult(text="", prompt_tokens=10, completion_tokens=5, ttft_s=0.5, total_s=0.5)
    invalid, rate = ladder._decode_row(r)
    assert invalid is True and rate is None


def test_decode_marks_a_stream_with_no_token_seen_invalid_end_to_end():
    """A server that reports usage but never streams any content or reasoning delta: the client
    never saw a token, so ttft_s falls back to total_s and the stream must be invalid, not a
    near-infinite rate."""
    def handler(request):
        events = [{"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = ladder.run_ladder(client, "http://x", "m", CountTok(), corpus="w " * 100, sizes=[],
                            repeats=1, decode_prompt_tokens=10, decode_max_tokens=5, concurrencies=[1])
    d = out["decode"][0]
    assert d["per_stream_tok_s"] == [None]
    assert d["invalid_streams"] == 1
    assert d["mean_per_stream_tok_s"] is None


def test_decode_marks_a_single_token_stream_invalid_end_to_end():
    def handler(request):
        events = [{"choices": [{"delta": {"content": "x"}}]},
                  {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}]
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = ladder.run_ladder(client, "http://x", "m", CountTok(), corpus="w " * 100, sizes=[],
                            repeats=1, decode_prompt_tokens=10, decode_max_tokens=1, concurrencies=[1])
    d = out["decode"][0]
    assert d["per_stream_tok_s"] == [None]
    assert d["invalid_streams"] == 1
