import json

import httpx

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
    cold_texts = {t for t in user_texts if t.startswith("cold-")}
    assert len(cold_texts) == 2
    assert all(r["prompt_tokens"] == 1234 and r["tok_s"] > 0 for r in out["prefill"])


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
