import json

import httpx

from harness import soak
from harness.soak import classify_reply, next_turn_messages


def sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def test_degenerate_reply_is_caught():
    assert classify_reply("x" * 10 + "/" * 200) == "degenerate"
    assert classify_reply("abc " * 200) == "degenerate"
    assert classify_reply("def f(x):\n    return x + 1\n") == "ok"
    assert classify_reply("") == "empty"


def test_turns_grow_the_prompt_and_stop_at_the_budget():
    msgs = [{"role": "system", "content": "s"}]
    msgs = next_turn_messages(msgs, assistant="a1", user="u2", budget_chars=10_000)
    assert [m["role"] for m in msgs][-2:] == ["assistant", "user"]
    big = next_turn_messages(msgs, assistant="z" * 20_000, user="u3", budget_chars=10_000)
    assert big is None


def test_a_gap_longer_than_the_stall_limit_is_a_stall_not_a_short_reply():
    from harness.soak import read_stream

    times = iter([0.0, 1.0, 2.0, 200.0])  # chunk arrival times; the 4th arrives 198 s after the 3rd
    chunks = ["a", "b", "c", "d"]
    text, status = read_stream(iter(chunks), stall_seconds=120, clock=lambda: next(times))
    assert status == "stall"
    assert text == "abc"
    times = iter([0.0, 1.0, 2.0, 3.0])
    assert read_stream(iter(chunks), stall_seconds=120, clock=lambda: next(times)) == ("abcd", "ok")


class WordTok:
    """Fake tokenizer: decode(ids) -> 'w<i> w<j> ...'. Enough for run_soak, which only decodes."""

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def fake_completions(reply_text, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        events = [{"choices": [{"delta": {"content": reply_text}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 3}}]
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_run_soak_records_turns_and_stops_at_the_fake_deadline():
    # `now` is a fake clock stepping by 1 on every call; run_soak checks it once for the initial
    # deadline and once per while-condition, so picking hours such that the deadline lands at 3
    # lets exactly one turn run before the loop ends, deterministically and instantly.
    clock = iter(range(0, 1000))
    out = soak.run_soak(fake_completions("looks fine, ship it"), "http://x", "m", WordTok(),
                        [{"id": "c0", "kind": "code", "token_ids": [1, 2, 3]}], ["review this"],
                        hours=3 / 3600, max_context=1000, stall_seconds=120, max_tokens=16,
                        now=lambda: next(clock))
    assert out["sessions"] == 1
    assert len(out["turns"]) == 1
    assert out["turns"][0]["status"] == "ok"
    assert out["cumulative"] == {"stalls": 0, "errors": 0, "empties": 0, "degenerates": 0}


def test_run_soak_counts_a_stall_and_starts_a_new_session():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no data", request=request)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    clock = iter(range(0, 1000))
    out = soak.run_soak(client, "http://x", "m", WordTok(),
                        [{"id": "c0", "kind": "code", "token_ids": [1, 2, 3]}], ["review this"],
                        hours=3 / 3600, max_context=1000, stall_seconds=120, max_tokens=16,
                        now=lambda: next(clock))
    assert out["cumulative"]["stalls"] == 1
    assert out["turns"][0]["status"] == "stall"


def test_run_soak_counts_a_degenerate_reply():
    clock = iter(range(0, 1000))
    out = soak.run_soak(fake_completions("x" * 10 + "/" * 200), "http://x", "m", WordTok(),
                        [{"id": "c0", "kind": "code", "token_ids": [1, 2, 3]}], ["review this"],
                        hours=3 / 3600, max_context=1000, stall_seconds=120, max_tokens=16,
                        now=lambda: next(clock))
    assert out["cumulative"]["degenerates"] == 1
    assert out["turns"][0]["status"] == "degenerate"


def test_on_progress_sees_sessions_and_cumulative_every_turn_not_just_at_the_end():
    # A reader of the output file mid-run (e.g. during a multi-hour soak, or after an external
    # kill) should see the running cumulative counts and session number, not just the raw turns
    # list that it would otherwise have to re-scan to re-derive them.
    clock = iter(range(0, 1000))
    seen_states = []
    soak.run_soak(fake_completions("x" * 10 + "/" * 200), "http://x", "m", WordTok(),
                 [{"id": "c0", "kind": "code", "token_ids": [1, 2, 3]}], ["review this"],
                 hours=3 / 3600, max_context=1000, stall_seconds=120, max_tokens=16,
                 now=lambda: next(clock), on_progress=lambda state: seen_states.append(state))
    assert seen_states  # on_progress was called at least once
    last = seen_states[-1]
    assert last["sessions"] == 1
    assert last["cumulative"] == {"stalls": 0, "errors": 0, "empties": 0, "degenerates": 1}
    assert len(last["turns"]) == 1
