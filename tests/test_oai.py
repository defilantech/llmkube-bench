import json

import httpx

from harness import oai


def sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def test_chat_captures_finish_reason_from_the_terminal_chunk():
    events = [{"choices": [{"delta": {"content": "hi"}}]},
              {"choices": [{"delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 1}}]

    def handler(request):
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    res = oai.chat(client, "http://x", "m", [], max_tokens=1)
    assert res.finish_reason == "stop"
    assert res.stalled is False


def test_chat_defaults_stalled_false_and_finish_reason_none_without_stall_seconds():
    def handler(request):
        return httpx.Response(200, text=sse([{"choices": [{"delta": {"content": "x"}}]}]),
                              headers={"content-type": "text/event-stream"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    res = oai.chat(client, "http://x", "m", [], max_tokens=1)
    assert res.stalled is False
    assert res.finish_reason is None


def test_chat_treats_a_socket_read_timeout_as_stalled():
    def handler(request):
        raise httpx.ReadTimeout("no data", request=request)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    res = oai.chat(client, "http://x", "m", [], max_tokens=1, stall_seconds=5)
    assert res.stalled is True
    assert res.text == ""


def test_chat_detects_a_mid_stream_gap_via_the_injected_clock():
    """Three content chunks; the fake clock reports a 199s gap before the 3rd arrives, which
    exceeds stall_seconds=120. The late chunk ("c") must not be included, matching
    harness.soak.read_stream's "the chunk that arrived late is not included" semantics."""
    events = [{"choices": [{"delta": {"content": "a"}}]},
              {"choices": [{"delta": {"content": "b"}}]},
              {"choices": [{"delta": {"content": "c"}}]}]

    def handler(request):
        return httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    times = iter([0.0, 1.0, 200.0])
    res = oai.chat(client, "http://x", "m", [], max_tokens=3, stall_seconds=120,
                   clock=lambda: next(times))
    assert res.stalled is True
    assert res.text == "ab"
