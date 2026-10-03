"""Minimal OpenAI-compatible streaming chat client for benchmarks.

The server's usage block is the truth for token counts: chat templates add
tokens, so a client-side count of the user text is never the prompt length.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_s: float
    total_s: float
    finish_reason: str | None = None
    stalled: bool = False


def chat(client: httpx.Client, endpoint: str, model: str, messages: list[dict], max_tokens: int,
         temperature: float = 0.0, extra: dict | None = None, stall_seconds: float | None = None,
         clock=time.monotonic) -> ChatResult:
    """stall_seconds, when set, guards against a stream that goes quiet mid-response.

    Two layers, both needed: an idle socket that never sends another byte only raises if the
    transport itself has a read timeout, so the request is sent with `timeout=stall_seconds`
    (httpx.ReadTimeout is caught and reported as ChatResult.stalled=True). But a socket that keeps
    sending *something* just slowly resets that transport-level timer on every byte, so each SSE
    data line is also timestamped with `clock()` and compared to the previous one; a gap over
    stall_seconds ends the stream there, with the late line excluded (mirrors
    harness.soak.read_stream, which applies the same rule to a plain iterator of chunks).
    """
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra or {})
    start = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage: dict = {}
    finish_reason = None
    stalled = False
    timeout = httpx.Timeout(stall_seconds) if stall_seconds is not None else None
    last_line_at = None
    try:
        with client.stream("POST", endpoint.rstrip("/") + "/v1/chat/completions", json=body,
                           timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                if stall_seconds is not None:
                    now = clock()
                    if last_line_at is not None and now - last_line_at > stall_seconds:
                        stalled = True
                        break
                    last_line_at = now
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except httpx.TimeoutException:
        stalled = True
    total = time.perf_counter() - start
    return ChatResult("".join(parts), usage.get("prompt_tokens"), usage.get("completion_tokens"),
                      ttft if ttft is not None else total, total, finish_reason=finish_reason,
                      stalled=stalled)
