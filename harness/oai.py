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


def chat(client: httpx.Client, endpoint: str, model: str, messages: list[dict], max_tokens: int,
         temperature: float = 0.0, extra: dict | None = None) -> ChatResult:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra or {})
    start = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage: dict = {}
    with client.stream("POST", endpoint.rstrip("/") + "/v1/chat/completions", json=body, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
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
    total = time.perf_counter() - start
    return ChatResult("".join(parts), usage.get("prompt_tokens"), usage.get("completion_tokens"),
                      ttft if ttft is not None else total, total)
