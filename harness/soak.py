"""Long single-session soak: a many-turn coding conversation that grows toward --max-context,
watching for the two failure modes a short bench never sees: a stream that stalls mid-reply, and a
reply that degrades into a repetitive, non-answer as context grows.

The loop: open a session with a system prompt plus one corpus code item, ask a coding question
about it, stream the reply with a per-chunk stall timeout, append the reply and the next code item
plus question, repeat. Once the running conversation would exceed --max-context (estimated at
~3.5 characters per token, since we do not want to pay for a server round trip just to count), the
session ends and a fresh one starts. This repeats for --hours wall-clock time.

A turn is one of: "ok", "empty" (blank reply), "degenerate" (see classify_reply), "stall" (no
chunk within --stall-seconds), or "error" (the request itself failed, e.g. connection reset). Any
of the last four across the whole run is a failing soak: exit 2.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from harness import oai

CODE_QUESTIONS = [
    "Explain what this code does and point out one thing you'd improve.",
    "Write a unit test for the function above.",
    "Refactor the function above for readability. Keep behavior identical.",
    "Find a potential bug or edge case the code above doesn't handle.",
    "Add type hints and a short docstring to the function above.",
]

_DEGENERATE_CHAR_TAIL = 200
_DEGENERATE_CHUNK_TAIL = 400
_DEGENERATE_CHUNK_SIZE = 4
_DEGENERATE_CHAR_RATIO = 0.5
_DEGENERATE_CHUNK_RATIO = 0.8


def _max_chunk_repeat_coverage(text: str, chunk_size: int = _DEGENERATE_CHUNK_SIZE) -> float:
    """Best-case fraction of text covered by one chunk value repeating, over all 4 alignments.

    Non-overlapping chunks are counted per alignment (offset 0..chunk_size-1) so a periodic
    pattern is found regardless of where in `text` its period happens to start.
    """
    if len(text) < chunk_size:
        return 0.0
    best = 0
    for offset in range(chunk_size):
        counts: Counter[str] = Counter()
        i = offset
        while i + chunk_size <= len(text):
            counts[text[i:i + chunk_size]] += 1
            i += chunk_size
        if counts:
            best = max(best, counts.most_common(1)[0][1])
    return (best * chunk_size) / len(text)


def classify_reply(text: str) -> str:
    """Returns "empty" for blank text, "degenerate" for a reply dominated by one character or
    one repeating 4-character chunk near its end, else "ok"."""
    if not text.strip():
        return "empty"
    tail_chars = text[-_DEGENERATE_CHAR_TAIL:]
    most_common_count = Counter(tail_chars).most_common(1)[0][1]
    if most_common_count / len(tail_chars) > _DEGENERATE_CHAR_RATIO:
        return "degenerate"
    tail_chunks = text[-_DEGENERATE_CHUNK_TAIL:]
    if _max_chunk_repeat_coverage(tail_chunks) > _DEGENERATE_CHUNK_RATIO:
        return "degenerate"
    return "ok"


def next_turn_messages(msgs: list[dict], assistant: str, user: str, budget_chars: int) -> list[dict] | None:
    """Appends the assistant reply and the next user turn; None if that would exceed budget_chars."""
    grown = msgs + [{"role": "assistant", "content": assistant}, {"role": "user", "content": user}]
    if sum(len(m["content"]) for m in grown) > budget_chars:
        return None
    return grown


def read_stream(chunks, stall_seconds: float, clock=time.monotonic) -> tuple[str, str]:
    """Consumes an iterator of text chunks, timestamping each with clock().

    The first chunk is never held to a gap check (there is no previous chunk to measure from).
    Every later chunk is compared to the one before it; a gap over stall_seconds ends the stream
    immediately, "stall", with the late chunk excluded from the returned text. This only catches
    gaps between chunks that do arrive: a chunk iterator that blocks forever without ever
    returning is a separate case, handled in the live loop by a socket read timeout instead (see
    harness.oai.chat's stall_seconds parameter for the same rule applied to a live SSE stream).
    """
    parts: list[str] = []
    last = None
    for chunk in chunks:
        now = clock()
        if last is not None and now - last > stall_seconds:
            return "".join(parts), "stall"
        parts.append(chunk)
        last = now
    return "".join(parts), "ok"


def run_soak(client, endpoint, model, tokenizer, code_items: list[dict], questions: list[str],
            hours: float, max_context: int, stall_seconds: float, max_tokens: int = 512,
            on_progress=None, now=time.monotonic) -> dict:
    """The turn loop described in the module docstring. now() is injectable for tests."""
    if not code_items:
        raise ValueError("code_items is empty; the soak needs at least one corpus code item")
    if not questions:
        raise ValueError("questions is empty; the soak needs at least one question to ask")
    budget_chars = int(max_context * 3.5)
    deadline = now() + hours * 3600
    item_iter = itertools.cycle(code_items)
    question_iter = itertools.cycle(questions)
    turns: list[dict] = []
    stalls = errors = empties = degenerates = 0
    session = 0

    def next_user_text() -> str:
        item = next(item_iter)
        return f"{tokenizer.decode(item['token_ids'])}\n\n{next(question_iter)}"

    def state() -> dict:
        # Reported after every turn, not just at the end, so a reader of the output file mid-run
        # (e.g. checking in on a multi-hour soak, or after it was killed) sees the running
        # cumulative counts and session number rather than having to re-derive them by scanning
        # the raw turns list itself.
        return {"sessions": session, "turns": turns,
               "cumulative": {"stalls": stalls, "errors": errors, "empties": empties,
                              "degenerates": degenerates}}

    while now() < deadline:
        session += 1
        messages = [{"role": "system", "content": "You are pair-programming with a colleague on "
                                                   "the code they paste. Answer their question about it."},
                   {"role": "user", "content": next_user_text()}]
        while now() < deadline:
            try:
                r = oai.chat(client, endpoint, model, messages, max_tokens=max_tokens, temperature=0.0,
                            stall_seconds=stall_seconds)
            except httpx.HTTPError as e:
                errors += 1
                turns.append({"session": session, "status": "error", "error": str(e)})
                if on_progress is not None:
                    on_progress(state())
                break
            if r.stalled:
                stalls += 1
                turns.append({"session": session, "status": "stall", "prompt_tokens": r.prompt_tokens,
                             "ttft_s": r.ttft_s, "finish_reason": r.finish_reason})
                if on_progress is not None:
                    on_progress(state())
                break
            status = classify_reply(r.text)
            if status == "empty":
                empties += 1
            elif status == "degenerate":
                degenerates += 1
            decode_tok_s = None
            if r.completion_tokens is not None and r.completion_tokens >= 2 and r.ttft_s < r.total_s:
                decode_tok_s = (r.completion_tokens - 1) / (r.total_s - r.ttft_s)
            turns.append({"session": session, "status": status, "prompt_tokens": r.prompt_tokens,
                         "ttft_s": r.ttft_s, "decode_tok_s": decode_tok_s,
                         "finish_reason": r.finish_reason})
            if on_progress is not None:
                on_progress(state())
            grown = next_turn_messages(messages, assistant=r.text, user=next_user_text(),
                                       budget_chars=budget_chars)
            if grown is None:
                break  # budget reached; start a new session
            messages = grown

    return state()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.soak")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--stall-seconds", type=float, default=120)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--decode-instruction", default=None,
                    help="single question asked every turn; omit to rotate a small built-in set")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    items = [json.loads(line) for line in a.corpus.read_text().splitlines() if line.strip()]
    code_items = [it for it in items if it.get("kind") == "code"] or items
    questions = [a.decode_instruction] if a.decode_instruction else list(CODE_QUESTIONS)

    meta = {"kind": "soak", "endpoint": a.endpoint, "model": a.model, "hours": a.hours,
           "max_context": a.max_context, "stall_seconds": a.stall_seconds}

    def write(state: dict, complete: bool) -> None:
        payload = dict(meta)
        payload.update(state)
        payload["complete"] = complete
        payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        a.output.write_text(json.dumps(payload, indent=1))

    with httpx.Client() as client:
        out = run_soak(client, a.endpoint, a.model, tok, code_items, questions, a.hours, a.max_context,
                       a.stall_seconds, a.max_tokens,
                       on_progress=lambda state: write(state, complete=False))

    write(out, complete=True)
    c = out["cumulative"]
    if c["stalls"] or c["errors"] or c["empties"] or c["degenerates"]:
        print(f"soak: {c['stalls']} stall(s), {c['errors']} error(s), {c['empties']} empty and "
             f"{c['degenerates']} degenerate reply/replies (see 'cumulative' in the output file)",
             file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
