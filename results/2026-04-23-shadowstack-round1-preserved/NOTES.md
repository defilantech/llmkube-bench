# Round 1 — preserved

This directory is a snapshot of the first full 36-cell run on 2026-04-23.

## What's here

- `summary.csv` — aggregated per-cell metrics across both runtimes
- `raw/` — per-request JSONL and per-cell Prometheus snapshots (gitignored at the repo level)

## What's valid

- **28 of 36 cells succeeded.**
- `llamacpp` × `{chat, coding, agentic}` × `{1, 4, 16, 64}` — 12 cells.
- `vllm` × `{chat, coding, long_context, agentic}` × `{1, 4, 16, 64}` — 16 cells.

## What failed

All **8 llama.cpp `long_context` and `long_context_extreme` cells returned
100% HTTP 400 Bad Request** from llama.cpp's OpenAI endpoint. Short
patterns on the same runtime were clean (100% success), so the failure
localizes to something about the long-context prompt shape interacting
with the Qwen3.6 chat template under `--jinja` + `--override-kv
tokenizer.chat_template.thinking=bool:false`.

The `long_context_extreme` cell was meant to be the headline "only
llama.cpp + TurboQuant can serve this" data point. Kept for reference,
not used in the published post.

## Why preserved

Round 2 re-runs the llama.cpp long-context cells with a fixed config
and writes into `../2026-04-23-shadowstack/` (same path, merged with
the successful round-1 cells). This copy is frozen so we can compare
before/after.
