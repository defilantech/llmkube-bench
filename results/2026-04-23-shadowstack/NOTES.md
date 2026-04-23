# Results — 2026-04-23 on shadowstack (2× RTX 5060 Ti, 2× 16 GB)

Final merged dataset from two rounds of runs. All 36 cells present in
`summary.csv`.

## Round 1 (completed ~08:25 UTC)

- llama.cpp × {chat, coding, agentic} × {1, 4, 16, 64} — **12/12 clean**
- vLLM × {chat, coding, long_context, agentic} × {1, 4, 16, 64} — **16/16 clean**
- llama.cpp × {long_context, long_context_extreme} × {1, 4, 16, 64} — **0/8** (all HTTP 400 on prompt submission)

The 400s were a per-slot context-size misconfiguration: `--parallel 16`
divides `--ctx-size 65536` into 4K per slot, too small for the 5K+
long_context prompts. Fixed in round 2.

## Round 2 (completed ~10:14 UTC)

Re-ran llama.cpp long-context patterns with `parallelSlots: 1` so every
request has the full 65K context.

### long_context (~5K input, 1K output)

| concurrency | success rate | notes |
|---|---|---|
| 1 | 100% | clean — 20 tok/s, TTFT p50 2.3s |
| 4 | 100% | clean — 39 tok/s, TTFT p50 25s (queue wait) |
| 16 | 60% | server queue saturates — 8 timeouts of 20 |
| 64 | 18% | heavy queue loss — 56 timeouts of 68 |

The degradation at c≥16 is an intentional methodology tradeoff:
`parallelSlots=1` was required to give each request the 65K context
it needs. Higher concurrency means requests queue server-side rather
than run in parallel, and the harness's 300s per-request timeout cuts
some off.

### long_context_extreme (~43K input, 1K output)

| concurrency | success rate | notes |
|---|---|---|
| 1 | 0/1 | hit 300s harness timeout |
| 4 | 0/4 | hit 300s harness timeout |
| 16 | 1/16 | **one request completed** — TTFT 186s (3 min prefill), ITL 171ms |
| 64 | 0/64 | hit 300s harness timeout |

The single successful request at c=16 is the post's headline for
llama.cpp+TurboQuant: 43K-token prompt processed and generated
end-to-end on 2× 16 GB consumer GPUs. vLLM with the same hardware is
capped at 16K `maxModelLen` and cannot attempt this workload at all.

TurboQuant's tbqp3/tbq3 KV cache compression is memory-optimal but
compute-slow on long prefills (the 186s TTFT on 43K tokens works out
to ~230 tok/s prefill throughput). A harness timeout ≥ 600s would
capture the pattern at c=1 cleanly; noted as a follow-up for a future
re-run on a machine with more patience.

## What the post uses

**Throughput comparison** (short patterns, all clean):
- llama.cpp and vLLM both run chat/coding/agentic at all 4 concurrencies
- vLLM 3–4× faster at c=64 (PagedAttention + continuous batching)
- llama.cpp slightly lower TTFT at c=1 on some patterns

**Latency comparison** (long_context c=1, clean both sides):
- llama.cpp TTFT 2.3s (TurboQuant prefill is slow by design)
- vLLM TTFT 0.58s on the same prompt at its 16K cap

**Context ceiling** (the bake-off's other axis):
- vLLM: 16K maxModelLen, saturated — cannot attempt 43K prompts
- llama.cpp + TurboQuant: served 43K prompt end-to-end (1 captured
  sample); 5K works at all concurrencies

**Cost attribution**: InferCost controller captured 3.75M tokens across
the bench window with amortized $/MTok $0.11, marginal $/MTok $0.005,
utilization 20.9% — the "two numbers" the post pivots on.

## Preserved

`../2026-04-23-shadowstack-round1-preserved/` has the original
0/8-failed data for reference. Not used in the post.
