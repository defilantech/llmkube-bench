# llmkube-bench

Reproducible head-to-head of **llama.cpp** vs **vLLM** as inference runtimes on the same Kubernetes cluster, same model family, same hardware. Deploys both via the [LLMKube](https://github.com/defilantech/llmkube) operator.

**Why this repo exists.** Operators picking a local inference stack have to choose between llama.cpp (ubiquitous, GGUF, broad quant support) and vLLM (throughput-focused, PagedAttention, FP8 on recent GPUs). The ecosystem answers that question with vibes and forum posts. This repo answers it with numbers — the same numbers, from the same hardware, re-runnable by anyone with a kubeconfig and two GPUs.

## What we measure

**Qwen3.6-27B** — Tongyi Lab's 2026-04-21 flagship-agentic-coding release — served on both runtimes via LLMKube, on 2× RTX 5060 Ti (Blackwell GB206, 2×16 GB consumer cards). Published same-day as the model itself.

| | llama.cpp | vLLM |
|---|---|---|
| Source | `unsloth/Qwen3.6-27B-GGUF` **Q4_K_M** (~17 GB) | `sakamakismile/Qwen3.6-27B-NVFP4` (~14 GB) |
| Weight bits | ~4.5 (Q4_K_M) | 4 (Blackwell-native NVFP4) |
| Parallelism | layer-split across 2 GPUs | tensor-parallel (TP=2) |
| KV cache | **TurboQuant tbqp3-K / tbq3-V** (~3 bits) | FP8 E4M3 (8 bits) |
| Max context | **65 536** | 32 768 |
| Image | `registry.defilan.net/llmkube-turboquant:amx-v1.5.2` | `vllm/vllm-openai:latest` |

Five workload patterns × four concurrency levels × two runtimes (with the `long_context_extreme` cell exclusive to llama.cpp since it exceeds vLLM's 32K cap) — **36 measured cells**. Per cell we capture TTFT p50/p95/p99, inter-token latency, aggregate tokens/sec, GPU utilization, VRAM used, and power draw.

## The real story

**This isn't just a runtime bake-off — it's a capability tradeoff.**

- **vLLM** wins on throughput and latency under concurrent load, thanks to PagedAttention + CUDA graphs + prefix caching + chunked prefill + FLASHINFER kernels on Blackwell.
- **llama.cpp with TurboQuant KV** wins on context length — at the same VRAM budget it serves **2× the context window** that vLLM can. On workloads that need 48K+ tokens of context (long-file code review, RAG-heavy agent turns, overnight refactors), llama.cpp is the only option on this hardware.

Both are legitimate production choices for different workloads. See [`docs/METHOD.md`](docs/METHOD.md) for the full rationale and the earlier Qwen3.5-27B-FP8 attempt that motivated the NVFP4 pivot (METHOD Appendix A — a publishable-in-itself data point about VLM overhead on consumer GPUs).

## Not apples-to-apples — and that's the point

These are not identical quants. They are the choices an operator actually makes per runtime. Quality is addressed separately in [docs/QUALITY-GATE.md](docs/QUALITY-GATE.md): the same five prompts run through both, outputs pasted side-by-side, so readers judge the drift themselves.

## Reproduce

Requirements:

- Kubernetes cluster with LLMKube v0.4+ installed
- 2× CUDA GPUs with ≥16 GB each (we run on 2× RTX 5060 Ti)
- `kubectl` context set to the target cluster
- Python 3.11+ and `uv` or `pip`
- A HuggingFace token Secret named `hf-token` in the `bench` namespace (key: `HF_TOKEN`). vLLM pulls FP8 safetensors directly from HuggingFace.

```bash
# One-time: clone + install harness deps
git clone https://github.com/defilantech/llmkube-bench.git
cd llmkube-bench
make install

# Smoke check: deploy each runtime in turn and run one request
make smoke

# Full matrix (~6 hours, largely unattended)
make bench RESULTS_DIR=results/$(date +%Y-%m-%d)-myhardware

# Aggregate + summarize
make analyze RESULTS_DIR=results/...
```

Every number in our published write-ups comes from running `make bench` on the hardware described in [docs/METHOD.md](docs/METHOD.md). Results from our runs live under `results/`.

## Repo layout

```
manifests/            Model + InferenceService CRs (llamacpp/, vllm/), namespace, vLLM PodMonitor
harness/              Python asyncio load generator + Prometheus snapshotter
harness/patterns/     Workload JSONL (chat, coding, long_context, agentic)
bench.sh              Orchestrator: deploys each runtime, runs matrix, scales down
results/              Captured runs (raw/ gitignored by default)
docs/METHOD.md        Hardware, image pinning, all flags
docs/QUALITY-GATE.md  Side-by-side output samples
grafana/              Benchmark dashboard JSON
```

## License

Apache 2.0 — same as LLMKube itself.
