# llmkube-bench

Reproducible head-to-head of **llama.cpp** vs **vLLM** as inference runtimes on the same Kubernetes cluster, same model family, same hardware. Deploys both via the [LLMKube](https://github.com/defilantech/llmkube) operator.

**Why this repo exists.** Operators picking a local inference stack have to choose between llama.cpp (ubiquitous, GGUF, broad quant support) and vLLM (throughput-focused, PagedAttention, FP8 on recent GPUs). The ecosystem answers that question with vibes and forum posts. This repo answers it with numbers — the same numbers, from the same hardware, re-runnable by anyone with a kubeconfig and two GPUs.

## What we measure

Qwen3-14B at each runtime's production-typical quant:

| | llama.cpp | vLLM |
|---|---|---|
| Source | `unsloth/Qwen3-14B-GGUF` **Q4_K_M** | `Qwen/Qwen3-14B-FP8` |
| Parallelism | layer-split across 2 GPUs | tensor-parallel (TP=2) |
| KV cache | f16 (default) | FP8 E4M3 |

Four workload patterns × four concurrency levels × two runtimes = 32 measured cells. Per cell we capture TTFT p50/p95/p99, inter-token latency, aggregate tokens/sec, GPU utilization, VRAM used, and power draw.

**Why 14B, not 27B.** We started at Qwen3.5-27B-FP8 because of the
"qwen 27B on a 3090" discourse. Qwen's only official 27B-class FP8
release is the VLM, whose resident vision encoder pushes the model past
the 30.96 GiB usable budget on 2× RTX 5060 Ti. See
[docs/METHOD.md Appendix A](docs/METHOD.md) for the chronology and the
exact OOM numbers at each mitigation step — that's a publishable data
point in its own right about hardware sizing on consumer silicon.

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
