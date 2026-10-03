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
| Image | AmesianX's TurboQuant fork v1.5.2 (build locally, see below) | `vllm/vllm-openai:latest` |

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
- A container registry your cluster can pull from (needed for the TurboQuant image; see "Build the TurboQuant image" below)
- A HuggingFace token Secret in the `bench` namespace. Create it with:
  ```bash
  kubectl -n bench create secret generic hf-token \
    --from-literal=HF_TOKEN=hf_your_actual_token_here
  ```
  The manifests reference this Secret by name; the token value stays in your cluster and is never committed to git.

### Build the TurboQuant image

llama.cpp runs on AmesianX's TurboQuant fork (v1.5.2), which isn't published on Docker Hub. Build it yourself:

1. Clone and build [AmesianX/llama.cpp](https://github.com/AmesianX/llama.cpp) at tag `v1.5.2` with CUDA support (a `Dockerfile` in that repo handles this).
2. Push to your container registry.
3. In `manifests/llamacpp/isvc.yaml`, replace `<your-registry>/llmkube-turboquant:amx-v1.5.2` with your image reference.
4. If your registry needs authentication, create a dockerconfigjson Secret named `turboquant-registry-cred` in the `bench` namespace.

The provided `manifests/bench-runner/kaniko-build.yaml` also demonstrates how to build the bench harness image in-cluster via Kaniko.

### Run the bench

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

## Quality and ladder tools

A single-stream ladder harness and a set of quality tools live alongside the
throughput bench. Each is runnable on its own; see [docs/METHOD.md](docs/METHOD.md)
for the full method behind them.

| Tool | Purpose | Minimal example |
|---|---|---|
| `harness.ladder` | Prefill-size ladder plus low-concurrency decode timing against one OpenAI-compatible endpoint. | `python -m harness.ladder --endpoint http://host:8000 --model m --label run1 --tokenizer tokenizer.json --corpus corpus.txt --output ladder.json` |
| `harness.quality.build_corpus` | Builds the fixed PPL/KL corpus deterministically from an Apache-2.0 LLMKube checkout at a pinned commit. | `python -m harness.quality.build_corpus --repo ../llmkube --commit <sha> --tokenizer tokenizer.json --output corpus.jsonl` |
| `harness.quality.logprobs` | Captures and compares perplexity and top-k KL divergence by teacher forcing through vLLM's `prompt_logprobs`. | `python -m harness.quality.logprobs capture --endpoint http://host:8000 --model m --corpus corpus.jsonl --output ref.jsonl` |
| `harness.quality.lcb` | Selects a deterministic LiveCodeBench subset, generates solutions, bundles them for execution, and scores results. | `python -m harness.quality.lcb generate --endpoint http://host:8000 --model m --problems lcb-release_v6.jsonl --output solutions.jsonl` |
| `harness.quality.lcb_exec` | Stdlib-only sandboxed executor for LCB stdin/stdout solutions; runs standalone or as the program `lcb.py bundle` produces. | `python harness/quality/lcb_exec.py < solutions.jsonl > results.jsonl` |
| vendored battery (`harness/third_party/tonyd2wild_quality_battery.py`) | 12-item auto-graded math/logic/code/format quality battery, identical prompts across lanes. See [harness/third_party/README.md](harness/third_party/README.md) for provenance and the thinking-switch caveat. | `python harness/third_party/tonyd2wild_quality_battery.py http://host:8000 served-model lane-name --thinking off` |

## Repo layout

```
manifests/              Model + InferenceService CRs (llamacpp/, vllm/), namespace, vLLM PodMonitor
harness/                Python asyncio load generator + Prometheus snapshotter
harness/patterns/       Workload JSONL (chat, coding, long_context, agentic)
harness/quality/        PPL/KL, LiveCodeBench, and corpus-building tools (see table above)
harness/third_party/    Vendored quality battery + provenance notes
bench.sh                Orchestrator: deploys each runtime, runs matrix, scales down
tests/                  pytest suite for the harness and quality tools
results/                Captured runs (raw/ gitignored by default)
docs/METHOD.md          Hardware, image pinning, all flags
docs/QUALITY-GATE.md    Side-by-side output samples
```

## License

Apache 2.0 — same as LLMKube itself.
