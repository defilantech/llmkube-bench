# Method

## Hardware

| | |
|---|---|
| Node name | `shadowstack` |
| CPU | — (not load-bearing for this bench) |
| GPUs | 2× NVIDIA GeForce RTX 5060 Ti (16 GB GDDR7 each; 32 GB total) |
| Driver | CUDA 13.x via NVIDIA GPU Operator |
| OS | Ubuntu 24.04.3 LTS, kernel 6.17.0-oem |
| Kubernetes | MicroK8s v1.32.13 (single-node) |
| Container runtime | containerd 1.6.36 |

## Cluster components

- LLMKube operator (Helm chart **0.7.0**, controller **0.7.1-dev**)
- NVIDIA GPU Operator + DCGM exporter
- kube-prometheus-stack (for metric scrape + Grafana)
- microk8s-hostpath storage class for the model-cache PVC

## Images

**Tag pinning.** Manifests reference tags today for readability. For publishable
runs, replace the tags with SHA digests captured from the first successful
smoke run and record them here:

| Runtime | Tag | Digest (fill in after smoke) |
|---|---|---|
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda13` | `sha256:…` |
| vLLM | `vllm/vllm-openai:latest` | `sha256:…` |

## Candidate configuration

### llama.cpp

- Model: `unsloth/Qwen3.5-27B-GGUF` → `Qwen3.5-27B-Q4_K_M.gguf` (~17 GB)
- Sharding: `split-mode=layer` across both GPUs (LLMKube `Model.spec.hardware.gpu.sharding.strategy: layer`)
- Full-GPU offload (`gpuLayers: -1`, resolves to `--n-gpu-layers 99`)
- Context: 32 768
- Parallel slots: 16
- Flash attention: on
- Jinja chat templating: on
- Batch / ubatch: 2048 / 512
- KV cache dtype: default (f16)

### vLLM

- Model: `Qwen/Qwen3.5-27B-FP8` (~28 GB, official Qwen FP8 safetensors)
- Tensor parallel: 2
- `maxModelLen`: 32 768
- Attention backend: FLASHINFER
- Prefix caching: **on** (`enablePrefixCaching: true`)
- Chunked prefill: **on** (via `extraArgs: ["--enable-chunked-prefill"]`; typed in newer LLMKube CRDs)
- KV cache dtype: fp8_e4m3 (via `extraArgs: ["--kv-cache-dtype", "fp8_e4m3"]`)
- `maxNumBatchedTokens`: 8192 (via extraArgs)
- Quantization: fp8 (via extraArgs; the v0.7.0 CRD enum predates fp8)

### Why the configs are not identical

Each runtime is set to what a production operator would actually pick for
on-prem inference of a 27B model on 2× 16 GB consumer GPUs. See README.md
and QUALITY-GATE.md for the framing around this choice.

## Workload matrix

4 patterns × 4 concurrency levels × 2 runtimes = **32 cells**.

| Pattern | Avg message chars | Target output tokens | Prompts |
|---|---|---|---|
| chat | ~50 | 256 | 20 |
| coding | ~300 | 1024 | 20 |
| long_context | ~47 000 | 1024 | 10 |
| agentic | ~2 900 (shared system) | 512 | 20, all sharing one prefix |

Concurrency: `1, 4, 16, 64`.
Per cell: 2 min warmup (discarded) + 5 min measurement (recorded).
Wall-time estimate: ~6 hours, mostly unattended.

## Sampling

- Temperature: 0.0, seed: 42 (both runtimes) — keeps quality drift comparable
- Streaming: yes (both runtimes emit OpenAI-compatible SSE)
- `stream_options.include_usage`: true (vLLM returns token counts on the final chunk; llama.cpp ignores)

## Metrics captured

**Client-side (from `harness/run.py`):**
- TTFT (time to first streamed content token) — ms, p50/p95/p99
- Inter-token latency — ms, p50/p95/mean
- Per-request total latency — ms
- Aggregate generation throughput — tokens/sec (over the measurement wall-clock)
- Success rate (useful for detecting silent OOMs at high concurrency)

**Server-side (from `harness/prom_snapshot.py`):**
- `DCGM_FI_DEV_GPU_UTIL` — per-GPU utilization
- `DCGM_FI_DEV_FB_USED` — per-GPU VRAM used (MiB)
- `DCGM_FI_DEV_POWER_USAGE` — total power draw (W)
- vLLM: `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, prefix cache queries/hits
- llama.cpp: `llamacpp:tokens_predicted_total`, `llamacpp:prompt_tokens_total` (rate over 5m)

## vLLM PodMonitor gap (why a local one ships here)

As of LLMKube chart v0.7.0, the bundled `PodMonitor` targets `port: http`
(8080) with the llama.cpp `/metrics` path only. vLLM serves its own
Prometheus endpoint on its configured port (8000 here). Until the chart
grows runtime-aware scraping, this repo ships `manifests/podmonitor-vllm.yaml`
to close the gap inside the `bench` namespace only.

Follow-up issue to land upstream: add `vllm`-runtime PodMonitor to
`charts/llmkube/templates/`.

## Reproducing

```bash
make install                                  # harness Python deps
make smoke RESULTS_DIR=results/smoke/          # 2-3 min; verifies plumbing
make bench RESULTS_DIR=results/$(date +%F)/    # full matrix
```

Outputs:

- `$RESULTS_DIR/raw/<runtime>/<pattern>/c<N>.jsonl` — per-request samples
- `$RESULTS_DIR/raw/<runtime>/<pattern>/c<N>.prom.json` — Prometheus snapshot
- `$RESULTS_DIR/summary.csv` — aggregated table, one row per cell
