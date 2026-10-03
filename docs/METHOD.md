# Method

## Hardware

| | |
|---|---|
| Node name | `shadowstack` (the author's specific MicroK8s host; substitute your own when applying the manifests) |
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

- Model: `unsloth/Qwen3-14B-GGUF` → `Qwen3-14B-Q4_K_M.gguf` (~9 GB)
- Sharding: `split-mode=layer` across both GPUs (LLMKube `Model.spec.hardware.gpu.sharding.strategy: layer`)
- Full-GPU offload (`gpuLayers: -1`, resolves to `--n-gpu-layers 99`)
- Context: 16 384 (matched to vLLM; see notes below)
- Parallel slots: 16
- Flash attention: on
- Jinja chat templating: on
- Batch / ubatch: 2048 / 512
- KV cache dtype: default (f16)

### vLLM

- Model: `Qwen/Qwen3-14B-FP8` (~14 GB, official Qwen text-only FP8 safetensors)
- Tensor parallel: 2
- `maxModelLen`: 16 384 (see *Why 16K* below)
- Attention backend: FLASHINFER
- Prefix caching: **on** (`enablePrefixCaching: true`)
- Chunked prefill: **on** (via `extraArgs: ["--enable-chunked-prefill"]`; typed in newer LLMKube CRDs)
- KV cache dtype: fp8_e4m3 (via `extraArgs: ["--kv-cache-dtype", "fp8_e4m3"]`)
- `maxNumBatchedTokens`: 8192 (via extraArgs)
- Quantization: fp8 (via extraArgs; the v0.7.0 CRD enum predates fp8)

### Why 16K context on both sides

Qwen3-14B has 40 transformer layers and a 5120-dim hidden size. At 32K
tokens × 2 (K,V) × 40 × 5120 × 1 byte (FP8) / 2 (tensor shards) ≈ 6.5 GiB
of KV cache per shard. On a 15.48 GiB card with ~7 GiB of FP8 weights
already resident, vLLM can't also capture CUDA graphs and hold activations
— it OOMs during graph compile. At 16K, the KV budget roughly halves to
~3.25 GiB/shard and vLLM fits with ~1 GiB of headroom. llama.cpp is
matched to 16K so the comparison is apples-to-apples.

### Why the configs are not identical

Each runtime is set to what a production operator would actually pick for
on-prem inference of a 14B model on 2× 16 GB consumer GPUs. See README.md
and QUALITY-GATE.md for the framing around this choice.

### Why 14B and not 27B

The original target was Qwen3.5-27B — motivated by the "qwen 27B on a 3090
is indistinguishable from frontier models" discourse in the community.
That claim is reported on 3090s (24 GB). Our hardware is 2× RTX 5060 Ti
(15.48 GiB usable per card after driver reserve), and Qwen's only
official FP8 release in the 27B class is the VLM (`Qwen/Qwen3.5-27B-FP8`)
— the weights include a vision encoder that stays resident regardless of
whether multimodal requests are ever sent.

See **Appendix A** below for the full chronology of the 27B attempt — it
is itself a publishable data point about hardware sizing for on-prem
inference on consumer silicon.

---

## Appendix A: the Qwen3.5-27B-FP8 attempt

On 2× RTX 5060 Ti (15.48 GiB usable per card, 30.96 GiB aggregate) we
tried to stand up `Qwen/Qwen3.5-27B-FP8` under vLLM with TP=2. Three
successive mitigations, each measured against the crash logs:

1. **Default config.** vLLM OOMs during `profile_run`, allocating vision
   encoder position embeddings:

   > `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate
   > 576.00 MiB. GPU 0 has a total capacity of 15.48 GiB of which 175.19
   > MiB is free. This process has 15.30 GiB memory in use.`

2. **Add `--limit-mm-per-prompt image=0,video=0`, drop `maxModelLen`
   from 32 K to 16 K, drop `max-num-batched-tokens` from 8 192 to
   4 096.** Skips multimodal dummy inputs during profiling; reduces KV
   cache budget. The vision weights stay resident. Progress — OOM now
   at `determine_available_memory`:

   > `Tried to allocate 1.19 GiB. GPU 0 has a total capacity of 15.48
   > GiB of which 1.02 GiB is free. This process has 14.45 GiB memory
   > in use.`

   We've reclaimed ~850 MiB of headroom versus the default, but still
   short.

3. **Add `--gpu-memory-utilization 0.95` and
   `PYTORCH_ALLOC_CONF=expandable_segments:True`.** vLLM defaults to
   `0.9`, leaving ~10% unused; bumping to `0.95` gives vLLM more room.
   `expandable_segments` reduces fragmentation per the crash's own
   recommendation. Pushes hard against the wall:

   > `Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 15.48
   > GiB of which 3.19 MiB is free. This process has 15.47 GiB memory
   > in use.`

   We're using 15.47 / 15.48 GiB — within driver reservation noise.
   `torch._inductor` can't find 32 MiB for a constant allocation. There
   is no knob left.

### Conclusion

**Qwen3.5-27B-FP8 cannot be served via vLLM on 2× RTX 5060 Ti (2×16 GiB
consumer-class) in any configuration we found.** On a 3090 (24 GB) the
budget doubles and the FP8 weights + vision encoder + KV cache comfortably
fit. The "run Qwen 27B locally" discourse has a hardware footnote: not
every pair of 16 GB consumer GPUs is enough for the current-release 27B
FP8 checkpoints, and no "skip the vision head" flag exists in stock vLLM
0.19 that would reclaim those weights from VRAM.

llama.cpp Q4_K_M of the same 27B fits trivially (~17 GB split across both
cards). The runtime bake-off over that config was possible on paper, but
it would have been comparing vLLM failure against llama.cpp success —
not a useful comparison. We pivoted both sides to 14B, where the
comparison is honest: both runtimes have ample headroom, and prefix
caching / FP8 KV / chunked prefill all exercise at scale.

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

## Ladder and quality method

Method notes for `harness.ladder` and the `harness.quality.*` tools (see the
README's tool table for what each one does).

**Cold vs. cached prompts.** Every cold prompt in the ladder starts with a
unique leading nonce so it cannot land in the server's prefix cache. A cached
prompt is sent twice with identical text; only the second, timed request is
recorded, so its prefill measures a cache hit rather than a cold one.

**Rates come from server usage, not client estimates.** Both the prefill
ladder and the decode runs read token counts from the server's `usage` block
(chat templates add tokens, so a client-side count of the input text is never
the true prompt length). A row whose response carried no usage block, or a
decode stream in which no token was actually seen, is marked invalid with a
`None` rate rather than estimated from the requested size.

**What the top-k KL approximates.** `harness.quality.logprobs` computes KL
divergence over the reference model's top-k id set at each position, not the
full vocabulary: `p` is the reference's probabilities renormalized over that
set, and `q` is the candidate's probability for the same ids, with any id
missing from the candidate's own top-k given the candidate's smallest
returned probability as a floor (an upper bound on how much mass it could
hold). This keeps the metric well-defined and comparable across candidates
without requiring either side to return a full-vocabulary distribution; it is
not the true full-vocabulary KL.

**PPL and KL are teacher-forced on token ids.** Both scores come from posting
the corpus's raw token ids (not re-tokenized text) to vLLM's completions
endpoint with `prompt_logprobs`, so the reference and every candidate score
the exact same token sequence with no tokenizer or chat-template drift
between them.

**LiveCodeBench here is not the LCB leaderboard.** `harness.quality.lcb` scores
generated solutions against public test cases only, executed in a no-network
sandbox pod (see `harness/quality/lcb_exec.py`). The official LiveCodeBench
leaderboard also runs private test cases and different tooling entirely, so
pass rates from this harness are not comparable to leaderboard numbers, only
to other runs made with this same harness.
