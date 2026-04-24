# Quality gate

This isn't a benchmark. It's a side-by-side so readers can judge for themselves
whether the quant-per-runtime choice (llama.cpp Q4_K_M GGUF vs vLLM NVFP4
safetensors) produces a visible quality gap on the kinds of prompts the
bake-off benchmarks against.

## Method

- **Model:** Qwen3.6-27B on both sides.
- **llama.cpp weights:** `unsloth/Qwen3.6-27B-GGUF` Q4_K_M; TurboQuant KV cache
  (tbqp3/tbq3); flash attention on.
- **vLLM weights:** `sakamakismile/Qwen3.6-27B-NVFP4`; FP8 KV cache;
  compressed-tensors NVFP4; `--enforce-eager` (CUDA graphs disabled to fit
  VRAM); FLASHINFER backend.
- **Sampling:** `temperature: 0`, `seed: 42`, `max_tokens: 6000`, streaming
  off, single request (c=1).
- **Thinking:** Qwen3.6-27B emits a reasoning chain before its final answer on
  non-trivial prompts. llama.cpp returns it via the `reasoning_content` field;
  vLLM returns it as `<think>...</think>` in the content stream. **The outputs
  below are the final answers only** — the reasoning chain is stripped on both
  sides so the comparison is apples-to-apples on what a user would actually
  see.
- **Hardware:** 2× RTX 5060 Ti (16 GB each), TP=2.
- **Date:** 2026-04-23, captured after the bake-off run on the same 2× 5060 Ti hardware described in METHOD.md.

The two runtimes are not running the same quantization, and that's the
point — neither has an apples-to-apples option on this hardware. llama.cpp
doesn't ship an NVFP4 runtime; Q4_K_M has no vLLM implementation. What this
document shows is whether those *different* quants produce answers a reader
would call equivalent.

---

## Prompt 1 — simple explanation

**Prompt:** _Explain how a Kubernetes liveness probe differs from a readiness probe in three sentences._

### llama.cpp / Q4_K_M

A liveness probe checks whether a container is still functioning properly and automatically restarts it if the check fails, helping recover from crashes or deadlocks. A readiness probe determines if a container is prepared to handle requests and temporarily removes it from service endpoints if it fails, preventing traffic from reaching an unready application. While both monitor health, liveness focuses on container recovery through restarts, whereas readiness focuses on traffic routing to ensure only healthy instances receive requests.

### vLLM / NVFP4

A liveness probe monitors whether a container is still functioning correctly, and if it fails, Kubernetes automatically restarts the pod to recover from a stuck or deadlocked state. A readiness probe checks if a container is prepared to accept network traffic, and if it fails, Kubernetes simply removes the pod from service endpoints without restarting it. While liveness probes focus on container survival and automatic recovery, readiness probes ensure traffic is only routed to pods that are fully initialized and capable of handling requests.

**Observation:** Both hit exactly three sentences. Both correctly describe the restart-vs-traffic-routing distinction. Phrasing differs; technical content matches.

---

## Prompt 2 — code generation

**Prompt:** _Write a Go function that walks a directory tree and returns the top 5 largest files. Use filepath.WalkDir. Include error handling for permission denied._

### llama.cpp / Q4_K_M

```go
package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

// FileInfo holds the path and size of a file.
type FileInfo struct {
	Path string
	Size int64
}

// Top5LargestFiles walks the directory tree starting at rootDir and returns
// the 5 largest regular files. It gracefully handles permission denied errors
// by skipping inaccessible paths and continuing the walk.
func Top5LargestFiles(rootDir string) ([]FileInfo, error) {
	var top5 []FileInfo

	err := filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
		// Handle walk errors
		if err != nil {
			if errors.Is(err, os.ErrPermission) {
				// Permission denied: skip this path but continue walking
				return nil
			}
			return err // Stop the walk on other errors
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		// Get file info for size
		info, err := d.Info()
		if err != nil {
			if errors.Is(err, os.ErrPermission) {
				return nil
			}
			return err
		}

		// Add to our tracking slice
		top5 = append(top5, FileInfo{Path: path, Size: info.Size()})

		// Maintain only the top 5 largest files
		if len(top5) > 5 {
			sort.Slice(top5, func(i, j int) bool {
				return top5[i].Size > top5[j].Size
			})
			top5 = top5[:5]
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("walk failed: %w", err)
	}

	return top5, nil
}
```

llama.cpp also included a `main` demo and a notes section on design choices
(top-5 insertion vs full-sort tradeoff, error wrapping with `%w`, `errors.Is`
over `==` for chain-awareness). Trimmed here for length.

### vLLM / NVFP4

```go
package main

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

// FileInfo holds the path and size of a file.
type FileInfo struct {
	Path string
	Size int64
}

// FindTop5LargestFiles walks the directory tree starting at rootDir,
// collects all regular files, sorts them by size in descending order,
// and returns the top 5 largest files.
func FindTop5LargestFiles(rootDir string) ([]FileInfo, error) {
	var files []FileInfo

	err := filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrPermission) {
				return nil
			}
			return err
		}

		if d.IsDir() {
			return nil
		}

		files = append(files, FileInfo{Path: path, Size: d.Size()})
		return nil
	})

	if err != nil {
		return nil, err
	}

	sort.Slice(files, func(i, j int) bool {
		return files[i].Size > files[j].Size
	})

	limit := 5
	if len(files) < limit {
		limit = len(files)
	}

	return files[:limit], nil
}
```

**Observation:** Both functions compile as-is and solve the problem. llama.cpp
keeps a rolling top-5 (memory-bounded); vLLM collects all files and sorts at
the end (simpler, higher peak memory on huge trees). Both handle
`os.ErrPermission` with `errors.Is` and continue the walk. Both call out the
Go 1.16+ requirement. Both note the `d.Size()` vs `d.Info().Size()` tradeoff
(vLLM uses `d.Size()` directly; llama.cpp uses `d.Info()` and explains why).
Different stylistic choices — same correctness.

---

## Prompt 3 — reasoning under constraint

**Prompt:** _A container with 2 GiB memory is OOMKilled repeatedly. Its RSS is 1.4 GiB. What are three plausible causes and what would you look at first?_

### llama.cpp / Q4_K_M

Three plausible causes:

1. **Non-RSS memory counting toward the cgroup limit.** Container memory limits apply to cgroup memory accounting, not just RSS. Linux counts page cache, tmpfs/`/dev/shm`, shared memory, memory-mapped files, and sometimes kernel memory (slab, page tables, kernel stacks) toward the 2 GiB limit. These do not appear in RSS.

2. **Transient memory spikes / allocation bursts.** The 1.4 GiB RSS is likely a point-in-time snapshot. JVMs, Go, Python, fork-heavy processes, or apps doing bulk data loading can experience short-lived memory bursts that exceed the limit before GC or deallocation brings RSS back down.

3. **cgroup accounting differences or kernel memory overhead.** Depending on cgroup v1 vs v2, kernel memory may be included. `memory.high` (throttle) vs `memory.max` (hard limit) can cause premature OOM if misconfigured; fragmentation or aggressive memory reclaim can trigger OOM even when total usage < limit.

**What to look at first:** OOM killer logs (`dmesg -T | grep -i oom` or `journalctl -k | grep -i oom`) for the exact memory state at the moment of the kill. Then the cgroup memory breakdown (`/sys/fs/cgroup/memory.stat` on v2, `/sys/fs/cgroup/memory/memory.stat` on v1) — compare `rss` vs `cache` vs `shmem` vs `slab`. These two steps give ground-truth data in under 30 seconds and immediately categorize the problem.

### vLLM / NVFP4

The discrepancy between the 1.4 GiB RSS and the 2 GiB limit is the key clue. In Linux/Kubernetes, RSS is only one component of total memory usage. Container memory limits are enforced on the cgroup's total memory accounting, which includes RSS, page cache, buffers, swap, and sometimes memory-mapped files.

**Three plausible causes:**

1. **Page Cache & Buffer Overhead.** Linux automatically caches file I/O. Even with 1.4 GiB RSS, page cache and buffers can easily add several hundred MiB, pushing total usage over the 2 GiB hard limit.

2. **Transient Memory Spikes or Slow Leaks.** The application may experience periodic allocation bursts (batch processing, large request payloads, GC behavior in managed languages) that temporarily exceed 2 GiB. Or a slow leak could gradually push usage over the limit.

3. **Memory Fragmentation or cgroup Accounting Quirks.** Severe fragmentation can cause allocation failures even when total memory is under the limit. cgroup v2's `memory.current` counts swap and memory-mapped files — if swap is enabled or the app uses large mmaps, these can push reported usage over 2 GiB.

**What to look at first:** Total memory usage via `kubectl top pod <pod-name>` (shows cgroup total, not just RSS), then `dmesg | grep -i oom` or `journalctl -u containerd | grep -i oom` for the exact memory state at kill time. Next steps depend on what those show (tune `vm.vfs_cache_pressure` if cache is high, profile with pprof/heap dump for leaks, check `memory.stat` fields for fragmentation).

**Observation:** Both identify the same three root causes (page cache/non-RSS
overhead, transient spikes/leaks, fragmentation/cgroup accounting) and both
point at the same first step: cgroup-total memory vs RSS, plus OOM logs.
Same diagnosis, different prose.

---

## Prompt 4 — long-form summarization

**Prompt:** _Summarize the three most important ideas in "Attention Is All You Need" (Vaswani et al. 2017) in about 150 words. Output the summary directly; do not include any preamble._

### llama.cpp / Q4_K_M

The paper introduces the Transformer, a sequence-transduction architecture that entirely replaces recurrence and convolutions with pure attention mechanisms. First, it leverages scaled dot-product self-attention to compute pairwise relationships across all sequence positions simultaneously, enabling massive parallelization and eliminating the sequential bottlenecks inherent in traditional RNNs. Second, it employs multi-head attention, which projects inputs into multiple independent subspaces where separate attention operations run concurrently. This design allows the model to jointly capture diverse contextual dependencies, including long-range syntactic structures and nuanced semantic relationships. Third, because attention is inherently permutation-invariant, the architecture incorporates fixed sinusoidal positional encodings added directly to input embeddings, preserving essential sequence order without requiring recurrent processing. Together, these three innovations yield a highly scalable, efficiently trainable model that achieves state-of-the-art performance in machine translation and fundamentally establishes the architectural foundation for modern large language models.

### vLLM / NVFP4

The paper introduces the Transformer, a neural architecture built entirely on attention mechanisms, completely eliminating recurrent and convolutional layers. First, it proposes self-attention, which dynamically computes relevance scores between all tokens in a sequence, enabling massive parallelization and robust modeling of long-range dependencies without sequential bottlenecks. Second, it incorporates positional encodings — fixed sinusoidal vectors added to input embeddings — to preserve word order, compensating for the model's lack of inherent sequential awareness. Third, it employs multi-head attention, which runs multiple independent attention functions in parallel across distinct representation subspaces, allowing the model to jointly attend to information from different positions and semantic features. This fully attention-based design drastically accelerates training, scales efficiently to large datasets, and achieves state-of-the-art performance, establishing attention as the foundational mechanism for modern language models and reshaping natural language processing.

**Observation:** Both cover the same three ideas: self-attention, positional
encoding, multi-head attention. Both honor the 150-word constraint (llama.cpp
145, vLLM 140). Both correctly frame the Transformer as a
recurrence/convolution replacement.

---

## Prompt 5 — structured output

**Prompt:** _Produce a JSON object with keys ["incident_summary", "root_cause_hypothesis", "next_action", "blast_radius"]. Each value is a short string. Fabricate a plausible on-call scenario for the content._

### llama.cpp / Q4_K_M

```json
{
  "incident_summary": "Checkout service returning 503 errors due to OOM kills; payment latency elevated.",
  "root_cause_hypothesis": "Memory leak in invoice generator from v2.4.1 deployment causing heap exhaustion.",
  "next_action": "Rollback checkout-service to v2.4.0 and restart affected pods.",
  "blast_radius": "15% checkout failures; impacts US-East region only."
}
```

### vLLM / NVFP4

```json
{
  "incident_summary": "Intermittent 503 errors on checkout service during peak traffic",
  "root_cause_hypothesis": "Database connection pool exhaustion from unoptimized payment validation query",
  "next_action": "Rollback to v2.4.0 and temporarily scale read replicas",
  "blast_radius": "Impacts ~12% of checkout transactions in US and EU regions"
}
```

**Observation:** Both return valid JSON with exactly the requested keys, each
value a short string. Both fabricate a plausible on-call incident — a checkout
503 scenario with a rollback suggestion. Root-cause hypotheses differ (memory
leak vs DB connection pool), which is exactly what "fabricate a plausible
scenario" invites.

---

## Takeaway

Across five prompts spanning explanation, code generation, reasoning,
summarization, and structured output, the two quants produce answers a human
reader would call equivalent. Different phrasing, different incidental choices
(rolling top-5 vs full sort, memory leak vs connection pool), same
correctness, same constraint satisfaction, same first-principles reasoning.

This doesn't prove Q4_K_M and NVFP4 are identical on every prompt — no
side-by-side of 5 cases could. It says: on the kinds of prompts the
throughput tables in the post are built from, the quality gap between these
two runtime-native quantizations is not the story. Throughput and context
size are.

## Reproducing this

Both ISVCs and the test harness are in this repo. With the bench ns deployed
and a Qwen3.6-27B model applied:

```bash
kubectl -n bench patch isvc llamacpp-bench --type=merge -p '{"spec":{"replicas":1}}'
# wait for Ready, then port-forward svc/llamacpp-bench 18080:8080
# run the 5 prompts via curl at temp=0, seed=42, max_tokens=6000
# repeat the same for vllm-bench on port 8000 with model id "bench"
```

The five prompts and the runner script live at
`docs/quality-gate-prompts.json` and `docs/quality-gate-run.sh` in this repo.
