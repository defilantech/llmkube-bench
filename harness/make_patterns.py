"""Generate workload JSONL files into harness/patterns/.

Run once to produce chat / coding / long_context / agentic patterns.
Prompts are curated for the bake-off, not drawn from a secret eval set —
reproducibility matters more than novelty here.

    python -m harness.make_patterns
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "patterns"

# --- chat: 128-in / 256-out -------------------------------------------------

CHAT_PROMPTS = [
    "Explain how a Kubernetes liveness probe differs from a readiness probe in three sentences.",
    "What is tensor parallelism? Be concise.",
    "Summarize the role of the LLMKube operator in two sentences.",
    "Describe the difference between FP8 E4M3 and FP8 E5M2 for neural network weights.",
    "What does llama.cpp's --parallel flag do?",
    "Explain PagedAttention's main insight in 3 sentences.",
    "What is prefix caching and when does it help?",
    "Describe the difference between Q4_K_M and FP8 quantization.",
    "What is chunked prefill in vLLM?",
    "Explain the role of FlashInfer in LLM inference.",
    "What is the GGUF file format? One paragraph.",
    "Why would you pick vLLM over Ollama?",
    "Why would you pick llama.cpp over vLLM?",
    "What is speculative decoding in one paragraph.",
    "Describe how PodMonitor differs from ServiceMonitor.",
    "What is an InferenceService CR in LLMKube?",
    "Explain the purpose of a model init container.",
    "What does --enable-chunked-prefill do?",
    "What is a KV cache in an LLM? Keep it short.",
    "What does --tensor-split do in llama.cpp?",
]

# --- coding: ~1K-in / 1K-out ------------------------------------------------

CODING_SYSTEM = (
    "You are a careful senior engineer. Always include an approach overview, "
    "then the code, then a short note on what you would test next."
)

CODING_TASKS = [
    ("Write a Go function that walks a directory tree and returns the top 5 largest files. "
     "Use filepath.WalkDir. Stream results through a channel. Include error handling for "
     "permission denied. Include a small table-driven test."),
    ("Implement a Python coroutine that rate-limits to N requests per second using an "
     "asyncio.Semaphore plus a time-based token bucket. Show usage from a for-loop calling "
     "an async httpx client."),
    ("Write a small Rust CLI that reads a JSONL file and prints a histogram of an integer "
     "field specified by --field. Use clap. Handle malformed lines gracefully."),
    ("Implement a thread-safe LRU cache in Go with generics. Include get/put/len. Write a "
     "concurrent test that verifies no race conditions with -race."),
    ("Write a bash script that waits for a Kubernetes Deployment to be Available=true, "
     "timing out after a user-specified duration. Make it robust to kubectl intermittent "
     "failures (retry 3 times with backoff)."),
    ("Build a minimal Python CLI that fetches an OpenAI-compatible /v1/chat/completions "
     "endpoint, streams tokens, and prints tokens/sec to stderr. Use httpx."),
    ("Write a Go worker pool with panic recovery. N workers pull Task values off a buffered "
     "channel; panics in one worker do not kill the pool. Include a Stop() that drains."),
    ("Implement a small Kubernetes admission webhook in Go that rejects Pods without a "
     "resources.requests.cpu value. Use kubebuilder/controller-runtime if applicable."),
    ("Write a Python function that merges two monotonically-increasing token streams from "
     "async generators into one monotonically-increasing stream, preserving order."),
    ("Write a TypeScript function that debounces an async function. The last call wins. "
     "Pending promises from earlier calls resolve with the latest call's result."),
    ("Implement a Go JSON parser that reads newline-delimited JSON from a reader and emits "
     "typed records via a generic channel. Handle partial reads correctly."),
    ("Write a Python script that scrapes Prometheus /api/v1/query_range for a given query "
     "and writes CSV. Include retries with exponential backoff."),
    ("Build a minimal Redis-like in-memory KV store in Go with SET/GET/DEL and TTL. Use "
     "time.AfterFunc for expiry. Write a test for expiry correctness."),
    ("Implement a command-line fuzzy file finder in Rust using ripgrep-style patterns. "
     "Read file list from stdin; use crossterm for the TUI."),
    ("Write a Python async queue processor that dequeues jobs from Redis (with LPUSH/BRPOP) "
     "and dispatches them to a worker function with concurrency limit N. Include graceful "
     "shutdown on SIGTERM."),
    ("Build a Go HTTP handler that streams server-sent events for a long-running job's "
     "status. Close cleanly when the client disconnects."),
    ("Write a Python context manager that starts a subprocess (e.g., ffmpeg), captures its "
     "stderr, and reraises on nonzero exit with the captured stderr included."),
    ("Implement a Go ring buffer with fixed capacity. Append is O(1); Read(n) returns up "
     "to n most recent items in chronological order. Include a test."),
    ("Write a Python async function that reads a CSV from S3 in chunks, transforms rows "
     "via a user-supplied callback, and writes the result to a new S3 object without "
     "materializing the whole file in memory."),
    ("Implement a Go function that computes a streaming median (rolling window of N) "
     "using two heaps. Include benchmarks."),
]

# --- long_context: ~8K-in / 1K-out ------------------------------------------

LONG_CONTEXT_PRELUDE = (
    "You are reviewing a production Go codebase for a Kubernetes operator. "
    "Read the following source file and respond with: "
    "(1) three concrete correctness risks, "
    "(2) two performance opportunities, "
    "(3) one API design concern. "
    "Be specific — cite function names and line behavior, not generalities.\n\n"
)

# A synthesized ~8K-token fake-but-plausible Go file. We don't care about real
# code quality; we care about realistic token shape and structure.
LONG_CONTEXT_BODY = """package controller

// --- this file is generated for the llmkube-bench long-context pattern ---
// It is intentionally verbose and full of structure; it stands in for a real
// production Go source file when stressing long-context behavior.
""" + "\n".join(
    f"""
// Helper{n} coordinates a reconcile sub-step for the InferenceService
// controller. It is called from Reconcile in pkg/controller/inferenceservice.go
// and should never return an error that does not deserve a requeue.
func Helper{n}(ctx context.Context, r *Reconciler, isvc *inferencev1alpha1.InferenceService) (ctrl.Result, error) {{
    log := logf.FromContext(ctx).WithValues("step", "helper-{n}")
    if isvc == nil {{
        log.Info("nil InferenceService, nothing to do")
        return ctrl.Result{{}}, nil
    }}
    if isvc.Spec.ModelRef == "" {{
        log.Info("modelRef not set; skipping helper-{n}")
        return ctrl.Result{{}}, nil
    }}
    var model inferencev1alpha1.Model
    if err := r.Get(ctx, client.ObjectKey{{Namespace: isvc.Namespace, Name: isvc.Spec.ModelRef}}, &model); err != nil {{
        if errors.IsNotFound(err) {{
            log.Info("model not yet created", "name", isvc.Spec.ModelRef)
            return ctrl.Result{{RequeueAfter: 15 * time.Second}}, nil
        }}
        return ctrl.Result{{}}, fmt.Errorf("get model: %w", err)
    }}
    if model.Status.Phase != "Ready" {{
        log.Info("model not ready", "phase", model.Status.Phase)
        return ctrl.Result{{RequeueAfter: 30 * time.Second}}, nil
    }}
    // ... [pretend this does real reconciliation work]
    return ctrl.Result{{}}, nil
}}""".strip() for n in range(1, 36)
)

LONG_CONTEXT_ASKS = [
    "Focus your review on the error handling around Get calls.",
    "Focus your review on requeue behavior and how it interacts with controller-runtime backoff.",
    "Focus your review on log line cardinality and structured logging choices.",
    "Focus your review on how NotFound is distinguished from other transient errors.",
    "Focus your review on context propagation and potential leaks.",
    "Focus your review on what would happen under a thundering herd of reconciles.",
    "Focus your review on helper naming and whether there's an extractable abstraction.",
    "Focus your review on what metrics you would add and why.",
    "Focus your review on what tests you would add first and what they would catch.",
    "Focus your review on how you would split this file if it got 3x larger.",
]

# --- agentic: shared 4K prefix + small delta --------------------------------

AGENTIC_SYSTEM = """You are an autonomous platform-engineering coding agent operating inside a
production Kubernetes environment. You have access to the following tools:

- kubectl_get(resource: str, name: str, namespace: str | None) -> str
    Returns a YAML-serialized representation of a Kubernetes resource.
    Use this sparingly; prefer kubectl_list when exploring.

- kubectl_list(resource: str, namespace: str | None, selector: str | None) -> str
    Returns a YAML-serialized list of Kubernetes resources.

- kubectl_logs(pod: str, namespace: str, container: str | None, since: str | None) -> str
    Returns pod logs. Default since window is 10 minutes.

- kubectl_apply(yaml: str, dry_run: bool = True) -> str
    Applies a manifest. dry_run=True is strongly preferred until the user
    has explicitly confirmed a change. Never pass dry_run=False without
    checking with the user first.

- prometheus_query(promql: str, time_range: str = "1h") -> str
    Runs a PromQL instant or range query. Use this for any metric-based
    investigation (GPU util, pod restarts, request rate, error budget).

- grafana_link(dashboard_uid: str, params: dict[str, str]) -> str
    Builds a Grafana URL with templated variables filled in. Include these
    in your final response so a human can visually verify.

You operate under the following rules:
1. Always surface the reasoning behind any proposed change — never apply
   silently. Even with dry_run=True, state what you expect to see.
2. Do not propose changes to Secret resources or ClusterRole / ClusterRoleBinding
   without explicit human sign-off in the same turn.
3. Prefer investigation over action. If a failure is ambiguous, gather more
   signal (logs, metrics, events) before proposing a mitigation.
4. When a fix requires multiple steps, describe them as an ordered checklist
   with a rollback plan for each step.
5. Be explicit about blast radius: name the namespaces and workloads
   potentially affected by any proposed action.
6. Preserve idempotency: all proposed manifests should be safe to re-apply
   without drift.
7. Use structured output: {"plan": [...], "actions": [...], "risk": "..."}

You are currently working inside the `platform-sre` team. You have been
paged for investigation. The alerting system fired InferenceServiceNotReady
on the `shadowstack` cluster. You have one-shot authority to investigate
but not to remediate without confirmation.

The alert context you received:
- Firing for: 17 minutes
- Cluster: shadowstack
- Namespace: bench
- InferenceService: llamacpp-bench
- Last-known phase: Creating (stuck)
- On-call: chris (human, will review your findings)

When you produce output, always include:
(a) a summary of what you investigated
(b) the most likely root cause with evidence
(c) a proposed action plan with blast radius
(d) what a human should verify before confirming
"""

AGENTIC_USER_ASKS = [
    "Investigate and report. Start by looking at the InferenceService status and related pod events.",
    "I just saw the image pull secret change. Could that be related? Investigate.",
    "The controller-manager seems to have restarted 40 minutes ago. Does that correlate?",
    "There's a new NetworkPolicy in the namespace. Could it be blocking the init container?",
    "Someone applied a resource quota yesterday. Is the pod being denied for quota reasons?",
    "Investigate and give me a 3-bullet summary I can paste in our incident channel.",
    "The model-cache PVC filled up yesterday. Could the init container be blocked on disk?",
    "We pushed a new LLMKube CRD schema this morning. Could that be causing the stuck state?",
    "Our DCGM exporter shows GPU0 at 0% utilization. Check if the pod is even scheduled.",
    "The NodeReady condition on shadowstack flapped twice. Did that affect the ISVC pod?",
    "Investigate and include a timeline of what happened in the last 30 minutes.",
    "I think the readiness probe is wrong. Can you look at the probe config and recent probe outcomes?",
    "Tell me whether this is a config problem, a cluster problem, or an image problem.",
    "Give me three things to check that would falsify your root-cause hypothesis.",
    "I need to present this to leadership in 10 minutes. Give me an executive summary.",
    "What would scaling the ISVC to 0 and back do? Walk through it before I approve.",
    "If we leave this alone for another hour, what breaks downstream?",
    "Double-check that the GPU resource requests are actually satisfiable on the node.",
    "Would restarting the controller-manager help or hurt here? Explain why.",
    "Summarize the state for a post-incident write-up in under 150 words.",
]


def emit(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def build_chat() -> list[dict]:
    return [
        {
            "id": f"chat-{i:03d}",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 256,
        }
        for i, p in enumerate(CHAT_PROMPTS, 1)
    ]


def build_coding() -> list[dict]:
    return [
        {
            "id": f"coding-{i:03d}",
            "messages": [
                {"role": "system", "content": CODING_SYSTEM},
                {"role": "user", "content": t},
            ],
            "max_tokens": 1024,
        }
        for i, t in enumerate(CODING_TASKS, 1)
    ]


def build_long_context() -> list[dict]:
    return [
        {
            "id": f"long-{i:03d}",
            "messages": [
                {
                    "role": "user",
                    "content": LONG_CONTEXT_PRELUDE + LONG_CONTEXT_BODY + "\n\n" + ask,
                },
            ],
            "max_tokens": 1024,
        }
        for i, ask in enumerate(LONG_CONTEXT_ASKS, 1)
    ]


def build_agentic() -> list[dict]:
    # All 20 share the same system prompt exactly — this is what vLLM's
    # automatic prefix cache keys on. Token count: ~4K for the system prompt.
    prefix_id = "agentic-isvc-investigation-v1"
    return [
        {
            "id": f"agentic-{i:03d}",
            "prefix_id": prefix_id,
            "messages": [
                {"role": "system", "content": AGENTIC_SYSTEM},
                {"role": "user", "content": ask},
            ],
            "max_tokens": 512,
        }
        for i, ask in enumerate(AGENTIC_USER_ASKS, 1)
    ]


def main() -> None:
    emit(OUT / "chat.jsonl", build_chat())
    emit(OUT / "coding.jsonl", build_coding())
    emit(OUT / "long_context.jsonl", build_long_context())
    emit(OUT / "agentic.jsonl", build_agentic())


if __name__ == "__main__":
    main()
