#!/usr/bin/env bash
# llmkube-bench orchestrator — deploys each runtime, runs the matrix,
# scales to zero between runs. Inspired by llmkube-internal/benchmarks/
# turboquant/turboquant-benchmark.sh but explicit and trimmed.
#
# Usage:
#     ./bench.sh smoke
#     ./bench.sh full RESULTS_DIR=results/2026-04-22-shadowstack
#     ./bench.sh runtime llamacpp        # deploy + bench one runtime
#     ./bench.sh teardown
#
# Environment overrides:
#     KUBECTL_CONTEXT     default: shadowstack
#     NAMESPACE           default: bench
#     PROMETHEUS_URL      default: http://localhost:9090 (via port-forward)
#     PATTERNS            default: chat coding long_context agentic
#     CONCURRENCIES       default: 1 4 16 64
#     DURATION            default: 5m
#     WARMUP              default: 2m
#     READY_TIMEOUT_S     default: 900 (15 min; vLLM pulls ~28 GB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Prefer the repo's venv for harness invocations so Prometheus/httpx deps
# come from a pinned set rather than the host's Python.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY="python3"
fi

KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-shadowstack}"
NAMESPACE="${NAMESPACE:-bench}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
PATTERNS="${PATTERNS:-chat coding long_context agentic}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
DURATION="${DURATION:-5m}"
WARMUP="${WARMUP:-2m}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
RESULTS_DIR="${RESULTS_DIR:-results/$(date +%Y-%m-%d)-local}"

KUBECTL=(kubectl --context "$KUBECTL_CONTEXT")

log() { printf '[bench %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

kill_portforward() {
    if [[ -n "${PF_PID:-}" ]] && kill -0 "$PF_PID" 2>/dev/null; then
        kill "$PF_PID" 2>/dev/null || true
        wait "$PF_PID" 2>/dev/null || true
    fi
    unset PF_PID
}
trap kill_portforward EXIT

ensure_ns() {
    "${KUBECTL[@]}" apply -f manifests/bench-namespace.yaml >/dev/null
    if ! "${KUBECTL[@]}" -n "$NAMESPACE" get secret hf-token >/dev/null 2>&1; then
        echo "ERROR: Secret 'hf-token' missing in namespace '$NAMESPACE'" >&2
        echo "       create one with: kubectl -n $NAMESPACE create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx" >&2
        exit 1
    fi
    "${KUBECTL[@]}" apply -f manifests/podmonitor-vllm.yaml >/dev/null
}

apply_runtime() {
    local runtime="$1"
    log "applying $runtime manifests"
    "${KUBECTL[@]}" apply -f "manifests/$runtime/model.yaml" >/dev/null
    "${KUBECTL[@]}" apply -f "manifests/$runtime/isvc.yaml" >/dev/null
    # Ensure replicas=1 (we scale to 0 between runs).
    local isvc="${runtime}-bench"
    "${KUBECTL[@]}" -n "$NAMESPACE" patch inferenceservices.inference.llmkube.dev "$isvc" \
        --type merge -p '{"spec":{"replicas":1}}' >/dev/null || true
}

wait_ready() {
    local isvc="$1"
    local start=$(date +%s)
    log "waiting for $isvc (timeout ${READY_TIMEOUT_S}s)"
    while true; do
        local phase
        phase=$("${KUBECTL[@]}" -n "$NAMESPACE" get inferenceservices.inference.llmkube.dev "$isvc" \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        if [[ "$phase" == "Ready" ]]; then
            log "$isvc is Ready after $(( $(date +%s) - start ))s"
            return 0
        fi
        if (( $(date +%s) - start > READY_TIMEOUT_S )); then
            log "ERROR: $isvc did not reach Ready within ${READY_TIMEOUT_S}s (last phase=$phase)"
            "${KUBECTL[@]}" -n "$NAMESPACE" describe inferenceservices.inference.llmkube.dev "$isvc" >&2 || true
            return 1
        fi
        sleep 5
    done
}

start_portforward() {
    local svc="$1"
    local local_port="$2"
    local remote_port="$3"
    kill_portforward
    log "port-forward $svc ${local_port}->${remote_port}"
    "${KUBECTL[@]}" -n "$NAMESPACE" port-forward "svc/$svc" "${local_port}:${remote_port}" \
        >/tmp/bench-pf.log 2>&1 &
    PF_PID=$!
    # Wait up to 10s for the tunnel to open
    for _ in $(seq 1 20); do
        sleep 0.5
        if curl -fsS "http://localhost:${local_port}/v1/models" >/dev/null 2>&1 \
                || curl -fsS "http://localhost:${local_port}/health" >/dev/null 2>&1; then
            return 0
        fi
    done
    log "WARN: endpoint not yet responsive on localhost:$local_port; proceeding anyway"
}

run_cell() {
    local runtime="$1"
    local pattern="$2"
    local concurrency="$3"
    local local_port="$4"
    local runtime_dir="$RESULTS_DIR/raw/$runtime/$pattern"
    mkdir -p "$runtime_dir"
    local output="$runtime_dir/c${concurrency}.jsonl"
    local prom_output="$runtime_dir/c${concurrency}.prom.json"
    local endpoint="http://localhost:${local_port}/v1/chat/completions"

    log "CELL runtime=$runtime pattern=$pattern c=$concurrency"
    local start_ts
    start_ts=$("$PY" -c 'import time; print(time.time())')

    "$PY" -m harness.run run \
        --endpoint "$endpoint" \
        --pattern "$pattern" \
        --concurrency "$concurrency" \
        --duration "$DURATION" \
        --warmup "$WARMUP" \
        --runtime "$runtime" \
        --output "$output"

    PROMETHEUS_URL="$PROMETHEUS_URL" "$PY" -m harness.prom_snapshot \
        --start "$start_ts" \
        --end 0 \
        --output "$prom_output" || log "prom snapshot failed (non-fatal)"
}

bench_one_runtime() {
    local runtime="$1"
    local svc port
    case "$runtime" in
        llamacpp) svc="llamacpp-bench"; port="8080" ;;
        vllm)     svc="vllm-bench";     port="8000" ;;
        *) echo "unknown runtime: $runtime" >&2; return 1 ;;
    esac

    apply_runtime "$runtime"
    wait_ready "${runtime}-bench"
    start_portforward "$svc" "$port" "$port"

    for pattern in $PATTERNS; do
        for c in $CONCURRENCIES; do
            run_cell "$runtime" "$pattern" "$c" "$port"
        done
    done

    kill_portforward
    log "scaling $runtime ISVC to 0"
    "${KUBECTL[@]}" -n "$NAMESPACE" patch inferenceservices.inference.llmkube.dev "${runtime}-bench" \
        --type merge -p '{"spec":{"replicas":0}}' >/dev/null
}

aggregate() {
    local out="$RESULTS_DIR/summary.csv"
    mkdir -p "$RESULTS_DIR"
    {
        echo "runtime,pattern,concurrency,requests_total,requests_ok,success_rate,gen_tokens_total,wall_s,throughput_gen_tps,ttft_p50_ms,ttft_p95_ms,ttft_p99_ms,itl_p50_ms,itl_p95_ms,itl_mean_ms"
        for runtime_dir in "$RESULTS_DIR"/raw/*/; do
            runtime=$(basename "$runtime_dir")
            for pattern_dir in "$runtime_dir"*/; do
                pattern=$(basename "$pattern_dir")
                for jsonl in "$pattern_dir"c*.jsonl; do
                    [[ -f "$jsonl" ]] || continue
                    # Ask the harness for a summary, then project to CSV
                    "$PY" -m harness.run summarize "$jsonl" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
fields = ["runtime","pattern","concurrency","requests_total","requests_ok","success_rate","gen_tokens_total","wall_s","throughput_gen_tps","ttft_p50_ms","ttft_p95_ms","ttft_p99_ms","itl_p50_ms","itl_p95_ms","itl_mean_ms"]
print(",".join(str(d.get(k, "")) for k in fields))'
                done
            done
        done
    } > "$out"
    log "aggregated $(wc -l < "$out") lines to $out"
}

cmd_smoke() {
    ensure_ns
    PATTERNS="chat" CONCURRENCIES="1" DURATION="30s" WARMUP="10s" bench_one_runtime "llamacpp"
    PATTERNS="chat" CONCURRENCIES="1" DURATION="30s" WARMUP="10s" bench_one_runtime "vllm"
    aggregate
    log "smoke complete — review $RESULTS_DIR/summary.csv"
}

cmd_full() {
    ensure_ns
    bench_one_runtime "llamacpp"
    bench_one_runtime "vllm"
    aggregate
}

cmd_runtime() {
    ensure_ns
    bench_one_runtime "$1"
    aggregate
}

cmd_teardown() {
    log "scaling both ISVCs to 0 (keeping Model CRs and cache)"
    for isvc in llamacpp-bench vllm-bench; do
        "${KUBECTL[@]}" -n "$NAMESPACE" patch inferenceservices.inference.llmkube.dev "$isvc" \
            --type merge -p '{"spec":{"replicas":0}}' >/dev/null 2>&1 || true
    done
}

main() {
    local cmd="${1:-full}"
    shift || true
    # Absorb optional KEY=VALUE overrides from remaining args
    for kv in "$@"; do
        if [[ "$kv" == *=* ]]; then
            export "$kv"
        fi
    done
    mkdir -p "$RESULTS_DIR"
    case "$cmd" in
        smoke)    cmd_smoke ;;
        full)     cmd_full ;;
        runtime)  cmd_runtime "${1:-llamacpp}" ;;
        teardown) cmd_teardown ;;
        *) echo "usage: $0 {smoke|full|runtime <name>|teardown} [KEY=VALUE ...]" >&2; exit 2 ;;
    esac
}

main "$@"
