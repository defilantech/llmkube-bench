# Bench-runner — run the bake-off from inside the cluster

The full bake-off is orchestrated by a single Kubernetes Job that reuses
the same `bench.sh` you'd run locally. Inside the cluster it skips
port-forwarding (talks to ISVCs via cluster DNS), writes results to a
PVC, and authenticates to the API server through its own ServiceAccount.

## Flow

```
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ Local workstation            │       │ Your K8s node                │
│                              │       │                              │
│ rsync ./ → ~/llmkube-bench ──┼──────▶│ <your-workspace>/llmkube-bench│
│                              │       │          │                   │
│                              │       │          ▼                   │
│                              │       │ Kaniko Job (hostPath)        │
│                              │       │          │                   │
│                              │       │          ▼                   │
│                              │       │ registry.registry.svc:5000   │
│                              │       │   └ llmkube-bench:v1         │
│                              │       │          │                   │
│                              │       │          ▼                   │
│                              │       │ bench-runner Job             │
│                              │       │   └ bench.sh full            │
│                              │       │      → bench-results PVC     │
└──────────────────────────────┘       └──────────────────────────────┘
```

## Deploy (from a workstation)

```bash
# 1. Rsync the repo to the node (Kaniko reads from hostPath)
rsync -av --delete --exclude='.venv' --exclude='results/' \
  ./ <your-user>@<your-node>:<your-workspace>/llmkube-bench/

# 2. One-time setup — namespace, secret, RBAC, PVC, podmonitor
kubectl --context <your-context> apply -f manifests/bench-namespace.yaml
# (hf-token secret must already exist in bench; see ../../README.md)
kubectl --context <your-context> apply -f manifests/podmonitor-vllm.yaml
kubectl --context <your-context> apply -f manifests/bench-runner/rbac.yaml
kubectl --context <your-context> apply -f manifests/bench-runner/results-pvc.yaml

# 3. Build the image (Kaniko, a few minutes on the GPU node)
kubectl --context <your-context> apply -f manifests/bench-runner/kaniko-build.yaml
kubectl --context <your-context> -n bench wait --for=condition=complete --timeout=15m job/llmkube-bench-build

# 4. Launch the bench (the Job runs ~5h)
kubectl --context <your-context> apply -f manifests/bench-runner/bench-job.yaml
kubectl --context <your-context> -n bench logs -f job/bench-runner

# 5. When the Job is Complete, copy results out
BENCH_POD=$(kubectl --context <your-context> -n bench get pod -l app=bench-runner -o jsonpath='{.items[0].metadata.name}')
kubectl --context <your-context> -n bench cp "$BENCH_POD:/results" ./results-from-cluster
```

## Re-runs

```bash
# update the repo, rsync again
rsync -av --delete --exclude='.venv' --exclude='results/' \
  ./ <your-user>@<your-node>:<your-workspace>/llmkube-bench/

# rebuild the image
kubectl --context <your-context> -n bench delete job/llmkube-bench-build --ignore-not-found
kubectl --context <your-context> apply -f manifests/bench-runner/kaniko-build.yaml

# restart the bench
kubectl --context <your-context> -n bench delete job/bench-runner --ignore-not-found
kubectl --context <your-context> apply -f manifests/bench-runner/bench-job.yaml
```

## Teardown

```bash
kubectl --context <your-context> -n bench delete job/llmkube-bench-build job/bench-runner --ignore-not-found
kubectl --context <your-context> -n bench delete pvc bench-results  # only if you've pulled results
```
