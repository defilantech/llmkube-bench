# llmkube-bench: agentic suite

This package runs a fixed set of (model x task) pairs through Foreman on one hardware tier, emits one structured JSON record per run, runs a perf load sweep for each model, and renders a markdown comparison report. It is the "agentic capability" companion to the serving bake-off at the repo root (the llama.cpp vs vLLM throughput suite). Design and reference data live in the private `llmkube-internal` specs/research tree.

## Requirements

- Python 3.12+ (stdlib only for the runner core)
- `pyyaml` and `pytest` from `requirements.txt`
- `kubectl` and `gh` on PATH for live runs against a real cluster

Install:

```bash
python3 -m pip install -r requirements.txt
# or, if the repo uses uv:
uv pip install -r requirements.txt
```

Tests need no cluster and no live credentials:

```bash
python3 -m pytest -q
```

## Layout

```
runner/
  config.py      load and validate YAML config into typed dataclasses
  record.py      RunRecord: the canonical JSON schema for one (model, task) result
  behavior.py    extract verdict/gate/turns/tool_calls/files_edited from a Foreman transcript
  perf.py        run a concurrency sweep, compute p50/p95 latency and tok/s per level
  cluster.py     KubectlCluster: drive kubectl/Foreman for live runs; FakeCluster for tests
  orchestrate.py run_matrix: outer loop over models then tasks, producing List[RunRecord]
  report.py      render a list of RunRecord objects to a markdown comparison table
  cli.py         argparse entry point: `run` and `report` subcommands

records/         JSON records written by `run` (one file per (model, task))
reports/         markdown reports written by `report`
tests/           pytest unit tests (14 tests, no cluster required)
```

## Usage

### Run the full matrix

```bash
python -m runner.cli run \
  --config config.example.yaml \
  --records records \
  --endpoint <gateway>/v1 \
  --token <jwt> \
  --harness-version 0.8.17
```

For each model the runner:
1. Scales that InferenceService to 1 replica and all others to 0 (GPU serialization).
2. Repoints the `fleet-router` coder backend to the active model.
3. Runs a perf concurrency sweep (single occupant).
4. Dispatches each corpus task as a Foreman `AgenticTask` on its own branch.
5. Waits for terminal status, extracts behavior from the transcript, and writes one JSON record per (model, task) to `--records`.

`--endpoint` defaults to `http://localhost:18080`; `--token` defaults to empty.

### Render a report

```bash
python -m runner.cli report --records records --out reports/latest.md
```

Reads all `*.json` files from `--records` and renders a markdown table grouped by model, with perf and capability columns.

### Run tests

```bash
python3 -m pytest -q
```

## Config walkthrough

`config.example.yaml` has four top-level blocks:

**`hardware`**: identifies the target tier.

| key | meaning |
|-----|---------|
| `tier` | human label written into every record (e.g. `strix-halo-gfx1151`) |
| `context` | kubectl context to use |
| `namespace` | Kubernetes namespace |
| `router` | ModelRouter resource name (e.g. `fleet-router`) |
| `gateway_backend_index` | index in `fleet-router.spec.backends[]` that is the swappable coder slot |

**`agent`**: name of the Foreman `Agent` resource that routes coder tasks (e.g. `ornith-coder-lever`).

**`models`**: list of models to benchmark. Each entry:

| key | meaning |
|-----|---------|
| `name` | display name written into records |
| `quant` | quantization label (e.g. `Q4_K_M`) |
| `params_b` | parameter count in billions |
| `arch` | `moe` or `dense` |
| `inference_service` | InferenceService resource name to scale |
| `model_cutoff` | training cutoff date (ISO 8601); compared to `filed_date` to flag contamination |

**`corpus`**: list of tasks. Each entry:

| key | meaning |
|-----|---------|
| `id` | short slug used in record filenames and branch names |
| `repo` | GitHub repo (`owner/name`) |
| `issue` | issue number |
| `filed_date` | date the issue was filed; `model_cutoff < filed_date` means contamination-free |
| `kind` | task kind (currently `issue-fix`) |
| `difficulty` | `clean` or `gotcha` |

The example corpus deliberately mixes one clean issue and one gotcha issue to probe both normal convergence and adversarial corner cases.

**`perf`**: controls the load sweep.

| key | meaning |
|-----|---------|
| `concurrencies` | list of concurrency levels to test |
| `reqs_per_level` | number of requests per concurrency level |
| `max_tokens` | max tokens per request |

## Caveats

**Gateway JWT expiry.** The gateway token expires roughly every 10 hours. A stale token surfaces at turn 1 as `401: Jwt is expired`. Mint a fresh token via Keycloak client_credentials before starting a run and pass it via `--token`. Alternatively, patch the in-cluster `gateway-token` secret with a fresh value.

**GPU serialization and router repointing.** The Strix has a single accelerator, so the runner serves one model at a time. For each model it scales that InferenceService to 1 and the others to 0, and patches the `fleet-router` backend at `gateway_backend_index` to point at the active InferenceService. The live AMD coder slot therefore changes during a run and is left pointing at the last model in the matrix when the run finishes. If you need a specific model served after a run, scale your preferred InferenceService back to 1 and patch the backend index to point at it.

## Status

Slice 3 is a thin end-to-end runner. The pure logic (config, record, behavior, perf stats, report rendering) is unit-tested with 14 tests and no cluster required. The cluster-driving paths (`serve_only`, `dispatch`, `wait_terminal`, `transcript_of`) are exercised against a live Strix by hand, not in CI.

Deliberately out of scope for this slice:

- A generalized "point at any repo" CLI
- A rotating external multi-repo corpus
- A published web leaderboard
- A multi-hardware matrix (multiple tiers in one run)
- N-of-K repeat runs with statistical aggregation
- Automatic gateway JWT mint/refresh
