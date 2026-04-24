# Contributing

Thanks for considering a contribution. This repo is a benchmark and
reproducibility artifact more than an actively developed tool, but issues
and PRs that improve accuracy, clarity, or reproducibility are welcome.

## Reporting issues

When opening an issue, please include:

- **Hardware**: GPU model, VRAM per card, number of cards, host OS
- **Cluster**: Kubernetes distribution + version, LLMKube version,
  GPU driver / Operator version
- **Runtime image**: exact tag or digest of the image you ran
- **Pattern and concurrency**: which cell(s) of the matrix
- **Expected vs. observed**: what you saw, what the README/METHOD said
  you should see, and any logs or `summary.csv` snippets that illustrate
  the gap

See [`docs/METHOD.md`](docs/METHOD.md) for the methodology the original
runs used. If you think the methodology itself is wrong, open an issue —
we'd rather hear it than ship bad numbers.

## Pull requests

Appreciated especially for:

- Reproducing results on new hardware (add a results/ subdirectory,
  update the README with a pointer)
- Adding runtime or model variants (keep the manifests structured the
  same way as the existing ones)
- Harness improvements (better error reporting, additional Prometheus
  metrics, fixes to the JSONL emitters)

Please:

1. Keep PRs focused — one concern per PR
2. Update docs where relevant (METHOD.md, QUALITY-GATE.md, or README)
3. If you change the harness or `bench.sh`, include a short description
   of what you verified locally (ran the smoke, ran one cell end-to-end,
   etc.)

## License

All contributions are accepted under the repo's Apache 2.0 license. By
opening a PR you agree that your contribution is licensed on those terms.
