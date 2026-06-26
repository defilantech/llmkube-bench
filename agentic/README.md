# llmkube-bench: agentic suite

Runs a fixed set of (model x task) through Foreman on a target hardware tier, emits one
structured JSON record per run, runs a perf load sweep, and renders a comparison report.

    python -m runner.cli run --config config.example.yaml
    python -m runner.cli report --records records/ --out reports/latest.md

See `../llmkube-internal` specs for design. Requires `kubectl` and `gh` on PATH.
