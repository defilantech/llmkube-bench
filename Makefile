SHELL := /bin/bash

RESULTS_DIR ?= results/$(shell date +%Y-%m-%d)-local
KUBECTL_CONTEXT ?= $(shell kubectl config current-context 2>/dev/null)

PY ?= python3

.PHONY: help
help:
	@echo "llmkube-bench — llama.cpp vs vLLM on Kubernetes"
	@echo ""
	@echo "Targets:"
	@echo "  install     install harness Python deps (pip)"
	@echo "  patterns    regenerate workload JSONL files"
	@echo "  smoke       deploy each runtime, run one quick cell, scale to 0"
	@echo "  bench       run the full 32-cell matrix (~6h, mostly unattended)"
	@echo "  llamacpp    run only the llama.cpp runtime"
	@echo "  vllm        run only the vLLM runtime"
	@echo "  analyze     (re-)aggregate results from RESULTS_DIR"
	@echo "  teardown    scale both ISVCs to 0"
	@echo "  clean       remove __pycache__"
	@echo ""
	@echo "Variables: RESULTS_DIR=$(RESULTS_DIR)"
	@echo "           KUBECTL_CONTEXT=$(KUBECTL_CONTEXT)"

.PHONY: install
install:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found; install from https://github.com/astral-sh/uv"; exit 1; }
	uv venv --clear -q .venv
	uv pip install -r harness/requirements.txt --python .venv/bin/python

.PHONY: patterns
patterns:
	$(PY) -m harness.make_patterns

.PHONY: smoke
smoke:
	RESULTS_DIR=$(RESULTS_DIR) KUBECTL_CONTEXT=$(KUBECTL_CONTEXT) ./bench.sh smoke

.PHONY: bench
bench:
	RESULTS_DIR=$(RESULTS_DIR) KUBECTL_CONTEXT=$(KUBECTL_CONTEXT) ./bench.sh full

.PHONY: llamacpp
llamacpp:
	RESULTS_DIR=$(RESULTS_DIR) KUBECTL_CONTEXT=$(KUBECTL_CONTEXT) ./bench.sh runtime llamacpp

.PHONY: vllm
vllm:
	RESULTS_DIR=$(RESULTS_DIR) KUBECTL_CONTEXT=$(KUBECTL_CONTEXT) ./bench.sh runtime vllm

.PHONY: analyze
analyze:
	@echo "Re-aggregating $(RESULTS_DIR)/summary.csv"
	@RESULTS_DIR=$(RESULTS_DIR) ./bench.sh full --aggregate-only 2>/dev/null || true
	@if [[ -f $(RESULTS_DIR)/summary.csv ]]; then column -t -s, $(RESULTS_DIR)/summary.csv | less -S; fi

.PHONY: teardown
teardown:
	KUBECTL_CONTEXT=$(KUBECTL_CONTEXT) ./bench.sh teardown

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
