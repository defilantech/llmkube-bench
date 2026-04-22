# Quality gate

This isn't a benchmark. It's a side-by-side so readers can judge for themselves
whether the quant-per-runtime choice (Q4_K_M GGUF vs FP8 safetensors) produces
a visible quality gap on the kinds of prompts this bake-off benchmarks against.

Each prompt is run through both runtimes with temperature=0 and seed=42. Exact
outputs are pasted below.

---

## Prompt 1 — simple explanation

**Prompt:** _Explain how a Kubernetes liveness probe differs from a readiness probe in three sentences._

### llama.cpp / Q4_K_M

_(to be filled in after smoke run)_

### vLLM / FP8

_(to be filled in after smoke run)_

---

## Prompt 2 — code generation

**Prompt:** _Write a Go function that walks a directory tree and returns the top 5 largest files. Use filepath.WalkDir. Include error handling for permission denied._

### llama.cpp / Q4_K_M

_(to be filled in after smoke run)_

### vLLM / FP8

_(to be filled in after smoke run)_

---

## Prompt 3 — reasoning under constraint

**Prompt:** _A container with 2 GiB memory is OOMKilled repeatedly. Its RSS is 1.4 GiB. What are three plausible causes and what would you look at first?_

### llama.cpp / Q4_K_M

_(to be filled in after smoke run)_

### vLLM / FP8

_(to be filled in after smoke run)_

---

## Prompt 4 — long-form summarization

**Prompt:** _Summarize the three most important ideas in "The Annotated Transformer" (Rush 2018) in 150 words._

### llama.cpp / Q4_K_M

_(to be filled in after smoke run)_

### vLLM / FP8

_(to be filled in after smoke run)_

---

## Prompt 5 — structured output

**Prompt:** _Produce a JSON object with keys ["incident_summary", "root_cause_hypothesis", "next_action", "blast_radius"]. Each value is a short string. Fabricate a plausible on-call scenario for the content._

### llama.cpp / Q4_K_M

_(to be filled in after smoke run)_

### vLLM / FP8

_(to be filled in after smoke run)_
