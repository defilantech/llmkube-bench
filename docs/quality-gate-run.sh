#!/usr/bin/env bash
# Run the 5 quality-gate prompts against an OpenAI-compatible endpoint and
# append results to outfile. Used to reproduce docs/QUALITY-GATE.md.
#
# Usage:
#   ./quality-gate-run.sh <endpoint> <outfile> <model-id>
#
# Example:
#   ./quality-gate-run.sh http://localhost:18080 qg-llamacpp.txt model.gguf
#   ./quality-gate-run.sh http://localhost:18000 qg-vllm.txt bench
#
# Requires jq and curl. Set max_tokens high enough for thinking + answer
# (Qwen3.6-27B reasoning chains can run 2000-4000 tokens on complex prompts).

set -euo pipefail

endpoint="$1"
outfile="$2"
model="$3"

prompts_file="$(dirname "$0")/quality-gate-prompts.json"

: > "$outfile"
jq -c '.[]' "$prompts_file" | while read -r item; do
  id=$(echo "$item" | jq -r '.id')
  title=$(echo "$item" | jq -r '.title')
  prompt=$(echo "$item" | jq -r '.prompt')
  echo ">>> Prompt $id ($title)" | tee -a "$outfile"
  body=$(jq -nc --arg m "$model" --arg p "$prompt" '{
    model: $m,
    temperature: 0,
    seed: 42,
    max_tokens: 6000,
    messages: [{role:"user", content:$p}]
  }')
  resp=$(curl -sS -X POST "$endpoint/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$body")
  echo "$resp" | jq -r '.choices[0].message.content // .error // empty' >> "$outfile"
  echo >> "$outfile"
done
echo "Done. Output in $outfile"
