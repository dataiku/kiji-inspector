#!/usr/bin/env bash
# Sweep every scenario of the full pairs dataset, one model load per scenario.
#
# Each scenario needs its own system prompt and tool list, so they cannot share
# a single vLLM session. Runs are resumable: sweep.jsonl is keyed by request
# text, so re-running this script skips whatever already landed.
set -uo pipefail

MODEL=${MODEL:-/models/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-no-mtp}
ROOT=${ROOT:-demo/steering/sweep/output/sweep_candidates}
BATCH=${BATCH:-8000}
# tool_selection first: it is the scenario with genuine tool decisions.
SCENARIOS=${SCENARIOS:-"tool_selection manufacturing supply_chain investment customer_support"}

for scenario in $SCENARIOS; do
  echo "=================== $scenario  $(date -u +%H:%M:%SZ) ==================="
  docker run --rm --gpus all \
    -v "$PWD:/workspace" \
    -v "${MODELS_DIR:-$HOME/models}:/models:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/workspace/src \
    -w /workspace \
    575lab/kiji-inspector:dev \
    python demo/steering/sweep/sweep_pairs_batched.py \
      --model-name "$MODEL" \
      --scenario "$scenario" \
      --candidates "$ROOT/$scenario/meta.json" \
      --results "/workspace/$ROOT/$scenario/sweep.jsonl" \
      --batch-size "$BATCH" 2>&1 \
    | grep -vE "^\(EngineCore|^INFO|^WARNING|^=====|^ *$|CUDA Version|NVIDIA|container|license|Copyright|^By pulling|^https|^A copy"
  echo "--- $scenario exit: ${PIPESTATUS[0]} ---"
done
echo "ALL SCENARIOS COMPLETE $(date -u +%H:%M:%SZ)"
