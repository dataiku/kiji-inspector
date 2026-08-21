#!/usr/bin/env bash
# One HF-backend session for the spec sheet:
#   1. the tool_selection causal battery at the four untouched depths
#      (layers 6/13/20/34, shipped SAEs; 27 and 43 already exist), and
#   2. the same battery at layer 43 with dictionaries the capture never used
#      (tool_selection_only / joint / joint_seed123), families re-derived from
#      the captured vLLM residuals via --active-from-sae.
#
# Runs each step in the Docker image with the fused-Mamba kernels pinned
# (without kernels==0.15.2 the HF path drifts from vLLM). GPU-exclusive:
# do not run while a vLLM engine is resident.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp}
ACT=demo/tool_selection/output/capture/activations.npz

run_in_docker() {
  docker run --rm --gpus all \
    -v "$PWD":/workspace \
    -v /home/shadeform/models:/models:ro \
    -v /ephemeral/cache/huggingface:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/workspace/src:/workspace/demo/tool_selection \
    -w /workspace 575lab/kiji-inspector:dev \
    bash -c "pip install -q 'kernels==0.15.2' && $1"
}

for L in 6 13 20 34; do
  if [ -f "demo/tool_selection/output/steering_layer${L}/steering_results.json" ]; then
    echo "== depth layer ${L}: exists, skipping"
    continue
  fi
  echo "== depth layer ${L}"
  run_in_docker "python demo/tool_selection/attribute_pairs.py \
    --model-name $MODEL --layer $L --activations $ACT"
done

for D in tool_selection_only joint joint_seed123; do
  OUT="demo/spec_sheet/output/robustness/${D}_layer43"
  if [ -f "$OUT/steering_results.json" ]; then
    echo "== robustness ${D}: exists, skipping"
    continue
  fi
  echo "== robustness ${D} @ layer 43"
  run_in_docker "python demo/tool_selection/attribute_pairs.py \
    --model-name $MODEL --layer 43 \
    --sae-local-dir demo/spec_sheet/output/saes/${D} \
    --activations $ACT --active-from-sae \
    --results-dir $OUT"
done

echo "HF session complete."
