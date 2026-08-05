#!/usr/bin/env bash
# Step 6 (causal ablation) for Qwen/Qwen3.6-35B-A3B, layers 12/20/28/36.
#
# Ablation is not reachable via `pipeline.py --step 6` -- argparse accepts the
# value but nothing dispatches it -- so it runs as its own module. It also
# must use the HuggingFace backend: vLLM cannot intervene mid-forward.
#
# Unlike the Nemotron runs this needs no mamba_ssm/causal-conv1d build; Qwen3.5
# MoE uses linear attention rather than Mamba2. If HF asks for a kernel package
# (e.g. flash-linear-attention) the first layer will say so -- install it once
# in this long-lived container rather than per layer.
#
# Usage: bash samples/run_ablation_qwen36.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SUBJECT_MODEL="${SUBJECT_MODEL:-Qwen/Qwen3.6-35B-A3B}"
OUTPUT_DIR="${OUTPUT_DIR:-output_qwen3.6-35b-a3b}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
LAYERS="${LAYERS:-12 20 28 36}"
IMAGE="575lab/kiji-inspector:dev"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONTAINER_NAME="${CONTAINER_NAME:-kiji-ablation-qwen36}"

for l in $LAYERS; do
    ck="$REPO_DIR/$OUTPUT_DIR/layer_$l/sae_checkpoints/sae_final.pt"
    cf="$REPO_DIR/$OUTPUT_DIR/layer_$l/activations/contrastive_features.json"
    [[ -f "$ck" ]] || { echo "error: missing SAE checkpoint $ck" >&2; exit 1; }
    [[ -f "$cf" ]] || { echo "error: missing $cf" >&2; exit 1; }
done

LOG_DIR="$OUTPUT_DIR/ablation_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ablation_$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || date +%Y%m%d%H%M%S).log"

echo "=== Ablation: Qwen3.6-35B-A3B, layers $LAYERS ==="
echo "  output-dir: $OUTPUT_DIR"
echo "  log file:   $LOG_FILE"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --gpus all \
  -v "$REPO_DIR:/workspace" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src \
  -w /workspace \
  --name "$CONTAINER_NAME" \
  "$IMAGE" sleep infinity
trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT

{
for layer in $LAYERS; do
    echo "=============== ABLATION - layer $layer ($(date -u +%H:%M:%SZ)) ==============="
    rm -rf "$OUTPUT_DIR/layer_$layer/ablation"
    docker exec "$CONTAINER_NAME" python -m kiji_inspector.experiments.ablation \
        --sae-checkpoint "$OUTPUT_DIR/layer_$layer/sae_checkpoints/sae_final.pt" \
        --contrastive-features "$OUTPUT_DIR/layer_$layer/activations/contrastive_features.json" \
        --pairs-dir "$PAIRS_DIR" \
        --output-dir "$OUTPUT_DIR/layer_$layer/ablation" \
        --model "$SUBJECT_MODEL" \
        --layer "$layer"
    echo "--- layer $layer ablation done"
done
} 2>&1 | tee "$LOG_FILE"

echo "--- Fixing ownership (docker ran as root)... ---"
sudo chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR" 2>/dev/null || chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "=== Ablation complete for layers: $LAYERS. Log: $LOG_FILE ==="
