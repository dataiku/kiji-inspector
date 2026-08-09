#!/usr/bin/env bash
# Pipeline steps 1-4 for Qwen/Qwen3.6-35B-A3B.
#
# Model geometry is set explicitly:
#   40 layers, hidden_size 2048 (nested under config.text_config -- this is a
#   multimodal MoE, Qwen3_5MoeForConditionalGeneration) -> d-sae 8192 (4x).
#   Layers 12/20/28/36 sit at 30/50/70/90% depth and are all `linear_attention`
#   in the layer_types pattern (full_attention lands on 3,7,11,...,39), so no
#   layer-type confound across the sweep.
#
# Subject is a hub model, not a local checkpoint, so only the HF cache is
# mounted; no read-only model bind mount is required.
#
# NOTE ON THE LABELER: step 4 labels with Qwen3.6-35B-A3B, which is also the
# subject here. That is deliberate -- it keeps the labeler and judge identical
# to the GA run so the subject model is the only variable. The labeler sees
# only token-highlighted text excerpts, never activations, and step 5's judge
# (gemma-4-31B-it, different family) is independent of both, so the validation
# chain is not self-grading. Set JUDGING_MODEL to override.
#
# Step 5 is deliberately not run here -- its per-token cache needs per-layer
# isolation (samples/run_step5_per_layer_qwen36.sh). Step 6 is a separate
# module (samples/run_ablation_qwen36.sh).
#
# Usage: bash samples/run_pipeline_qwen36.sh
#        STEPS="1 2" bash samples/run_pipeline_qwen36.sh   # subset

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SUBJECT_MODEL="${SUBJECT_MODEL:-Qwen/Qwen3.6-35B-A3B}"
JUDGING_MODEL="${JUDGING_MODEL:-Qwen/Qwen3.6-35B-A3B}"
OUTPUT_DIR="${OUTPUT_DIR:-output_qwen3.6-35b-a3b}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
LAYERS="${LAYERS:-12 20 28 36}"
D_SAE="${D_SAE:-8192}"
TARGET_L0="${TARGET_L0:-75}"
STEPS="${STEPS:-1 2 3 4}"
IMAGE="575lab/kiji-inspector:dev"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONTAINER_NAME="${CONTAINER_NAME:-kiji-pipeline-qwen36}"

# Disk guard. Step 1 writes 3,149,046 vectors x 2048 dims x 4 bytes ~= 26 GB
# per layer, ~103 GB for four layers.
MIN_FREE_GB="${MIN_FREE_GB:-140}"
ABORT_FREE_GB="${ABORT_FREE_GB:-25}"

free_gb() { df -BG --output=avail "$REPO_DIR" | tail -1 | tr -dc '0-9'; }

[[ -d "$PAIRS_DIR" ]] || { echo "error: pairs dir not found: $PAIRS_DIR" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/pipeline_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || date +%Y%m%d%H%M%S)"
LOG_FILE="$LOG_DIR/pipeline_steps_$STAMP.log"
DISK_LOG="$LOG_DIR/disk_$STAMP.log"

echo "=== Qwen3.6-35B-A3B pipeline, steps: $STEPS ==="
echo "  subject model:  $SUBJECT_MODEL"
echo "  labeling model: $JUDGING_MODEL   (step 4; == subject, see header)"
echo "  output-dir:     $OUTPUT_DIR"
echo "  pairs-dir:      $PAIRS_DIR"
echo "  layers:         $LAYERS  (of 40; 30/50/70/90% depth)"
echo "  d-sae:          $D_SAE   (4 x hidden_size 2048)"
echo "  target-l0:      $TARGET_L0"
echo "  free disk:      $(free_gb)G  (need >= ${MIN_FREE_GB}G for step 1)"
echo "  log file:       $LOG_FILE"
echo "==============================================="

if [[ " $STEPS " == *" 1 "* ]] && (( $(free_gb) < MIN_FREE_GB )); then
    echo "error: only $(free_gb)G free, step 1 needs >= ${MIN_FREE_GB}G" >&2
    exit 1
fi

watchdog() {
    while true; do
        avail=$(free_gb)
        echo "$(date -u +%H:%M:%S) free=${avail}G $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)" >> "$DISK_LOG"
        if (( avail < ABORT_FREE_GB )); then
            echo "$(date -u +%H:%M:%S) ABORT: free=${avail}G < ${ABORT_FREE_GB}G, killing $CONTAINER_NAME" >> "$DISK_LOG"
            docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
            return
        fi
        sleep 60
    done
}
watchdog &
WATCHDOG_PID=$!
cleanup() {
    kill "$WATCHDOG_PID" 2>/dev/null || true
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# NOTE: do not add -u "$(id -u):$(id -g)" -- torch._inductor's getpass.getuser()
# crashes with KeyError inside this image for a host UID with no /etc/passwd
# entry. Run as root; chown the output afterward.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --gpus all \
  -v "$REPO_DIR:/workspace" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/src \
  -w /workspace \
  --name "$CONTAINER_NAME" \
  "$IMAGE" sleep infinity

{
for step in $STEPS; do
    echo
    echo "=================== STEP $step  ($(date -u +%H:%M:%SZ), free $(free_gb)G) ==================="
    docker exec "$CONTAINER_NAME" \
      python -m kiji_inspector.pipeline \
        --step "$step" \
        --output-dir "$OUTPUT_DIR" \
        --subject-model "$SUBJECT_MODEL" \
        --judging-model "$JUDGING_MODEL" \
        --pairs-dir "$PAIRS_DIR" \
        --layers $LAYERS \
        --d-sae "$D_SAE" \
        --target-l0 "$TARGET_L0" \
        --generation-tp-size 1 \
        --no-thinking
    echo "--- step $step done ($(date -u +%H:%M:%SZ), free $(free_gb)G)"
done
} 2>&1 | tee "$LOG_FILE"

echo
echo "=== Fixing ownership (docker ran as root)... ==="
sudo chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR" 2>/dev/null || chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR"
echo "=== Steps [$STEPS] complete. Log: $LOG_FILE ==="
echo "=== Next: bash samples/run_ablation_qwen36.sh, then run_step5_per_layer_qwen36.sh ==="
