#!/usr/bin/env bash
# Full pipeline (steps 1-5) for google/gemma-4-E4B-it, from scratch.
#
# Runs steps 1-4 across all layers in one container, then step 5 ONE LAYER AT A
# TIME. That split is not cosmetic: step 5 caches per-token activations sized
# layers x features x examples-per-feature, and a single five-layer pass reached
# 377 GB and filled a 713 GB disk. Per layer it is ~11-25 GB.
#
# Usage:
#   bash samples/run_full_pipeline_gemma4.sh
#
# Environment overrides:
#   LAYERS         Layers to process        (default: 12 18 24 30 36)
#   TARGET_L0      Adaptive-L1 target       (default: 75; empty to disable)
#   SUBJECT_MODEL  Model under study        (default: google/gemma-4-E4B-it)
#   JUDGING_MODEL  Judge for steps 4/5      (default: Qwen/Qwen3.6-35B-A3B)
#   STEPS_1_4      Steps to run in bulk     (default: 1 2 3 4)
#   SKIP_STEP5     Set to 1 to stop after step 4
#   NEED_GB        Free GB required upfront (default: 220)
#   LOG            Log file                 (default: logs/pipeline_<stamp>.log)
#
# Notes:
#   - Reasoning suppression is auto-detected from the subject model's chat
#     template, so --no-thinking is not passed; gemma-4 needs none anyway.
#   - The assistant prefill has no trailing space, so the decision token is the
#     tool name. Verified 40/40 prompts on gemma-4, Nemotron and Qwen3.6.
set -euo pipefail

IMAGE="${IMAGE:-575lab/kiji-inspector:dev}"
SUBJECT_MODEL="${SUBJECT_MODEL:-google/gemma-4-E4B-it}"
JUDGING_MODEL="${JUDGING_MODEL:-Qwen/Qwen3.6-35B-A3B}"
LAYERS="${LAYERS:-12 18 24 30 36}"
TARGET_L0="${TARGET_L0-75}"
STEPS_1_4="${STEPS_1_4:-1 2 3 4}"
NEED_GB="${NEED_GB:-220}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

mkdir -p "$REPO_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-$REPO_DIR/logs/pipeline_$STAMP.log}"

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
free_gb() { df -BG --output=avail "$REPO_DIR" | tail -1 | tr -dc '0-9'; }

# --- Pre-flight --------------------------------------------------------------
if ! compgen -G "$REPO_DIR/$PAIRS_DIR/*.parquet" > /dev/null; then
    echo "error: no parquet shards in $REPO_DIR/$PAIRS_DIR" >&2
    echo "       generate them first: python -m kiji_inspector.generate_pairs <N>" >&2
    exit 1
fi
avail="$(free_gb)"
if (( avail < NEED_GB )); then
    echo "error: ${avail} GB free, need ${NEED_GB} GB (step 1 writes ~32 GB per layer)" >&2
    exit 1
fi

{
    echo "================ FULL PIPELINE $STAMP ================"
    echo "image:         $IMAGE"
    echo "subject model: $SUBJECT_MODEL"
    echo "judging model: $JUDGING_MODEL"
    echo "layers:        $LAYERS"
    echo "target L0:     ${TARGET_L0:-<unset>}"
    echo "steps:         $STEPS_1_4 (bulk), then 5 per layer"
    echo "pairs:         $PAIRS_DIR ($(compgen -G "$REPO_DIR/$PAIRS_DIR/*.parquet" | wc -l) shards)"
    echo "gpus:          $NUM_GPUS"
    echo "free space:    ${avail} GB"
    echo "log:           $LOG"
    echo
} | tee -a "$LOG"

PIPE_ARGS="--output-dir '$OUTPUT_DIR' --pairs-dir '$PAIRS_DIR' --layers $LAYERS \
 --subject-model '$SUBJECT_MODEL' --judging-model '$JUDGING_MODEL' --backend vllm \
 --extraction-tp-size $NUM_GPUS --generation-tp-size $NUM_GPUS"
[[ -n "$TARGET_L0" ]] && PIPE_ARGS="$PIPE_ARGS --target-l0 '$TARGET_L0'"

# --- Steps 1-4 (all layers, one container) -----------------------------------
INNER="set -e
for step in $STEPS_1_4; do
  echo
  echo \"=============== STEP \$step  (\$(date -u +%H:%M:%S)) ===============\"
  python -m kiji_inspector.pipeline --step \$step $PIPE_ARGS
done"

docker run --rm --gpus all \
    --ipc=host \
    -v "$REPO_DIR:/workspace" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/workspace/src \
    -w /workspace \
    "$IMAGE" \
    bash -c "$INNER" 2>&1 | tee -a "$LOG"

echo "--- steps $STEPS_1_4 done, $(free_gb) GB free" | tee -a "$LOG"

# --- Step 5 (one layer per container) ----------------------------------------
if [[ "${SKIP_STEP5:-0}" == "1" ]]; then
    echo "SKIP_STEP5=1 — stopping after step 4" | tee -a "$LOG"
else
    LAYERS="$LAYERS" SUBJECT_MODEL="$SUBJECT_MODEL" JUDGING_MODEL="$JUDGING_MODEL" \
        IMAGE="$IMAGE" OUTPUT_DIR="$OUTPUT_DIR" PAIRS_DIR="$PAIRS_DIR" \
        bash "$REPO_DIR/samples/run_step5_per_layer.sh" 2>&1 | tee -a "$LOG"
fi

{
    echo
    echo "================ PIPELINE COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ================"
    echo "free space: $(free_gb) GB"
    for l in $LAYERS; do
        fh="$REPO_DIR/$OUTPUT_DIR/layer_$l/sae_checkpoints/feature_health.json"
        fz="$REPO_DIR/$OUTPUT_DIR/layer_$l/activations/fuzzing_summary.json"
        printf 'layer %-3s health=%s fuzzing=%s\n' "$l" \
            "$([[ -f $fh ]] && echo ok || echo MISSING)" \
            "$([[ -f $fz ]] && echo ok || echo MISSING)"
    done
} | tee -a "$LOG"
