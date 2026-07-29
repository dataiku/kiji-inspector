#!/usr/bin/env bash
# Re-run pipeline steps 2-5 for the layers whose SAEs collapsed to a dense
# code, this time with an adaptive L1 targeting a fixed L0.
#
# Context: the first training run used a fixed l1_coefficient=5e-3 with no
# --target-l0. Layers 12/24/30/36 settled at L0 ~250-310 with only ~260-340
# alive features, i.e. the alive features fire on 86-99% of inputs — a rotated
# dense basis, not a sparse dictionary. Layer 18 landed at L0 73 (5.6% firing
# rate) on its own and is EXCLUDED here so its good result is not destroyed.
#
# Step 1 is NOT re-run: the extracted activations in
# output/layer_N/activations/ are fine and are reused as-is.
#
# Usage:
#   bash samples/rerun_failed_layers_target_l0.sh
#
# Environment overrides:
#   LAYERS        Layers to retrain          (default: 12 24 30 36)
#   TARGET_L0     Adaptive-L1 target L0      (default: 75)
#   STEPS         Pipeline steps to run      (default: 2 3 4 5)
#   IMAGE         Docker image               (default: 575lab/kiji-inspector:dev)
#   JUDGING_MODEL Judge for steps 4/5        (default: Qwen/Qwen3.6-35B-A3B)
#   OUTPUT_DIR    Repo-relative output dir   (default: output)
#   PAIRS_DIR     Pre-generated pairs        (default: output/pairs)
#   HF_CACHE      Host HF cache              (default: ~/.cache/huggingface)
#   SKIP_BACKUP   Set to 1 to skip archiving the previous run's metrics
set -euo pipefail

IMAGE="${IMAGE:-575lab/kiji-inspector:dev}"
JUDGING_MODEL="${JUDGING_MODEL:-Qwen/Qwen3.6-35B-A3B}"
LAYERS="${LAYERS:-12 24 30 36}"
TARGET_L0="${TARGET_L0:-75}"
STEPS="${STEPS:-2 3 4 5}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# --- Guards -----------------------------------------------------------------

# Layer 18 is the one healthy SAE. Retraining it would overwrite a good result.
for l in $LAYERS; do
    if [[ "$l" == "18" ]]; then
        echo "error: layer 18 is in LAYERS. Its SAE trained correctly (L0 73.45," >&2
        echo "       1304 alive, 5.6% firing rate); re-running would overwrite it." >&2
        echo "       Remove it, or set LAYERS explicitly if this is intentional." >&2
        exit 1
    fi
done

# Step 1 is not run here, so the activation shards must already exist.
for l in $LAYERS; do
    act="$REPO_DIR/$OUTPUT_DIR/layer_$l/activations"
    if ! compgen -G "$act/shard_*.npy" > /dev/null; then
        echo "error: no activation shards in $act" >&2
        echo "       this script skips step 1 — run extraction first" >&2
        exit 1
    fi
done

if ! compgen -G "$REPO_DIR/$PAIRS_DIR/*.parquet" > /dev/null; then
    echo "error: no parquet shards in $REPO_DIR/$PAIRS_DIR" >&2
    exit 1
fi

# --- Back up the previous run's metrics and reports --------------------------
# Steps 2-5 overwrite feature_health.json / metrics.jsonl and the step 3-5
# reports. Archive the small files so the failed run stays comparable; the
# ~1 GB of .pt weights per layer is skipped since step 2 regenerates it.
if [[ "${SKIP_BACKUP:-0}" != "1" ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    BACKUP="$REPO_DIR/$OUTPUT_DIR/backup_pre_target_l0_$STAMP"
    for l in $LAYERS; do
        mkdir -p "$BACKUP/layer_$l"
        for f in config.json feature_health.json metrics.jsonl; do
            src="$REPO_DIR/$OUTPUT_DIR/layer_$l/sae_checkpoints/$f"
            [[ -f "$src" ]] && cp "$src" "$BACKUP/layer_$l/"
        done
        for f in contrastive_features.json decision_report.json \
                 feature_descriptions.json fuzzing_results.json \
                 fuzzing_summary.json; do
            src="$REPO_DIR/$OUTPUT_DIR/layer_$l/activations/$f"
            [[ -f "$src" ]] && cp "$src" "$BACKUP/layer_$l/"
        done
    done
    echo "backed up previous metrics to ${BACKUP#"$REPO_DIR/"}"
fi

# --- Run ---------------------------------------------------------------------

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1

echo "image:         $IMAGE"
echo "judging model: $JUDGING_MODEL"
echo "layers:        $LAYERS  (18 excluded — already healthy)"
echo "target L0:     $TARGET_L0 (adaptive l1_coefficient)"
echo "steps:         $STEPS  (step 1 skipped, activations reused)"
echo "gpus:          $NUM_GPUS"
echo

# --step takes a single value, so loop inside one container. `set -e` in the
# inner shell aborts the chain if any step fails, rather than feeding a broken
# checkpoint into the next stage.
INNER="set -e
for step in $STEPS; do
  echo
  echo '=============== STEP '\$step' ==============='
  python -m kiji_inspector.pipeline \
    --step \$step \
    --output-dir '$OUTPUT_DIR' \
    --pairs-dir '$PAIRS_DIR' \
    --layers $LAYERS \
    --target-l0 '$TARGET_L0' \
    --judging-model '$JUDGING_MODEL' \
    --backend vllm \
    --extraction-tp-size $NUM_GPUS \
    --generation-tp-size $NUM_GPUS
done"

exec docker run --rm --gpus all \
    --ipc=host \
    -v "$REPO_DIR:/workspace" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PYTHONPATH=/workspace/src \
    -w /workspace \
    "$IMAGE" \
    bash -c "$INNER"
