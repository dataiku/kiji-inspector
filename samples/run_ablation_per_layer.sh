#!/usr/bin/env bash
# Run the causal ablation experiment, one layer per container.
#
# Ablation zeroes the top contrastive SAE features during the forward pass and
# measures whether the model's tool choice flips, against a random-feature
# control. Unlike steps 4/5 it never loads the judge — it runs the subject
# model through HF forward pre-hooks — so it is comparatively cheap.
#
# Usage:
#   bash samples/run_ablation_per_layer.sh
#
# Environment overrides:
#   LAYERS       Layers to ablate           (default: 12 18 24 30 36)
#   MODEL        Subject model              (default: google/gemma-4-E4B-it)
#   N_FEATURES   Top features to ablate     (default: pipeline default of 10)
#   N_PROMPTS    Max prompts per contrast   (default: pipeline default of 100)
#   SEED         Random seed                (default: 42)
#   IMAGE        Docker image
#
# Notes:
#   - ablation.py's own defaults are wrong for this project: --model defaults to
#     nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 and --layer to 20, which is not
#     one of the trained layers. Both are always passed explicitly here.
#   - --layer N indexes model_layers[N] and hooks its *input*, matching the
#     convention the SAEs were trained under (the residual stream entering
#     block N). Do not offset these ids.
set -euo pipefail

IMAGE="${IMAGE:-575lab/kiji-inspector:dev}"
MODEL="${MODEL:-google/gemma-4-E4B-it}"
LAYERS="${LAYERS:-12 18 24 30 36}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# --- Pre-flight: every layer needs a trained SAE and step 3 features ---------
for l in $LAYERS; do
    ck="$REPO_DIR/$OUTPUT_DIR/layer_$l/sae_checkpoints/sae_final.pt"
    cf="$REPO_DIR/$OUTPUT_DIR/layer_$l/activations/contrastive_features.json"
    [[ -f "$ck" ]] || { echo "error: missing SAE checkpoint $ck" >&2; exit 1; }
    [[ -f "$cf" ]] || { echo "error: missing $cf — run step 3 first" >&2; exit 1; }
done

if ! compgen -G "$REPO_DIR/$PAIRS_DIR/*.parquet" > /dev/null; then
    echo "error: no parquet shards in $REPO_DIR/$PAIRS_DIR" >&2
    exit 1
fi

echo "image:         $IMAGE"
echo "subject model: $MODEL  (HF forward pre-hooks, no judge)"
echo "layers:        $LAYERS  (one container per layer)"
echo "seed:          $SEED"
echo

for layer in $LAYERS; do
    echo "=============== ABLATION — layer $layer ==============="

    ARGS=(
        --sae-checkpoint "$OUTPUT_DIR/layer_$layer/sae_checkpoints/sae_final.pt"
        --contrastive-features "$OUTPUT_DIR/layer_$layer/activations/contrastive_features.json"
        --pairs-dir "$PAIRS_DIR"
        --output-dir "$OUTPUT_DIR/layer_$layer/ablation"
        --model "$MODEL"
        --layer "$layer"
        --seed "$SEED"
    )
    [[ -n "${N_FEATURES:-}" ]] && ARGS+=(--n-features "$N_FEATURES")
    [[ -n "${N_PROMPTS:-}" ]] && ARGS+=(--n-prompts-per-type "$N_PROMPTS")

    docker run --rm --gpus all \
        --ipc=host \
        -v "$REPO_DIR:/workspace" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        -e PYTHONPATH=/workspace/src \
        -w /workspace \
        "$IMAGE" \
        python -m kiji_inspector.experiments.ablation "${ARGS[@]}"

    echo "--- layer $layer ablation done"
done

echo
echo "ablation complete for layers: $LAYERS"
