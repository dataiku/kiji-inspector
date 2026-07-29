#!/usr/bin/env bash
# Launch the Kiji Inspector pipeline inside the vLLM hidden-states Docker image
# with google/gemma-4-E4B-it as the subject model.
#
# Uses the ALREADY-GENERATED contrastive pairs in output/pairs — this script
# never calls generate_pairs, it only consumes the existing parquet shards.
#
# Weights are pre-fetched on the HOST with `uvx --from huggingface_hub hf
# download` into HF_CACHE, which the container then mounts read-through. This
# keeps the (slow, resumable) download out of the container lifecycle.
#
# Usage:
#   samples/run_pipeline_gemma4_e4b.sh [extra pipeline args...]
#
# Examples:
#   samples/run_pipeline_gemma4_e4b.sh                     # full run (steps 1-5)
#   samples/run_pipeline_gemma4_e4b.sh --step 1            # extraction only
#   LAYERS="23" samples/run_pipeline_gemma4_e4b.sh         # single layer
#
# Environment overrides:
#   IMAGE           Docker image                (default: 575lab/kiji-inspector:dev)
#   SUBJECT_MODEL   Model under study           (default: google/gemma-4-E4B-it)
#   JUDGING_MODEL   Labeling/fuzzing judge      (default: Qwen/Qwen3.6-35B-A3B)
#   LAYERS          Space-separated layer list  (default: 12 18 24 30 36,
#                   i.e. the output of blocks 11/17/23/29/35 — see below)
#   OUTPUT_DIR      Repo-relative output dir    (default: output)
#   PAIRS_DIR       Pre-generated pairs dir     (default: output/pairs)
#   HF_CACHE        Host HF cache               (default: ~/.cache/huggingface)
#   HF_TOKEN        HuggingFace token (optional — neither model is gated as of
#                   2026-07-28; set it if that changes or for rate limits)
#   SKIP_DOWNLOAD   Set to 1 to skip the uvx pre-fetch step
#
# Notes:
#   - Requires the NVIDIA Container Toolkit (`--gpus all`) and `uvx` on PATH.
#   - No `--no-thinking`: Gemma 4's chat template defaults enable_thinking to
#     false, so reasoning is opt-in and the prefill already lands at the
#     answer position. The flag would render an identical prompt. The judge
#     gets enable_thinking=False automatically via
#     extraction.vllm_activation_extractor.recommended_chat_template_kwargs.
#   - Steps 4/5 load JUDGING_MODEL with vLLM, sequentially after extraction —
#     the subject and judge models are not resident at the same time.
set -euo pipefail

IMAGE="${IMAGE:-575lab/kiji-inspector:dev}"
SUBJECT_MODEL="${SUBJECT_MODEL:-google/gemma-4-E4B-it}"
JUDGING_MODEL="${JUDGING_MODEL:-Qwen/Qwen3.6-35B-A3B}"
# Spread across the 42 layers of gemma-4-E4B's text tower (~26%-83% depth).
#
# The connector records layer id N *after* block N-1 returns (gemma4.py:1346
# emits index layer_idx+1; index 0 is the embedding output). So these ids are
# the residual stream LEAVING blocks 11/17/23/29/35 — attention and MLP both
# written in. Valid ids are 0-42.
LAYERS="${LAYERS:-12 18 24 30 36}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE" "$REPO_DIR/$OUTPUT_DIR"

# Pre-flight: the pairs must already exist. Fail loudly instead of letting the
# container spin up a model and die in step 1.
if ! compgen -G "$REPO_DIR/$PAIRS_DIR/*.parquet" > /dev/null; then
    echo "error: no parquet shards in $REPO_DIR/$PAIRS_DIR" >&2
    echo "       generate them first, e.g. inside the same image:" >&2
    echo "       python -m kiji_inspector.generate_pairs 1300 --output-dir $PAIRS_DIR" >&2
    exit 1
fi
NUM_SHARDS="$(compgen -G "$REPO_DIR/$PAIRS_DIR/*.parquet" | wc -l)"

# Pre-fetch weights on the host with uvx. `hf download` is resumable and a
# no-op once the snapshot is complete, so reruns cost a single revision check.
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    if ! command -v uvx > /dev/null; then
        echo "error: uvx not found on PATH (see https://docs.astral.sh/uv/)" >&2
        echo "       set SKIP_DOWNLOAD=1 to let the container fetch weights instead" >&2
        exit 1
    fi
    for repo in "$SUBJECT_MODEL" "$JUDGING_MODEL"; do
        # Local directories are passed through to the container as-is.
        [[ "$repo" == /* || "$repo" == .* || "$repo" == ~* ]] && continue
        echo "==> downloading $repo into $HF_CACHE"
        HF_HOME="$HF_CACHE" uvx --from huggingface_hub hf download "$repo"
    done
    echo
fi

# Default parallelism to the number of visible GPUs.
NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
EXTRACTION_TP_SIZE="${EXTRACTION_TP_SIZE:-$NUM_GPUS}"
GENERATION_TP_SIZE="${GENERATION_TP_SIZE:-$NUM_GPUS}"

PIPELINE_ARGS=(
    --subject-model "$SUBJECT_MODEL"
    --output-dir "$OUTPUT_DIR"
    --pairs-dir "$PAIRS_DIR"
    --judging-model "$JUDGING_MODEL"
    --backend vllm
    --extraction-tp-size "$EXTRACTION_TP_SIZE"
    --generation-tp-size "$GENERATION_TP_SIZE"
)
# shellcheck disable=SC2206  # LAYERS is intentionally word-split
[[ -n "${LAYERS:-}" ]] && PIPELINE_ARGS+=(--layers ${LAYERS})

echo "image:         $IMAGE"
echo "subject model: $SUBJECT_MODEL"
echo "judging model: $JUDGING_MODEL"
echo "layers:        $LAYERS"
echo "pairs:         $PAIRS_DIR ($NUM_SHARDS parquet shards, reused as-is)"
echo "gpus:          $NUM_GPUS"
echo "output:        $REPO_DIR/$OUTPUT_DIR"
echo

exec docker run --rm --gpus all \
    --ipc=host \
    -v "$REPO_DIR:/workspace" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PYTHONPATH=/workspace/src \
    -w /workspace \
    "$IMAGE" \
    python -m kiji_inspector.pipeline "${PIPELINE_ARGS[@]}" "$@"
