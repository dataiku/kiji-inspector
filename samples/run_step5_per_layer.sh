#!/usr/bin/env bash
# Run pipeline step 5 (fuzzing validation) one layer at a time.
#
# Why per-layer: step 5 extracts per-token activations for every fuzzing prompt
# and caches them as .npz files via tempfile.mkdtemp (fuzzing_evaluator.py:187).
# The extraction covers all requested layers in one pass, so the cache scales
# with layers x features x examples-per-feature:
#
#   5 layers x 10 ex/feat  ~= 420 GB   <- filled a 713 GB disk to 100%
#   1 layer  x 10 ex/feat  ~=  84 GB
#
# Running all five layers in one invocation cannot fit. This script does one
# layer per container, points the cache at a host directory so it is visible
# and reclaimable, and refuses to start a layer without room for it.
#
# Usage:
#   bash samples/run_step5_per_layer.sh
#
# Environment overrides:
#   LAYERS         Layers to fuzz            (default: 12 18 24 30 36)
#   EXAMPLES       --fuzz-examples-per-feature (default: pipeline default of 10)
#   NEED_GB        Free GB required per layer  (default: 120)
#   FUZZ_CACHE     Host dir for the npz cache  (default: ./.fuzz_cache)
#   IMAGE          Docker image
#   JUDGING_MODEL  Judge for step 5          (default: Qwen/Qwen3.6-35B-A3B)
#   KEEP_CACHE     Set to 1 to keep the cache after each layer (debugging)
set -euo pipefail

IMAGE="${IMAGE:-575lab/kiji-inspector:dev}"
# MUST be passed: step 5 re-extracts per-token activations with the subject
# model. Omitting it falls back to the pipeline default (Nemotron-3-Nano-30B),
# which yields 2688-dim activations and the wrong tokenizer, and the run dies
# feeding them to a 2560-dim gemma-trained SAE.
SUBJECT_MODEL="${SUBJECT_MODEL:-google/gemma-4-E4B-it}"
JUDGING_MODEL="${JUDGING_MODEL:-Qwen/Qwen3.6-35B-A3B}"
LAYERS="${LAYERS:-12 18 24 30 36}"
NEED_GB="${NEED_GB:-120}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PAIRS_DIR="${PAIRS_DIR:-output/pairs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
FUZZ_CACHE="${FUZZ_CACHE:-$REPO_DIR/.fuzz_cache}"

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1

free_gb() { df -BG --output=avail "$REPO_DIR" | tail -1 | tr -dc '0-9'; }

# The cache is root-owned (written from inside the container), so removal needs
# the same privilege. Prefer docker over sudo so this works either way.
clean_cache() {
    if [[ -d "$FUZZ_CACHE" ]] && [[ -n "$(ls -A "$FUZZ_CACHE" 2>/dev/null)" ]]; then
        docker run --rm -v "$FUZZ_CACHE:/fuzzcache" "$IMAGE" \
            bash -c 'rm -rf /fuzzcache/* /fuzzcache/.[!.]* 2>/dev/null; true' > /dev/null 2>&1 || true
    fi
}

mkdir -p "$FUZZ_CACHE"
trap 'echo; echo "interrupted — cleaning fuzz cache"; [[ "${KEEP_CACHE:-0}" == "1" ]] || clean_cache' INT TERM

echo "image:         $IMAGE"
echo "subject model: $SUBJECT_MODEL"
echo "judging model: $JUDGING_MODEL"
echo "layers:        $LAYERS  (one container per layer)"
echo "fuzz cache:    $FUZZ_CACHE  (cleaned between layers)"
echo "free space:    $(free_gb) GB, requiring ${NEED_GB} GB per layer"
echo

for layer in $LAYERS; do
    clean_cache

    avail="$(free_gb)"
    if (( avail < NEED_GB )); then
        echo "error: only ${avail} GB free, need ${NEED_GB} GB for layer $layer" >&2
        echo "       free space or lower EXAMPLES to shrink the cache" >&2
        exit 1
    fi

    echo "=============== STEP 5 — layer $layer (${avail} GB free) ==============="

    ARGS=(
        --step 5
        --output-dir "$OUTPUT_DIR"
        --pairs-dir "$PAIRS_DIR"
        --layers "$layer"
        --subject-model "$SUBJECT_MODEL"
        --judging-model "$JUDGING_MODEL"
        --backend vllm
        --extraction-tp-size "$NUM_GPUS"
        --generation-tp-size "$NUM_GPUS"
    )
    [[ -n "${EXAMPLES:-}" ]] && ARGS+=(--fuzz-examples-per-feature "$EXAMPLES")

    # TMPDIR redirects mkdtemp into the mounted host dir, keeping the cache off
    # the container's writable layer where it is invisible to df on the repo.
    docker run --rm --gpus all \
        --ipc=host \
        -v "$REPO_DIR:/workspace" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$FUZZ_CACHE:/fuzzcache" \
        -e TMPDIR=/fuzzcache \
        -e HF_HOME=/root/.cache/huggingface \
        -e PYTHONPATH=/workspace/src \
        -w /workspace \
        "$IMAGE" \
        python -m kiji_inspector.pipeline "${ARGS[@]}"

    echo "--- layer $layer done; cache peak $(du -sh "$FUZZ_CACHE" 2>/dev/null | cut -f1)"
    [[ "${KEEP_CACHE:-0}" == "1" ]] || clean_cache
done

clean_cache
echo
echo "step 5 complete for layers: $LAYERS"
