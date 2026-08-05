#!/usr/bin/env bash
# Run step 5 (fuzzing evaluation) one layer at a time, each in its own
# ephemeral container, for Qwen3.6-35B-A3B.
#
# Step 5a (per-token activation extraction) writes one full-sequence .npz
# per prompt to a temp dir INSIDE the container (not the bind mount) --
# unconditionally, for every top/bottom example across every labeled
# feature, regardless of --fuzz-examples-per-feature (which only limits a
# later, cheap sub-step). Running all 6 layers in one process accumulates
# this unbounded (crashed the host at ~309GB partway through before). Doing
# one layer per --rm container means each layer's cache is fully reclaimed
# on exit, so nothing accumulates across layers.
#
# Also runs a disk-space watchdog: if free space drops below a hard floor
# mid-layer, kill the container immediately rather than let the host fill.
#
# Usage: bash samples/run_step5_per_layer_qwen36.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SUBJECT_MODEL="${SUBJECT_MODEL:-Qwen/Qwen3.6-35B-A3B}"
# gemma-4 judge: a different model family from the Qwen labeler (step 4), so
# fuzzing scores measure independent agreement rather than the labeler
# grading its own labels. Overridable: JUDGING_MODEL=... bash this_script.
JUDGING_MODEL="${JUDGING_MODEL:-google/gemma-4-31B-it}"
OUTPUT_DIR="${OUTPUT_DIR:-output_qwen3.6-35b-a3b}"
PAIRS_DIR="output/pairs"
LAYERS="${LAYERS:-12 20 28 36}"
IMAGE="575lab/kiji-inspector:dev"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONTAINER_NAME="${CONTAINER_NAME:-kiji-step5-qwen36}"
MIN_FREE_GB=100          # refuse to start a layer below this
ABORT_FREE_GB=15         # kill mid-layer if free space drops this low

LOG_DIR="$OUTPUT_DIR/step5_logs"
mkdir -p "$LOG_DIR"

free_gb() {
    df --output=avail -BG / | tail -1 | tr -dc '0-9'
}

watchdog() {
    while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; do
        avail="$(free_gb)"
        if [[ "$avail" -lt "$ABORT_FREE_GB" ]]; then
            echo "!!! WATCHDOG: free disk ${avail}G < ${ABORT_FREE_GB}G floor -- killing $CONTAINER_NAME"
            docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
            break
        fi
        sleep 20
    done
}

for layer in $LAYERS; do
    avail="$(free_gb)"
    echo "=============== STEP 5 — layer $layer (${avail}G free) ==============="
    if [[ "$avail" -lt "$MIN_FREE_GB" ]]; then
        echo "error: only ${avail}G free, below ${MIN_FREE_GB}G minimum -- stopping before layer $layer" >&2
        exit 1
    fi

    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

    watchdog &
    watchdog_pid=$!

    set +e
    docker run --rm --gpus all \
      -v "$REPO_DIR:/workspace" \
      -v "$HF_CACHE:/root/.cache/huggingface" \
      -e HF_HOME=/root/.cache/huggingface \
      -e PYTHONPATH=/workspace/src \
      -w /workspace \
      --name "$CONTAINER_NAME" \
      "$IMAGE" \
      python -m kiji_inspector.pipeline \
        --step 5 \
        --output-dir "$OUTPUT_DIR" \
        --subject-model "$SUBJECT_MODEL" \
        --judging-model "$JUDGING_MODEL" \
        --pairs-dir "$PAIRS_DIR" \
        --layers "$layer" \
        --generation-tp-size 1 \
        --no-thinking \
      2>&1 | tee "$LOG_DIR/step5_layer_${layer}.log"
    status=${PIPESTATUS[0]}
    set -e

    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    if [[ "$status" -ne 0 ]]; then
        echo "error: step 5 failed for layer $layer (exit $status) -- stopping" >&2
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        sudo chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR" 2>/dev/null || chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR"
        exit 1
    fi

    echo "--- layer $layer step 5 done ($(free_gb)G free) ---"
done

echo "--- Fixing ownership (docker ran as root)... ---"
sudo chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR" 2>/dev/null || chown -R "$(whoami):$(whoami)" "$OUTPUT_DIR"
echo "=== Step 5 complete for layers: $LAYERS ==="
