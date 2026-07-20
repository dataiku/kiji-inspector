#!/usr/bin/env bash
# Run the hidden-states smoke test inside the built image.
#
# Usage:
#   samples/run_hidden_states_test.sh [IMAGE]
#
# Notes:
#   - Requires the NVIDIA Container Toolkit (`--gpus all`).
#   - Mounts a persistent HF cache so Qwen/Qwen3-8B (~16 GB) is downloaded once.
#   - Qwen3-8B is a public model, so no HF token is needed.
set -euo pipefail

IMAGE="${1:-kiji-inspector}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Persist HF downloads on the host so reruns are fast.
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

docker run --rm --gpus all \
    -v "$REPO_DIR:/workspace:ro" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMAGE" \
    python /workspace/samples/test_hidden_states.py
