#!/usr/bin/env bash
set -euo pipefail

SOCKET_PATH="${ACTIVATION_PROCESSOR_SOCKET:-/tmp/vllm-activation-processor.sock}"
BASE_MODEL="${ACTIVATION_BASE_MODEL:-${MODEL_NAME:-google/gemma-4-E4B-it}}"
ACTIVATION_LAYERS="${ACTIVATION_LAYERS:-8}"
ACTIVATION_EXPLANATION_TOP_K="${ACTIVATION_EXPLANATION_TOP_K:-5}"

uv run --script activation_processor_sidecar.py \
    --socket "${SOCKET_PATH}" \
    --base-model "${BASE_MODEL}" \
    --layers "${ACTIVATION_LAYERS}" \
    --top-k "${ACTIVATION_EXPLANATION_TOP_K}"
