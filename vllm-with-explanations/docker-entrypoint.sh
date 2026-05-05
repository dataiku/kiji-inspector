#!/usr/bin/env bash
set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export ACTIVATION_PROCESSOR_SOCKET="${ACTIVATION_PROCESSOR_SOCKET:-/tmp/vllm-activation-processor.sock}"

MODEL_NAME="${MODEL_NAME:-google/gemma-4-E4B-it}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ACTIVATION_LAYERS="${ACTIVATION_LAYERS:-8}"
ACTIVATION_EXPLANATION_TOP_K="${ACTIVATION_EXPLANATION_TOP_K:-5}"
ACTIVATION_PROCESSOR_DEVICE="${ACTIVATION_PROCESSOR_DEVICE:-auto}"
ACTIVATION_PROCESSOR_LOG_LEVEL="${ACTIVATION_PROCESSOR_LOG_LEVEL:-info}"

sidecar_pid=""
vllm_pid=""

cleanup() {
    if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>/dev/null; then
        kill "${vllm_pid}" 2>/dev/null || true
    fi
    if [[ -n "${sidecar_pid}" ]] && kill -0 "${sidecar_pid}" 2>/dev/null; then
        kill "${sidecar_pid}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

rm -f "${ACTIVATION_PROCESSOR_SOCKET}"

uv run --script activation_processor_sidecar.py \
    --socket "${ACTIVATION_PROCESSOR_SOCKET}" \
    --log-level "${ACTIVATION_PROCESSOR_LOG_LEVEL}" \
    --base-model "${MODEL_NAME}" \
    --layers "${ACTIVATION_LAYERS}" \
    --top-k "${ACTIVATION_EXPLANATION_TOP_K}" \
    --device "${ACTIVATION_PROCESSOR_DEVICE}" &
sidecar_pid="$!"

for _ in $(seq 1 600); do
    if [[ -S "${ACTIVATION_PROCESSOR_SOCKET}" ]]; then
        break
    fi
    if ! kill -0 "${sidecar_pid}" 2>/dev/null; then
        echo "Activation processor sidecar exited before creating ${ACTIVATION_PROCESSOR_SOCKET}" >&2
        exit 1
    fi
    sleep 1
done

if [[ ! -S "${ACTIVATION_PROCESSOR_SOCKET}" ]]; then
    echo "Timed out waiting for activation processor socket ${ACTIVATION_PROCESSOR_SOCKET}" >&2
    exit 1
fi

vllm_cmd=(
    .venv/bin/vllm serve "${MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --dtype "${VLLM_DTYPE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --extract-activation-layers
)

for layer in ${ACTIVATION_LAYERS}; do
    vllm_cmd+=("${layer}")
done

vllm_cmd+=(
    --activation-processor-socket "${ACTIVATION_PROCESSOR_SOCKET}"
)

vllm_cmd+=("$@")

"${vllm_cmd[@]}" &
vllm_pid="$!"

wait -n "${sidecar_pid}" "${vllm_pid}"
