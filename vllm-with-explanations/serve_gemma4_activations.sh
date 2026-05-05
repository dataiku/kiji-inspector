#!/usr/bin/env bash
set -euo pipefail

# DeepGEMM kernels are broken on this build. Disable them like the harness does.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export ACTIVATION_PROCESSOR_SOCKET="${ACTIVATION_PROCESSOR_SOCKET:-/tmp/vllm-activation-processor.sock}"

vllm serve google/gemma-4-E4B-it \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --extract-activation-layers 8 \
    --activation-processor-socket "${ACTIVATION_PROCESSOR_SOCKET}"
