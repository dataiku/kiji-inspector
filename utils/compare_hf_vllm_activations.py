#!/usr/bin/env python3
"""
Verify that HF (ActivationExtractor, forward-pre-hook) and vLLM
(VLLMActivationExtractor, native hidden-states connector) capture the SAME
residual-stream tensor for a given nominal layer index, on a real model.

This is a one-off manual verification script, not a pytest test: it needs a
real GPU and a real (tens-of-GB) model checkpoint loaded twice, which is
exactly what this repo's test suite is designed to avoid (see
tests/test_activation_extractors.py — everything there is synthetic/mocked
so it can run fast, offline, and without a GPU).

Because vLLM expects to profile and own GPU memory cleanly at engine
startup, HF and vLLM extraction run as SEPARATE PROCESSES (not sequentially
in one process) to avoid any risk of leftover CUDA allocations from one
framework confusing the other's memory profiling. Usage:

    # 1. HF pass
    python compare_hf_vllm_activations.py --backend hf \\
        --model /path/to/checkpoint --layers 6 20 43 --out /tmp/hf_acts.pt

    # 2. vLLM pass (separate process)
    python compare_hf_vllm_activations.py --backend vllm \\
        --model /path/to/checkpoint --layers 6 20 43 --out /tmp/vllm_acts.pt

    # 3. Compare (no GPU needed)
    python compare_hf_vllm_activations.py --backend compare \\
        --hf-file /tmp/hf_acts.pt --vllm-file /tmp/vllm_acts.pt

Exact numeric equality is not expected: HF (eager PyTorch / mamba_ssm
kernels) and vLLM (fused Triton/CUDA Mamba2 + attention kernels, possibly a
different attention backend) compute the same math with different kernels,
so small floating-point differences are normal. The pass/fail criterion is
cosine similarity + relative L2 error, not `torch.equal`.
"""

from __future__ import annotations

import argparse

import torch

# A realistic tool-selection prompt matching the pipeline's own prompt shape
# (system + tools + user request + assistant prefill), so the decision-token
# position is meaningful rather than an arbitrary string.
PROMPT = (
    "System: You are a helpful assistant with access to tools.\n\n"
    "Available tools:\n"
    "- internal_search: Search internal company documents for policies, "
    "runbooks, and API references\n"
    "- web_search: Search the public web for general information\n\n"
    "When you decide to use a tool, respond with the tool name.\n\n"
    "User: Search our docs for the current API rate limits.\n\n"
    "Assistant: I'll use the "
)


def run_hf(model_name: str, layers: list[int], out_path: str) -> None:
    from kiji_inspector.extraction.activation_extractor import ActivationConfig, ActivationExtractor

    config = ActivationConfig(model_name=model_name, layers=layers, token_positions="decision")
    extractor = ActivationExtractor(config=config)
    try:
        result = extractor.extract(PROMPT)
    finally:
        extractor.cleanup()

    tensors = {
        f"residual_{layer}": torch.from_numpy(result[f"residual_{layer}"]) for layer in layers
    }
    torch.save(tensors, out_path)
    print(f"Saved HF activations for layers {layers} -> {out_path}")


def run_vllm(model_name: str, layers: list[int], out_path: str) -> None:
    from kiji_inspector.extraction.vllm_activation_extractor import (
        VLLMActivationConfig,
        VLLMActivationExtractor,
    )

    config = VLLMActivationConfig(
        model_name=model_name,
        layers=layers,
        token_positions="decision",
    )
    extractor = VLLMActivationExtractor(config)
    try:
        result = extractor.extract(PROMPT)
    finally:
        extractor.cleanup()

    tensors = {
        f"residual_{layer}": torch.from_numpy(result[f"residual_{layer}"]) for layer in layers
    }
    torch.save(tensors, out_path)
    print(f"Saved vLLM activations for layers {layers} -> {out_path}")


def run_compare(hf_file: str, vllm_file: str) -> None:
    hf_acts = torch.load(hf_file)
    vllm_acts = torch.load(vllm_file)

    common_keys = sorted(set(hf_acts) & set(vllm_acts), key=lambda k: int(k.split("_")[1]))
    if not common_keys:
        raise SystemExit(f"No overlapping layer keys between {hf_file} and {vllm_file}")

    print(f"\n{'=' * 70}")
    print("  HF vs vLLM activation comparison (residual stream entering layer N)")
    print(f"{'=' * 70}")
    any_mismatch = False
    for key in common_keys:
        a = hf_acts[key].float()
        b = vllm_acts[key].float()
        if a.shape != b.shape:
            print(f"  {key:>14s}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            any_mismatch = True
            continue
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
        rel_l2 = (torch.linalg.norm(a - b) / torch.linalg.norm(a)).item()
        ok = cos > 0.99 and rel_l2 < 0.1
        any_mismatch = any_mismatch or not ok
        print(
            f"  {key:>14s}: cosine={cos:.6f}  rel_l2_error={rel_l2:.6f}  "
            f"[{'MATCH' if ok else 'MISMATCH'}]"
        )
    print(f"{'=' * 70}\n")
    if any_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["hf", "vllm", "compare"])
    parser.add_argument("--model", help="Model name/path (required for hf/vllm)")
    parser.add_argument("--layers", type=int, nargs="+", help="Layer indices (required for hf/vllm)")
    parser.add_argument("--out", help="Output .pt path (required for hf/vllm)")
    parser.add_argument("--hf-file", help="HF .pt file (required for compare)")
    parser.add_argument("--vllm-file", help="vLLM .pt file (required for compare)")
    args = parser.parse_args()

    if args.backend == "hf":
        run_hf(args.model, args.layers, args.out)
    elif args.backend == "vllm":
        run_vllm(args.model, args.layers, args.out)
    else:
        run_compare(args.hf_file, args.vllm_file)
