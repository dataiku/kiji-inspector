#!/usr/bin/env python3
"""Find the optimal TP/DP split for activation extraction.

Tests all valid (tp_size, dp_size) combinations where tp * dp <= num_gpus.
For each, checks correctness (no NaN) and measures throughput.

Usage:
    uv run python scripts/find_optimal_extraction.py \
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
        --num-gpus 4 \
        --layer 15 \
        --num-prompts 50
"""

from __future__ import annotations

import argparse
import gc
import os
import multiprocessing
import time


def _test_worker(
    rank: int,
    tp_size: int,
    model: str,
    layer: int,
    num_prompts: int,
    results_dict: dict,
):
    """Worker that loads a model with given TP and runs extraction."""
    gpu_ids = ",".join(str(rank * tp_size + i) for i in range(tp_size))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # Only disable P2P for single-GPU workers
    if tp_size == 1:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    import torch
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model=model,
            extract_activation_layers=[layer],
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            disable_log_stats=True,
        )

        sp = SamplingParams(max_tokens=1, temperature=0.0, extract_activations=True)
        prompts = ["The quick brown fox jumps over the lazy dog."] * num_prompts

        # Warmup
        llm.generate(prompts[:2], sp, use_tqdm=False)

        # Benchmark
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        # Check for NaN
        nan_count = 0
        total_count = 0
        for out in outputs:
            act = out.outputs[0].activations
            if act:
                for _, tensor in act.items():
                    nan_count += torch.isnan(tensor).sum().item()
                    total_count += tensor.numel()

        results_dict[rank] = {
            "nan_count": nan_count,
            "total_count": total_count,
            "elapsed": elapsed,
            "prompts": num_prompts,
            "throughput": num_prompts / elapsed,
        }

        del llm
        torch.cuda.empty_cache()

    except Exception as e:
        results_dict[rank] = {"error": str(e)}


def test_config(tp_size: int, dp_size: int, model: str, layer: int, num_prompts: int) -> dict:
    """Test a (tp_size, dp_size) configuration."""
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    results_dict = manager.dict()

    prompts_per_worker = num_prompts // dp_size

    processes = []
    for rank in range(dp_size):
        p = ctx.Process(
            target=_test_worker,
            args=(rank, tp_size, model, layer, prompts_per_worker, results_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=300)
        if p.is_alive():
            p.terminate()
            return {"error": "timeout"}

    # Aggregate results
    worker_results = dict(results_dict)
    for r in worker_results.values():
        if "error" in r:
            return r

    total_nan = sum(r["nan_count"] for r in worker_results.values())
    total_elements = sum(r["total_count"] for r in worker_results.values())
    max_elapsed = max(r["elapsed"] for r in worker_results.values())
    total_throughput = sum(r["throughput"] for r in worker_results.values())

    return {
        "nan_count": total_nan,
        "total_elements": total_elements,
        "nan_pct": 100 * total_nan / total_elements if total_elements else 0,
        "elapsed": max_elapsed,
        "total_throughput": total_throughput,
        "correct": total_nan == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Find optimal TP/DP for extraction")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--num-prompts", type=int, default=50)
    args = parser.parse_args()

    # Generate all valid (tp, dp) where tp * dp <= num_gpus
    configs = []
    for tp in [1, 2, 4, 8]:
        if tp > args.num_gpus:
            continue
        for dp in range(1, args.num_gpus // tp + 1):
            if tp * dp <= args.num_gpus:
                configs.append((tp, dp))

    print(f"Model: {args.model}")
    print(f"GPUs:  {args.num_gpus}")
    print(f"Layer: {args.layer}")
    print(f"Prompts per test: {args.num_prompts}")
    print()
    print(f"{'TP':>4} {'DP':>4} {'GPUs':>5} {'Correct':>8} {'Throughput':>12} {'NaN%':>8} {'Notes'}")
    print("-" * 70)

    best = None
    for tp, dp in configs:
        print(f"{tp:>4} {dp:>4} {tp*dp:>5}  ", end="", flush=True)

        result = test_config(tp, dp, args.model, args.layer, args.num_prompts)

        if "error" in result:
            print(f"{'FAIL':>8} {'—':>12} {'—':>8}  {result['error'][:40]}")
            continue

        correct = result["correct"]
        throughput = result["total_throughput"]
        nan_pct = result["nan_pct"]

        status = "OK" if correct else "NaN!"
        tp_str = f"{throughput:.1f} p/s"
        nan_str = f"{nan_pct:.1f}%"

        notes = ""
        if correct and (best is None or throughput > best[1]):
            best = ((tp, dp), throughput)
            notes = " ← best"

        print(f"{status:>8} {tp_str:>12} {nan_str:>8} {notes}")

        gc.collect()

    print("-" * 70)
    if best:
        (bt, bd), btp = best
        print(f"\nOptimal: --extraction-tp-size {bt} --extraction-dp-size {bd}  ({btp:.1f} prompts/s)")
    else:
        print("\nNo valid configuration found!")


if __name__ == "__main__":
    main()
