#!/usr/bin/env python3
"""Find the optimal TP/DP split for activation extraction.

Tests all valid (tp_size, dp_size) combinations where tp * dp <= num_gpus.
For each, checks correctness (no NaN) and measures throughput.

Usage:
    uv run python -m utils.find_optimal_extraction \
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
        --num-gpus 4 \
        --layer 15 \
        --num-prompts 50
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import sys
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
    if tp_size == 1:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    # Suppress vLLM logging noise
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

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
            else:
                # No activations returned at all
                nan_count += 1
                total_count += 1

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


def test_config(
    tp_size: int, dp_size: int, model: str, layer: int, num_prompts: int, timeout: int
) -> dict:
    """Test a (tp_size, dp_size) configuration."""
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    results_dict = manager.dict()

    prompts_per_worker = max(1, num_prompts // dp_size)

    processes = []
    for rank in range(dp_size):
        p = ctx.Process(
            target=_test_worker,
            args=(rank, tp_size, model, layer, prompts_per_worker, results_dict),
        )
        p.start()
        processes.append(p)

    # Wait for all workers with timeout
    for p in processes:
        p.join(timeout=timeout)

    # Check for hangs / crashes
    for i, p in enumerate(processes):
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
            return {"error": f"worker {i} timed out after {timeout}s"}
        if p.exitcode != 0:
            # Worker crashed (OOM, segfault, etc.)
            error_msg = results_dict.get(i, {}).get("error", f"exit code {p.exitcode}")
            return {"error": f"worker {i} crashed: {error_msg}"}

    # Check all workers reported results
    worker_results = dict(results_dict)
    if len(worker_results) != dp_size:
        missing = set(range(dp_size)) - set(worker_results.keys())
        return {"error": f"workers {missing} returned no results"}

    for rank, r in worker_results.items():
        if "error" in r:
            return {"error": f"worker {rank}: {r['error']}"}

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
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per config in seconds")
    args = parser.parse_args()

    # Generate all valid (tp, dp) where tp * dp <= num_gpus
    # Order: most GPUs first, then prefer higher DP (more throughput)
    configs = []
    for tp in [1, 2, 4, 8]:
        if tp > args.num_gpus:
            continue
        for dp in range(1, args.num_gpus // tp + 1):
            if tp * dp <= args.num_gpus:
                configs.append((tp, dp))
    # Sort: most total GPUs first, then by DP (more parallelism)
    configs.sort(key=lambda x: (x[0] * x[1], x[1]), reverse=True)

    print(f"Model:   {args.model}")
    print(f"GPUs:    {args.num_gpus}")
    print(f"Layer:   {args.layer}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Timeout: {args.timeout}s per config")
    print()
    print(f"{'TP':>4} {'DP':>4} {'GPUs':>5} {'Status':>8} {'Throughput':>12} {'NaN%':>8} {'Notes'}")
    print("-" * 75)

    best = None
    results = []
    for tp, dp in configs:
        label = f"TP={tp}/DP={dp}"
        print(f"{tp:>4} {dp:>4} {tp*dp:>5}  ", end="", flush=True)

        result = test_config(tp, dp, args.model, args.layer, args.num_prompts, args.timeout)

        if "error" in result:
            err = result["error"]
            # Truncate long errors
            if len(err) > 50:
                err = err[:47] + "..."
            print(f"{'FAIL':>8} {'--':>12} {'--':>8}  {err}")
            results.append((tp, dp, "FAIL", 0, err))
            gc.collect()
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
            notes = " <-- best"

        print(f"{status:>8} {tp_str:>12} {nan_str:>8} {notes}")
        results.append((tp, dp, status, throughput, ""))

        gc.collect()

    print("-" * 75)

    # Print summary table
    print("\n\n=== RESULTS SUMMARY ===\n")
    print(f"{'TP':>4} {'DP':>4} {'GPUs':>5} {'Status':>8} {'Throughput':>12} {'Notes'}")
    print("-" * 55)
    for tp, dp, status, throughput, notes in results:
        tp_str = f"{throughput:.1f} p/s" if throughput > 0 else "--"
        mark = ""
        if best and (tp, dp) == best[0]:
            mark = " <-- BEST"
        elif notes:
            mark = f"  {notes}"
        print(f"{tp:>4} {dp:>4} {tp*dp:>5} {status:>8} {tp_str:>12} {mark}")
    print("-" * 55)

    if best:
        (bt, bd), btp = best
        print(f"\nOptimal: --extraction-tp-size {bt} --extraction-dp-size {bd}  ({btp:.1f} prompts/s)")
    else:
        print("\nNo valid configuration found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
