#!/usr/bin/env python3
"""
Standalone CLI tool for generating contrastive pairs via LLM.

This is a preparatory step that produces training data for the SAE pipeline.
Pairs are saved as parquet shards and can be reused across multiple pipeline runs.

Usage:
    # Generate 1300 pairs using all scenarios
    uv run python -m kiji_inspector.generate_pairs 1300

    # Generate to a custom output directory
    uv run python -m kiji_inspector.generate_pairs 500000 --output-dir output/pairs

    # Use specific scenarios
    uv run python -m kiji_inspector.generate_pairs 1300 --scenario scenarios/tool_selection.json
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate contrastive pairs for SAE training via LLM.",
    )
    p.add_argument(
        "num_samples",
        type=int,
        help="Number of contrastive pairs to generate.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/pairs",
        help="Directory for parquet shards and scenarios_meta.json (default: output/pairs).",
    )
    p.add_argument(
        "--judging-model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        dest="judging_model",
        help="HuggingFace model for pair generation via vLLM.",
    )
    # Backward-compatible alias (hidden from help)
    p.add_argument(
        "--qwen-model",
        type=str,
        default=argparse.SUPPRESS,
        dest="judging_model",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1). "
        "When set, uses HTTP API instead of in-process vLLM.",
    )
    p.add_argument(
        "--generation-tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM generation (default: 4).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max sequence length for vLLM generation model (default: 16384).",
    )
    p.add_argument(
        "--generation-batch",
        type=int,
        default=50,
        help="Max pairs to request per LLM prompt (default: 50).",
    )
    p.add_argument(
        "--vllm-batch-size",
        type=int,
        default=128,
        help="Max prompts per vLLM generate() call (default: 128).",
    )
    p.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Path to scenario JSON config. Can be specified multiple times "
        "to select a subset. Default: all *.json files in scenarios/.",
    )
    p.add_argument(
        "--disable-p2p",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help="Disable CUDA peer-to-peer access to avoid host OOM on GB200/Blackwell. "
        "'auto' detects Blackwell GPUs and disables P2P if found (default: auto).",
    )

    return p.parse_args()


class _OpenAILLMAdapter:
    """Wraps an OpenAI-compatible API to mimic vLLM's LLM.generate() interface."""

    def __init__(self, api_base: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=api_base, api_key="unused")
        self.model = model

    def generate(self, prompts: list[str], sampling_params: object, use_tqdm: bool = False) -> list:
        import concurrent.futures

        temperature = getattr(sampling_params, "temperature", 0.7)
        top_p = getattr(sampling_params, "top_p", 0.8)
        max_tokens = getattr(sampling_params, "max_tokens", 8000)

        def _call(prompt: str) -> object:
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return resp

        # Parallel HTTP requests for throughput
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            results = list(pool.map(_call, prompts))

        # Wrap into vLLM-compatible output shape
        class _Output:
            def __init__(self, text: str):
                self.text = text

        class _RequestOutput:
            def __init__(self, text: str):
                self.outputs = [_Output(text)]

        return [_RequestOutput(r.choices[0].text) for r in results]


def _run_generation_subprocess(
    num_samples: int,
    scenario_dicts: list[dict],
    judging_model: str,
    generation_batch: int,
    vllm_batch_size: int,
    tp_size: int,
    max_model_len: int,
    output_dir: str,
    api_base: str | None = None,
) -> None:
    """Child-process entry point: load vLLM (or connect to API), generate pairs, save, exit."""
    from kiji_inspector.data.contrastive_dataset import ContrastiveDataset
    from kiji_inspector.data.generator import ContrastivePairGenerator
    from kiji_inspector.data.scenario import ScenarioConfig

    # Check for existing pairs to append to
    existing_pairs: list = []
    output_path = Path(output_dir)
    existing_shards = sorted(output_path.glob("shard_*.parquet")) if output_path.exists() else []
    if existing_shards:
        existing_dataset = ContrastiveDataset.from_parquet(output_dir)
        existing_pairs = existing_dataset.pairs
        print(
            f"  [subprocess] Found {len(existing_pairs)} existing pairs in {output_dir}, will append"
        )

    if api_base:
        print(f"  [subprocess] Using OpenAI-compatible API: {api_base}")
        print(f"  [subprocess] Model: {judging_model}")
        llm = _OpenAILLMAdapter(api_base, judging_model)

        class _SamplingParams:
            def __init__(self, **kwargs: object):
                self.__dict__.update(kwargs)

        sampling_params = _SamplingParams(temperature=0.7, top_p=0.8, max_tokens=8000)
    else:
        from vllm import LLM, SamplingParams

        print(f"  [subprocess] Loading vLLM model: {judging_model}")
        print(f"  [subprocess] tensor_parallel_size={tp_size}, max_model_len={max_model_len}")

        llm = LLM(
            model=judging_model,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            disable_log_stats=True,
        )

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=8000,
        )

    scenarios = [ScenarioConfig.from_dict(d) for d in scenario_dicts]
    pairs_per_scenario = math.ceil(num_samples / len(scenarios))

    from tqdm import tqdm

    all_pairs: list = []

    # Single overall progress bar
    pbar = tqdm(
        total=num_samples,
        desc="Generating pairs",
        unit="pair",
        unit_scale=True,
        smoothing=0.1,
    )

    for scenario in scenarios:
        gen = ContrastivePairGenerator(
            llm=llm,
            tools=scenario.tools,
            contrast_types=scenario.contrast_types,
            scenario_name=scenario.name,
            sampling_params=sampling_params,
        )

        n_types = len(scenario.contrast_types)
        pairs_per_type = math.ceil(pairs_per_scenario / n_types)

        requests: list[tuple[str, int]] = []
        for ct in scenario.contrast_types:
            remaining = pairs_per_type
            while remaining > 0:
                chunk = min(remaining, generation_batch)
                requests.append((ct, chunk))
                remaining -= chunk

        pair_counters: dict[str, int] = {}
        scenario_pairs: list = []

        for round_start in range(0, len(requests), vllm_batch_size):
            round_requests = requests[round_start : round_start + vllm_batch_size]
            results = gen.generate_batched(round_requests)

            for (ct, _n), pairs in zip(round_requests, results, strict=True):
                base = pair_counters.get(ct, 0)
                for i, pair in enumerate(pairs):
                    pair.pair_id = f"{scenario.name}_{ct}_{base + i}"
                pair_counters[ct] = base + len(pairs)
                scenario_pairs.extend(pairs)
                pbar.update(len(pairs))

            # Update description with current scenario context
            pbar.set_postfix_str(f"{scenario.name} ({len(scenario_pairs)}/{pairs_per_scenario})")

        if gen._malformed_count > 0:
            print(f"  [subprocess] {scenario.name}: {gen._malformed_count} malformed pairs skipped")
        all_pairs.extend(scenario_pairs[:pairs_per_scenario])

    pbar.close()
    all_pairs = all_pairs[:num_samples]
    print(f"  [subprocess] New pairs generated: {len(all_pairs)}")

    # Merge with existing pairs if any were found
    if existing_pairs:
        combined_pairs = existing_pairs + all_pairs
        print(
            f"  [subprocess] Combined total: {len(existing_pairs)} existing + {len(all_pairs)} new = {len(combined_pairs)} pairs"
        )
    else:
        combined_pairs = all_pairs

    dataset = ContrastiveDataset(pairs=combined_pairs)
    shards = dataset.to_parquet(output_dir)
    print(f"  [subprocess] Saved {len(shards)} shard(s) to {output_dir}")


def generate_pairs(
    num_samples: int,
    scenario_dicts: list[dict],
    judging_model: str,
    generation_batch: int,
    vllm_batch_size: int,
    tp_size: int,
    max_model_len: int,
    output_dir: str,
    api_base: str | None = None,
) -> list:
    """Spawn a subprocess to generate pairs via vLLM (or OpenAI API), then load results."""
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_run_generation_subprocess,
        args=(
            num_samples,
            scenario_dicts,
            judging_model,
            generation_batch,
            vllm_batch_size,
            tp_size,
            max_model_len,
            output_dir,
            api_base,
        ),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Generation subprocess failed with exit code {p.exitcode}")

    from kiji_inspector.data.contrastive_dataset import ContrastiveDataset

    dataset = ContrastiveDataset.from_parquet(output_dir)
    return dataset.pairs


def main() -> None:
    args = parse_args()

    # Apply Blackwell P2P mitigations before any CUDA context is created
    if not args.api_base:
        from kiji_inspector.pipeline import _apply_p2p_mitigations

        _apply_p2p_mitigations(args.disable_p2p)

    from kiji_inspector.data.scenario import discover_scenarios, save_scenarios_meta

    scenarios = discover_scenarios(args.scenarios)
    scenario_names = [s.name for s in scenarios]
    total_contrast_types = sum(len(s.contrast_types) for s in scenarios)

    print("=" * 60)
    print("  Contrastive Pair Generation")
    print(f"  Requested samples : {args.num_samples:,}")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Scenarios         : {', '.join(scenario_names)}")
    print(f"  Contrast types    : {total_contrast_types} total across {len(scenarios)} scenario(s)")
    print(f"  Generation model  : {args.judging_model} (vLLM, TP={args.generation_tp_size})")
    print("=" * 60)

    t0 = time.time()
    pairs = generate_pairs(
        num_samples=args.num_samples,
        scenario_dicts=[s.to_dict() for s in scenarios],
        judging_model=args.judging_model,
        generation_batch=args.generation_batch,
        vllm_batch_size=args.vllm_batch_size,
        tp_size=args.generation_tp_size,
        max_model_len=args.max_model_len,
        output_dir=args.output_dir,
        api_base=args.api_base,
    )

    # Save scenarios_meta.json alongside the pairs
    meta_path = save_scenarios_meta(scenarios, Path(args.output_dir))
    print(f"  Saved scenarios metadata: {meta_path}")

    elapsed = time.time() - t0
    print(f"  Generation complete ({elapsed:.1f}s)")
    print(f"  {len(pairs)} total pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
