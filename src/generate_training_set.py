#!/usr/bin/env python3
"""
End-to-end pipeline for contrastive SAE analysis of tool-selection decisions.

Step 1: Generate contrastive pairs via Qwen3-VL (vLLM subprocess)
Step 2: Extract raw activations from Nemotron as numpy shards
Step 3: Train a JumpReLU SAE on the raw activations
Step 4: Identify decision-relevant SAE features using contrastive pairs
Step 5: Interpret features -- label with LLM, generate decision report
Step 6: Evaluate feature explanations via fuzzing

Usage:
    # Run steps 1 + 2 + 3 (full data pipeline + training)
    uv run python generate_training_set.py 1300

    # Generate pairs (num_samples required)
    uv run python generate_training_set.py 1300 --step 1

    # Steps 2-6 read pairs from disk (num_samples optional)
    uv run python generate_training_set.py --step 2 --pairs-dir output/pairs
    uv run python generate_training_set.py --step 3
    uv run python generate_training_set.py --step 4
    uv run python generate_training_set.py --step 5
    uv run python generate_training_set.py --step 6

    # Large-scale
    uv run python generate_training_set.py 100000 --output-dir output/activations
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a contrastive SAE training set.",
    )
    p.add_argument(
        "num_samples",
        type=int,
        nargs="?",
        default=None,
        help="Number of contrastive pairs to generate (required for step 1).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/activations",
        help="Directory for numpy activation shards (default: output/activations).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="GPU batch size (prompts) for activation extraction (default: 512).",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Activation vectors per numpy shard (default: 500000).",
    )
    # -- Generation model (Step 1) --
    p.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="HuggingFace model for pair generation via vLLM.",
    )
    p.add_argument(
        "--generation-tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM generation (default: 4 for 4xGB200).",
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
    # -- Extraction model (Step 2) --
    p.add_argument(
        "--nemotron-model",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="HuggingFace model for activation extraction.",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20],
        help="Transformer layers to hook (default: 20).",
    )
    p.add_argument(
        "--layer-key",
        type=str,
        default="residual_20",
        help="Layer key for activation extraction (default: residual_20).",
    )
    # -- SAE training (Step 3) --
    p.add_argument(
        "--d-sae",
        type=int,
        default=16384,
        help="SAE hidden dimension (default: 16384, typically 4-8x d_model).",
    )
    p.add_argument(
        "--sae-lr",
        type=float,
        default=3e-4,
        help="SAE training learning rate (default: 3e-4).",
    )
    p.add_argument(
        "--sae-batch-size",
        type=int,
        default=8192,
        help="SAE training batch size in tokens (default: 8192).",
    )
    p.add_argument(
        "--sae-steps",
        type=int,
        default=None,
        help="SAE training steps (default: auto from data size).",
    )
    p.add_argument(
        "--sae-epochs",
        type=int,
        default=10,
        help="SAE training epochs (default: 10).",
    )
    p.add_argument(
        "--no-auto-scale-steps",
        action="store_true",
        default=False,
        help="Disable auto-scaling of warmup/checkpoint step counts.",
    )
    p.add_argument(
        "--l1-coefficient",
        type=float,
        default=5e-3,
        help="SAE sparsity penalty (default: 5e-3).",
    )
    p.add_argument(
        "--sae-checkpoint-dir",
        type=str,
        default=None,
        help="SAE checkpoint directory (default: output/sae_checkpoints).",
    )
    p.add_argument(
        "--sae-resume",
        type=str,
        default=None,
        help="Resume SAE training from checkpoint.",
    )
    # -- Feature identification (Step 4) --
    p.add_argument(
        "--sae-checkpoint",
        type=str,
        default=None,
        help="Path to trained SAE checkpoint for step 4 (default: auto from step 3).",
    )
    p.add_argument(
        "--top-k-features",
        type=int,
        default=200,
        help="Number of top features to report per contrast type (default: 200).",
    )
    p.add_argument(
        "--min-effect-size",
        type=float,
        default=0.3,
        help="Minimum Cohen's d for contrastive features (default: 0.3).",
    )
    p.add_argument(
        "--min-activation",
        type=float,
        default=0.01,
        help="Minimum mean activation for at least one condition (default: 0.01).",
    )
    # -- Feature interpretation (Step 5) --
    p.add_argument(
        "--label-top-n",
        type=int,
        default=20,
        help="Top-activating examples per feature for labeling (default: 20).",
    )
    p.add_argument(
        "--label-bottom-n",
        type=int,
        default=10,
        help="Near-zero examples per feature for labeling (default: 10).",
    )
    p.add_argument(
        "--label-batch-size",
        type=int,
        default=512,
        help="GPU batch size for activation extraction in step 5a (default: 512).",
    )
    # -- Fuzzing evaluation (Step 6) --
    p.add_argument(
        "--fuzz-top-k-tokens",
        type=int,
        default=5,
        help="Tokens to highlight per fuzzing example (default: 5).",
    )
    p.add_argument(
        "--fuzz-examples-per-feature",
        type=int,
        default=10,
        help="Max fuzzing examples per feature (default: 10).",
    )
    p.add_argument(
        "--fuzz-batch-size",
        type=int,
        default=64,
        help="GPU batch size for per-token extraction in step 6a (default: 64).",
    )
    # -- Scenario configs --
    p.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Path to scenario JSON config. Can be specified multiple times "
        "to select a subset. Default: all *.json files in scenarios/.",
    )
    # -- Flow control --
    p.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="Run a single step: 1=generate, 2=extract, 3=train SAE, "
        "4=identify features, 5=interpret features, 6=fuzzing eval "
        "(default: all runs 1+2+3).",
    )
    p.add_argument(
        "--pairs-dir",
        type=str,
        default=None,
        help="Path to existing pairs parquet directory to skip generation.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Generate contrastive pairs (runs in a SUBPROCESS)
# ---------------------------------------------------------------------------


def _run_generation_subprocess(
    num_samples: int,
    scenario_dicts: list[dict],
    qwen_model: str,
    generation_batch: int,
    vllm_batch_size: int,
    tp_size: int,
    max_model_len: int,
    output_dir: str,
) -> None:
    """Child-process entry point: load vLLM, generate pairs for all scenarios, save, exit."""
    from vllm import LLM, SamplingParams

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from contrastive_dataset import ContrastiveDataset
    from generator import ContrastivePairGenerator
    from scenario import ScenarioConfig

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

    print(f"  [subprocess] Loading vLLM model: {qwen_model}")
    print(f"  [subprocess] tensor_parallel_size={tp_size}, max_model_len={max_model_len}")

    llm = LLM(
        model=qwen_model,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        enable_expert_parallel=True,
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
    qwen_model: str,
    generation_batch: int,
    vllm_batch_size: int,
    tp_size: int,
    max_model_len: int,
    pairs_dir: str,
) -> list:
    """Spawn a subprocess to generate pairs via vLLM, then load results."""
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_run_generation_subprocess,
        args=(
            num_samples,
            scenario_dicts,
            qwen_model,
            generation_batch,
            vllm_batch_size,
            tp_size,
            max_model_len,
            pairs_dir,
        ),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Generation subprocess failed with exit code {p.exitcode}")

    from contrastive_dataset import ContrastiveDataset

    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    return dataset.pairs


# ---------------------------------------------------------------------------
# Step 2: Extract raw activations (numpy shards)
# ---------------------------------------------------------------------------


def extract_activations(
    pairs: list,
    output_dir: str,
    nemotron_model: str,
    layers: list[int],
    layer_key: str,
    batch_size: int,
    shard_size: int,
    scenarios_meta: dict | None = None,
) -> Path:
    """Load Nemotron, extract raw activations, save as numpy shards."""
    import torch

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from activation_extractor import ActivationConfig, ActivationExtractor
    from extractor import RawActivationExtractor

    config = ActivationConfig(
        model_name=nemotron_model,
        layers=layers,
        dtype=torch.bfloat16,
    )

    extractor = ActivationExtractor(config=config)
    raw_extractor = RawActivationExtractor(
        base_extractor=extractor,
        model_type="nemotron",
        layer_key=layer_key,
    )

    result_dir = raw_extractor.extract_to_shards(
        pairs=pairs,
        output_dir=output_dir,
        batch_size=batch_size,
        shard_size=shard_size,
        scenarios_meta=scenarios_meta,
    )

    extractor.cleanup()
    return result_dir


# ---------------------------------------------------------------------------
# Step 3: Train JumpReLU SAE
# ---------------------------------------------------------------------------


def train_sae_step(
    activations_dir: str,
    checkpoint_dir: str,
    d_sae: int,
    learning_rate: float,
    batch_size: int,
    l1_coefficient: float,
    total_steps: int | None,
    num_epochs: int,
    resume_from: str | None,
    auto_scale_steps: bool = True,
) -> str:
    """Train a JumpReLU SAE on the numpy activation shards from Step 2."""
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from sae_trainer import SAETrainingConfig, train_sae

    config = SAETrainingConfig(
        d_sae=d_sae,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l1_coefficient=l1_coefficient,
        total_steps=total_steps,
        num_epochs=num_epochs,
        output_dir=checkpoint_dir,
        resume_from=resume_from,
        auto_scale_steps=auto_scale_steps,
    )

    return train_sae(activations_dir=activations_dir, config=config)


# ---------------------------------------------------------------------------
# Step 4: Contrastive feature identification
# ---------------------------------------------------------------------------


def identify_contrastive_features(
    pairs: list,
    sae_checkpoint: str,
    nemotron_model: str,
    layers: list[int],
    layer_key: str,
    batch_size: int,
    top_k: int,
    output_dir: str,
    min_effect_size: float = 0.3,
    min_activation: float = 0.01,
    scenarios_meta: dict | None = None,
) -> Path:
    """Identify which SAE features are decision-relevant using contrastive pairs.

    For each contrastive pair, encode both the anchor and contrast prompts
    through the SAE and compare which features activate differently.  Features
    that consistently differ across many pairs of the same contrast type are
    the decision-relevant features for that contrast.
    """
    import torch

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from activation_extractor import ActivationConfig, ActivationExtractor
    from extractor import build_agent_prompt
    from sae_model import JumpReLUSAE
    from scenario import default_scenario

    # Load the trained SAE
    sae_path = Path(sae_checkpoint)
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(str(sae_path), device=device)
    sae.eval()
    print(f"  Loaded SAE: d_model={sae.d_model}, d_sae={sae.d_sae}")

    # Load Nemotron for live extraction of contrastive pairs
    config = ActivationConfig(
        model_name=nemotron_model,
        layers=layers,
        dtype=torch.bfloat16,
    )
    extractor = ActivationExtractor(config=config)

    # Build scenario lookup for per-pair prompt construction
    _scenarios_meta = scenarios_meta or {}
    _default_scenario = default_scenario()

    # Group pairs by contrast type
    from collections import defaultdict

    pairs_by_type: dict[str, list] = defaultdict(list)
    for pair in pairs:
        pairs_by_type[pair.contrast_type].append(pair)

    results: dict[str, dict] = {}
    sae_dtype = next(sae.parameters()).dtype
    from tqdm import tqdm

    pbar = tqdm(total=len(pairs), desc="Extracting pair activations", unit="pair")

    for ct_value, ct_pairs in pairs_by_type.items():
        # Build all prompts for this contrast type
        anchor_prompts = []
        contrast_prompts = []
        for pair in ct_pairs:
            scenario = _scenarios_meta.get(pair.scenario_name, _default_scenario)
            anchor_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.anchor_prompt,
                    model_type="nemotron",
                )
            )
            contrast_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.contrast_prompt,
                    model_type="nemotron",
                )
            )

        # Extract in batches, accumulating results
        all_prompts = anchor_prompts + contrast_prompts
        all_acts: list[dict[str, np.ndarray]] = []
        for bi in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[bi : bi + batch_size]
            all_acts.extend(extractor.extract_batch(batch_prompts, batch_size=len(batch_prompts)))

        pbar.update(len(ct_pairs))

        n = len(ct_pairs)
        anchor_acts = [all_acts[i][layer_key] for i in range(n)]
        contrast_acts = [all_acts[n + i][layer_key] for i in range(n)]

        anchor_vecs = torch.from_numpy(np.stack(anchor_acts)).to(device=device, dtype=sae_dtype)
        contrast_vecs = torch.from_numpy(np.stack(contrast_acts)).to(device=device, dtype=sae_dtype)

        with torch.no_grad():
            anchor_features = sae.encode(anchor_vecs)
            contrast_features = sae.encode(contrast_vecs)

        feature_diffs = (anchor_features - contrast_features).abs().mean(dim=0)
        anchor_mean = anchor_features.mean(dim=0)
        contrast_mean = contrast_features.mean(dim=0)
        anchor_var = anchor_features.var(dim=0)
        contrast_var = contrast_features.var(dim=0)
        n_a = anchor_features.shape[0]
        n_c = contrast_features.shape[0]

        # Cohen's d effect size
        pooled_std = torch.sqrt(
            ((n_a - 1) * anchor_var + (n_c - 1) * contrast_var) / max(n_a + n_c - 2, 1)
        )
        cohens_d = (anchor_mean - contrast_mean).abs() / (pooled_std + 1e-8)

        # Filter: effect size >= threshold AND at least one side has meaningful activation
        effect_mask = cohens_d >= min_effect_size
        activation_mask = (anchor_mean.abs() > min_activation) | (
            contrast_mean.abs() > min_activation
        )
        valid_mask = effect_mask & activation_mask

        # Apply mask before top-K
        masked_diffs = feature_diffs.clone()
        masked_diffs[~valid_mask] = -1.0

        k_actual = min(top_k, int(valid_mask.sum().item()), feature_diffs.shape[0])
        if k_actual > 0:
            topk_vals, topk_indices = masked_diffs.topk(k_actual)
            # Keep only features that passed the mask
            keep = topk_vals > 0
            topk_vals = topk_vals[keep]
            topk_indices = topk_indices[keep]
        else:
            topk_vals = torch.tensor([])
            topk_indices = torch.tensor([], dtype=torch.long)

        feature_list = []
        for rank, (val, idx) in enumerate(
            zip(topk_vals.tolist(), topk_indices.tolist(), strict=True)
        ):
            idx = int(idx)
            feature_list.append(
                {
                    "rank": rank,
                    "feature_index": idx,
                    "mean_abs_diff": round(val, 6),
                    "anchor_mean_activation": round(anchor_mean[idx].item(), 6),
                    "contrast_mean_activation": round(contrast_mean[idx].item(), 6),
                    "cohens_d": round(cohens_d[idx].item(), 6),
                    "anchor_std": round(anchor_var[idx].sqrt().item(), 6),
                    "contrast_std": round(contrast_var[idx].sqrt().item(), 6),
                }
            )

        n_filtered_effect = int((~effect_mask).sum().item())
        n_filtered_activation = int((effect_mask & ~activation_mask).sum().item())

        results[ct_value] = {
            "num_pairs": n,
            "top_features": feature_list,
            "num_filtered_by_effect_size": n_filtered_effect,
            "num_filtered_by_min_activation": n_filtered_activation,
        }

        del anchor_vecs, contrast_vecs, anchor_features, contrast_features

    pbar.close()

    extractor.cleanup()

    # Compute utilization summary
    from collections import Counter

    all_feature_indices: list[int] = []
    for _ct_value, ct_info in results.items():
        for f in ct_info["top_features"]:
            all_feature_indices.append(f["feature_index"])

    unique_features = set(all_feature_indices)
    total_slots = len(all_feature_indices)
    feature_counts = Counter(all_feature_indices)
    multi_contrast = sum(1 for c in feature_counts.values() if c > 1)

    results["_summary"] = {
        "total_feature_slots": total_slots,
        "unique_features": len(unique_features),
        "dedup_ratio": round(len(unique_features) / max(total_slots, 1), 4),
        "features_in_multiple_contrasts": multi_contrast,
        "max_contrast_overlap": max(feature_counts.values()) if feature_counts else 0,
        "top_k_per_type": top_k,
        "min_effect_size": min_effect_size,
        "min_activation": min_activation,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "contrastive_features.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nContrastive feature report saved to {report_path}")
    print(f"  Contrast types analyzed: {len(results) - 1}")  # exclude _summary
    print(f"  Max top-{top_k} features per type (filtered by Cohen's d >= {min_effect_size})")
    print(f"  Total feature slots: {total_slots}")
    print(f"  Unique features: {len(unique_features)}")
    print(f"  Features in multiple contrasts: {multi_contrast}")
    print(f"  Dedup ratio: {results['_summary']['dedup_ratio']:.2%}")

    for ct_value, info in results.items():
        if ct_value.startswith("_"):
            continue
        n_feats = len(info["top_features"])
        top3 = info["top_features"][:3]
        top3_str = ", ".join(f"#{f['feature_index']}(d={f['cohens_d']:.2f})" for f in top3)
        print(f"  {ct_value}: {info['num_pairs']} pairs, {n_feats} features, top: {top3_str}")

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_pairs(args, pairs_dir: str) -> list:
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from contrastive_dataset import ContrastiveDataset

    src = args.pairs_dir or pairs_dir
    src_path = Path(src)
    if not src_path.exists() or not list(src_path.glob("shard_*.parquet")):
        print(f"  ERROR: no pair parquet shards found in {src}")
        print("  Run step 1 first, or provide --pairs-dir.")
        sys.exit(1)
    dataset = ContrastiveDataset.from_parquet(src)
    pairs = dataset.pairs
    if args.num_samples is not None:
        pairs = pairs[: args.num_samples]
    print(f"  Loaded {len(pairs)} pairs from {src}")
    return pairs


def _load_scenarios(args) -> list:
    """Load scenario configs from CLI args or discover all from scenarios/.

    When no --scenario flags are provided, all *.json files in the
    scenarios/ directory are loaded automatically.  Use --scenario to
    select a subset for targeted runs.
    """
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from scenario import load_scenarios

    if args.scenarios:
        scenario_paths = args.scenarios
    else:
        # Default: load all scenario files from scenarios/ directory
        scenarios_dir = Path(__file__).resolve().parent / "scenarios"
        scenario_paths = sorted(scenarios_dir.glob("*.json"))
        if not scenario_paths:
            raise FileNotFoundError(
                f"No scenario files found in {scenarios_dir}. "
                "Create at least one .json file or use --scenario."
            )

    return load_scenarios(scenario_paths)


def _run_step1(args, pairs_dir: str) -> None:
    if args.pairs_dir:
        print(f"\n[Step 1] Pairs already provided via --pairs-dir {args.pairs_dir}, skipping.")
        return

    if args.num_samples is None:
        print("\n[Step 1] ERROR: num_samples is required for pair generation.")
        print("  Usage: generate_training_set.py 1300 --step 1")
        sys.exit(1)

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from scenario import save_scenarios_meta

    scenarios = _load_scenarios(args)
    scenario_names = [s.name for s in scenarios]
    print("\n[Step 1] Generating contrastive pairs via vLLM (subprocess)...")
    print(f"  Scenarios: {', '.join(scenario_names)}")

    t0 = time.time()
    pairs = generate_pairs(
        num_samples=args.num_samples,
        scenario_dicts=[s.to_dict() for s in scenarios],
        qwen_model=args.qwen_model,
        generation_batch=args.generation_batch,
        vllm_batch_size=args.vllm_batch_size,
        tp_size=args.generation_tp_size,
        max_model_len=args.max_model_len,
        pairs_dir=pairs_dir,
    )

    # Save scenarios_meta.json alongside the pairs
    meta_path = save_scenarios_meta(scenarios, Path(pairs_dir))
    print(f"  Saved scenarios metadata: {meta_path}")

    elapsed = time.time() - t0
    print(f"  Generation complete ({elapsed:.1f}s)")
    print(f"  {len(pairs)} total pairs in {pairs_dir}")


def _run_step2(args, pairs_dir: str) -> None:
    pairs = _load_pairs(args, pairs_dir)

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from scenario import load_scenarios_meta

    scenarios_meta = load_scenarios_meta(Path(args.pairs_dir or pairs_dir))
    scenario_names = list(scenarios_meta.keys())

    print("\n[Step 2] Extracting raw activations to numpy shards...")
    print(f"  Output: {args.output_dir}")
    print(f"  Scenarios: {', '.join(scenario_names)}")
    print("  Each pair -> 2 activation vectors (anchor + contrast)")
    print(f"  Total prompts: {len(pairs) * 2}")
    t0 = time.time()
    extract_activations(
        pairs=pairs,
        output_dir=args.output_dir,
        nemotron_model=args.nemotron_model,
        layers=args.layers,
        layer_key=args.layer_key,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        scenarios_meta=scenarios_meta,
    )
    elapsed = time.time() - t0
    print(f"  Extraction complete ({elapsed:.1f}s)")


def _run_step3(args) -> str:
    checkpoint_dir = args.sae_checkpoint_dir or str(
        Path(args.output_dir).parent / "sae_checkpoints"
    )

    print("\n[Step 3] Training JumpReLU SAE...")
    print(f"  Activations: {args.output_dir}")
    print(f"  d_sae: {args.d_sae}")
    print(f"  Checkpoints: {checkpoint_dir}")
    t0 = time.time()
    final_path = train_sae_step(
        activations_dir=args.output_dir,
        checkpoint_dir=checkpoint_dir,
        d_sae=args.d_sae,
        learning_rate=args.sae_lr,
        batch_size=args.sae_batch_size,
        l1_coefficient=args.l1_coefficient,
        total_steps=args.sae_steps,
        num_epochs=args.sae_epochs,
        resume_from=args.sae_resume,
        auto_scale_steps=not args.no_auto_scale_steps,
    )
    elapsed = time.time() - t0
    print(f"  SAE training complete ({elapsed:.1f}s)")
    print(f"  Final checkpoint: {final_path}")
    return final_path


def _run_step4(args, pairs_dir: str, sae_checkpoint: str | None = None) -> None:
    checkpoint = sae_checkpoint or _resolve_sae_checkpoint(args)

    pairs = _load_pairs(args, pairs_dir)

    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from scenario import load_scenarios_meta

    scenarios_meta = load_scenarios_meta(Path(args.pairs_dir or pairs_dir))

    print("\n[Step 4] Identifying contrastive SAE features...")
    print(f"  SAE checkpoint: {checkpoint}")
    print(f"  Scenarios: {', '.join(scenarios_meta.keys())}")
    t0 = time.time()
    identify_contrastive_features(
        pairs=pairs,
        sae_checkpoint=checkpoint,
        nemotron_model=args.nemotron_model,
        layers=args.layers,
        layer_key=args.layer_key,
        batch_size=args.batch_size,
        top_k=args.top_k_features,
        output_dir=args.output_dir,
        min_effect_size=args.min_effect_size,
        min_activation=args.min_activation,
        scenarios_meta=scenarios_meta,
    )
    elapsed = time.time() - t0
    print(f"  Feature identification complete ({elapsed:.1f}s)")


def _resolve_sae_checkpoint(args) -> str:
    """Find the SAE checkpoint path from CLI args or default location."""
    if args.sae_checkpoint:
        return args.sae_checkpoint
    default_path = (
        Path(args.sae_checkpoint_dir or str(Path(args.output_dir).parent / "sae_checkpoints"))
        / "sae_final.pt"
    )
    if default_path.exists():
        return str(default_path)
    print("\n  ERROR: SAE checkpoint not found.")
    print("  Run step 3 first, or provide --sae-checkpoint.")
    sys.exit(1)


def _resolve_contrastive_features(args) -> str:
    """Find the contrastive_features.json from step 4 output."""
    path = Path(args.output_dir) / "contrastive_features.json"
    if path.exists():
        return str(path)
    print("\n  ERROR: contrastive_features.json not found.")
    print(f"  Expected at: {path}")
    print("  Run step 4 first.")
    sys.exit(1)


def _run_step5(args, sae_checkpoint: str | None = None) -> None:
    """Step 5: Interpret features -- load activations from shards, encode
    through SAE, label via LLM, generate decision report."""
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from feature_interpreter import (
        collect_max_activating_examples,
        generate_explanation_report,
        label_features_via_llm,
        load_activations_from_shards,
    )

    checkpoint = sae_checkpoint or _resolve_sae_checkpoint(args)
    contrastive_path = _resolve_contrastive_features(args)

    # Determine which features to analyze from contrastive_features.json
    with open(contrastive_path) as f:
        contrastive = json.load(f)

    feature_indices_set: set[int] = set()
    for ct_info in contrastive.values():
        for feat in ct_info.get("top_features", []):
            feature_indices_set.add(feat["feature_index"])
    feature_indices = sorted(feature_indices_set)
    print(f"\n[Step 5] Interpreting {len(feature_indices)} unique features")

    # 5a: Load activations from Step 2 numpy shards (no Nemotron needed)
    print("\n[Step 5a] Loading activations from numpy shards...")
    t0 = time.time()
    prompts, activations = load_activations_from_shards(
        activations_dir=args.output_dir,
    )
    elapsed = time.time() - t0
    print(f"  5a complete ({elapsed:.1f}s): {len(prompts)} prompts, shape {activations.shape}")

    # 5b: Encode through SAE, collect examples
    print("\n[Step 5b] Encoding through SAE and collecting max-activating examples...")
    t0 = time.time()
    feature_examples = collect_max_activating_examples(
        prompts=prompts,
        activations=activations,
        sae_checkpoint=checkpoint,
        feature_indices=feature_indices,
        top_n=args.label_top_n,
        bottom_n=args.label_bottom_n,
    )
    elapsed = time.time() - t0
    print(f"  5b complete ({elapsed:.1f}s): {len(feature_examples)} features analyzed")

    # Free Nemotron activations before loading vLLM for labeling
    del activations

    # 5c: Label features via LLM (subprocess for GPU isolation)
    print("\n[Step 5c] Labeling features via LLM (subprocess)...")
    t0 = time.time()
    feature_labels = label_features_via_llm(
        feature_examples=feature_examples,
        qwen_model=args.qwen_model,
        tp_size=args.generation_tp_size,
        max_model_len=args.max_model_len,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t0
    print(f"  5c complete ({elapsed:.1f}s): {len(feature_labels)} features labeled")

    # 5d: Generate report
    print("\n[Step 5d] Generating explanation report...")
    t0 = time.time()
    report_path = generate_explanation_report(
        contrastive_features_path=contrastive_path,
        feature_examples=feature_examples,
        feature_labels=feature_labels,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t0
    print(f"  5d complete ({elapsed:.1f}s): {report_path}")


def _resolve_feature_descriptions(args) -> str:
    """Find the feature_descriptions.json from step 5 output."""
    candidates = [
        Path(args.output_dir) / "feature_descriptions.json",
        Path(args.output_dir) / "activations" / "feature_descriptions.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    print("\n  ERROR: feature_descriptions.json not found.")
    print(f"  Searched: {[str(p) for p in candidates]}")
    print("  Run step 5 first.")
    sys.exit(1)


def _run_step6(args, pairs_dir: str) -> None:
    """Step 6: Evaluate feature explanations via fuzzing."""
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from extractor import build_agent_prompt
    from fuzzing_evaluator import (
        build_fuzzing_examples,
        compute_fuzzing_metrics,
        evaluate_fuzzing,
        extract_per_token_activations,
        save_fuzzing_report,
    )
    from scenario import default_scenario, load_scenarios_meta

    desc_path = _resolve_feature_descriptions(args)
    checkpoint = _resolve_sae_checkpoint(args)

    with open(desc_path) as f:
        feature_descriptions = json.load(f)

    # Load scenarios for per-prompt formatting
    scenarios_meta = load_scenarios_meta(Path(args.pairs_dir or pairs_dir))
    _default_scenario = default_scenario()

    # Load pairs to map prompts back to their scenario
    pairs = _load_pairs(args, pairs_dir)
    prompt_to_scenario: dict[str, str] = {}
    for pair in pairs:
        prompt_to_scenario[pair.anchor_prompt] = pair.scenario_name
        prompt_to_scenario[pair.contrast_prompt] = pair.scenario_name

    print(f"\n[Step 6] Fuzzing evaluation of {len(feature_descriptions)} features")
    print(f"  Scenarios: {', '.join(scenarios_meta.keys())}")

    # Collect all unique prompts referenced in feature descriptions
    all_prompts: list[str] = []
    seen: set[str] = set()
    for desc in feature_descriptions.values():
        for prompt in desc.get("top_examples", []) + desc.get("bottom_examples", []):
            if prompt not in seen:
                seen.add(prompt)
                all_prompts.append(prompt)

    print(f"  Unique prompts to process: {len(all_prompts)}")

    # Build formatted prompts for Nemotron (per-prompt scenario lookup)
    formatted_prompts = []
    for req in all_prompts:
        scenario_name = prompt_to_scenario.get(req, "")
        scenario = scenarios_meta.get(scenario_name, _default_scenario)
        formatted_prompts.append(
            build_agent_prompt(
                system_prompt=scenario.system_prompt,
                tools=scenario.tools,
                user_request=req,
                model_type="nemotron",
            )
        )

    # 6a: Extract per-token activations
    print("\n[Step 6a] Extracting per-token activations...")
    t0 = time.time()
    token_strings_list, token_activations_list = extract_per_token_activations(
        prompts=all_prompts,
        formatted_prompts=formatted_prompts,
        nemotron_model=args.nemotron_model,
        layers=args.layers,
        layer_key=args.layer_key,
        batch_size=args.fuzz_batch_size,
    )
    elapsed = time.time() - t0
    print(f"  6a complete ({elapsed:.1f}s): {len(token_strings_list)} prompts")

    # Build prompt -> index mapping
    prompt_to_idx = {p: i for i, p in enumerate(all_prompts)}

    # We need the tokenizer for text reconstruction in 6b.
    # Load it once (lightweight, no GPU).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.nemotron_model,
        trust_remote_code=True,
    )

    # 6b: Build fuzzing examples
    print("\n[Step 6b] Building fuzzing examples...")
    t0 = time.time()
    examples = build_fuzzing_examples(
        feature_descriptions=feature_descriptions,
        prompt_to_idx=prompt_to_idx,
        token_strings_list=token_strings_list,
        token_activations_list=token_activations_list,
        sae_checkpoint=checkpoint,
        tokenizer=tokenizer,
        top_k_tokens=args.fuzz_top_k_tokens,
        max_examples_per_feature=args.fuzz_examples_per_feature,
    )
    elapsed = time.time() - t0
    print(f"  6b complete ({elapsed:.1f}s): {len(examples)} examples")

    # Free per-token data and GPU memory before loading vLLM
    del token_strings_list, token_activations_list
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 6c: LLM judge (subprocess)
    print("\n[Step 6c] Running LLM judge (subprocess)...")
    t0 = time.time()
    results = evaluate_fuzzing(
        examples=examples,
        feature_descriptions=feature_descriptions,
        qwen_model=args.qwen_model,
        tp_size=args.generation_tp_size,
        max_model_len=args.max_model_len,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t0
    print(f"  6c complete ({elapsed:.1f}s): {len(results)} judgments")

    # 6d: Compute metrics and save
    print("\n[Step 6d] Computing metrics and saving report...")
    t0 = time.time()
    per_feature, summary = compute_fuzzing_metrics(results, feature_descriptions)
    save_fuzzing_report(per_feature, summary, results, args.output_dir)
    elapsed = time.time() - t0
    print(f"  6d complete ({elapsed:.1f}s)")


def main() -> None:
    args = parse_args()

    # "all" runs steps 1 + 2 + 3 (steps 4-6 require explicit invocation)
    steps = ["1", "2", "3"] if args.step == "all" else [args.step]

    # num_samples is required when generating pairs (step 1 or "all")
    if "1" in steps and not args.pairs_dir and args.num_samples is None:
        print("ERROR: num_samples is required for pair generation (step 1).")
        print("  Usage: generate_training_set.py 1300")
        print("  Or skip generation: generate_training_set.py --step 2 --pairs-dir output/pairs")
        sys.exit(1)

    pairs_dir = str(Path(args.output_dir).parent / "pairs")

    # Resolve scenarios for display
    scenarios = _load_scenarios(args)
    scenario_names = [s.name for s in scenarios]
    total_contrast_types = sum(len(s.contrast_types) for s in scenarios)

    print("=" * 60)
    print("  Contrastive SAE Pipeline")
    if args.num_samples is not None:
        print(f"  Requested samples : {args.num_samples:,}")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Pairs directory   : {args.pairs_dir or pairs_dir}")
    print(f"  Scenarios         : {', '.join(scenario_names)}")
    print(f"  Contrast types    : {total_contrast_types} total across {len(scenarios)} scenario(s)")
    print(f"  Steps to run      : {', '.join(steps)}")
    if "1" in steps:
        print(f"  Generation model  : {args.qwen_model} (vLLM, TP={args.generation_tp_size})")
    if "2" in steps:
        print(f"  Extraction model  : {args.nemotron_model}")
        print(f"  Layers            : {args.layers}")
        print(f"  Layer key         : {args.layer_key}")
        print(f"  GPU batch size    : {args.batch_size}")
        print(f"  Shard size        : {args.shard_size:,} vectors/shard")
    if "3" in steps:
        print(f"  SAE d_sae         : {args.d_sae}")
        print(f"  SAE batch size    : {args.sae_batch_size:,}")
        print(f"  SAE lr            : {args.sae_lr}")
        print(f"  SAE L1 coeff      : {args.l1_coefficient}")
    if "4" in steps:
        print(f"  SAE checkpoint    : {args.sae_checkpoint or '(auto from step 3)'}")
        print(f"  Top-K features    : {args.top_k_features}")
    if "5" in steps:
        print(f"  SAE checkpoint    : {args.sae_checkpoint or '(auto from step 3)'}")
        print(f"  Activations dir   : {args.output_dir}")
        print(f"  Label top-N       : {args.label_top_n}")
        print(f"  Label bottom-N    : {args.label_bottom_n}")
        print(f"  Labeling model    : {args.qwen_model} (vLLM, TP={args.generation_tp_size})")
    if "6" in steps:
        print(f"  SAE checkpoint    : {args.sae_checkpoint or '(auto from step 3)'}")
        print(f"  Fuzz top-K tokens : {args.fuzz_top_k_tokens}")
        print(f"  Fuzz examples/feat: {args.fuzz_examples_per_feature}")
        print(f"  Fuzz batch size   : {args.fuzz_batch_size}")
        print(f"  Judge model       : {args.qwen_model} (vLLM, TP={args.generation_tp_size})")
    print("=" * 60)

    sae_final_path = None

    if "1" in steps:
        _run_step1(args, pairs_dir)

    if "2" in steps:
        _run_step2(args, pairs_dir)

    if "3" in steps:
        sae_final_path = _run_step3(args)

    if "4" in steps:
        _run_step4(args, pairs_dir, sae_checkpoint=sae_final_path)

    if "5" in steps:
        _run_step5(args, sae_checkpoint=sae_final_path)

    if "6" in steps:
        _run_step6(args, pairs_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
