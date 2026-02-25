#!/usr/bin/env python3
"""
Layer sweep: run the SAE pipeline (steps 2-6) at multiple transformer layers
and compare feature quality across depths.

Addresses the reviewer concern: "You extract from layer 20 without explaining
why. Did you sweep across layers?"

This script:
  1. Reuses existing contrastive pairs (step 1 output)
  2. For each layer: extracts activations, trains an SAE, identifies
     contrastive features, interprets them, and runs fuzzing evaluation
  3. Produces a comparison report across all layers

Usage:
    uv run python layer_sweep.py \
        --pairs-dir output/pairs \
        --layers 8 16 20 32 44 \
        --base-output-dir output/layer_sweep \
        [--skip-steps 5 6]   # optionally skip expensive LLM steps

    # Minimal sweep (steps 2-4 only, no LLM needed):
    uv run python layer_sweep.py \
        --pairs-dir output/pairs \
        --layers 8 16 20 32 44 \
        --base-output-dir output/layer_sweep \
        --skip-steps 5 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer sweep for SAE pipeline")
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Transformer layers to sweep (e.g., 8 16 20 32 44).",
    )
    p.add_argument(
        "--pairs-dir",
        type=str,
        required=True,
        help="Path to existing contrastive pairs (shard_*.parquet).",
    )
    p.add_argument(
        "--base-output-dir",
        type=str,
        default="output/layer_sweep",
        help="Base directory for per-layer outputs (default: output/layer_sweep).",
    )
    p.add_argument(
        "--subject-model",
        type=str,
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        dest="subject_model",
        help="Subject model for activation extraction.",
    )
    p.add_argument(
        "--nemotron-model",
        type=str,
        default=argparse.SUPPRESS,
        dest="subject_model",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="LLM for labeling and judging (steps 5-6).",
    )
    p.add_argument(
        "--generation-tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM (default: 4).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Max sequence length for vLLM (default: 16384).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="GPU batch size for extraction (default: 512).",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Activation vectors per shard (default: 500000).",
    )
    p.add_argument(
        "--d-sae",
        type=int,
        default=16384,
        help="SAE hidden dimension (default: 16384).",
    )
    p.add_argument(
        "--sae-epochs",
        type=int,
        default=10,
        help="SAE training epochs (default: 10).",
    )
    p.add_argument(
        "--sae-batch-size",
        type=int,
        default=8192,
        help="SAE training batch size (default: 8192).",
    )
    p.add_argument(
        "--skip-steps",
        type=int,
        nargs="*",
        default=[],
        help="Steps to skip (e.g., --skip-steps 5 6 to skip LLM-dependent steps).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of pairs to use (default: all).",
    )
    return p.parse_args()


def run_layer(
    layer: int,
    args: argparse.Namespace,
    layer_output_dir: Path,
) -> dict:
    """Run pipeline steps 2-6 for a single layer.

    Returns a summary dict with feature health and fuzzing metrics (if available).
    """
    # Import generate_training_set functions
    from data.contrastive_dataset import ContrastiveDataset
    from data.scenario import load_scenarios_meta
    from pipeline import (
        extract_activations,
        identify_contrastive_features,
        train_sae_step,
    )

    layer_key = f"residual_{layer}"
    activations_dir = str(layer_output_dir / "activations")
    sae_checkpoint_dir = str(layer_output_dir / "sae_checkpoints")
    skip = set(args.skip_steps)

    # Load pairs — sample evenly across contrast types so that
    # --num-samples doesn't just grab the first (alphabetical) type.
    dataset = ContrastiveDataset.from_parquet(args.pairs_dir)
    if args.num_samples is not None:
        all_types = sorted({p.contrast_type for p in dataset.pairs})
        per_type = max(1, args.num_samples // len(all_types))
        sampled: list = []
        for ct in all_types:
            ct_pairs = [p for p in dataset.pairs if p.contrast_type == ct]
            sampled.extend(ct_pairs[:per_type])
        pairs = sampled[: args.num_samples]
    else:
        pairs = dataset.pairs

    scenarios_meta = load_scenarios_meta(Path(args.pairs_dir))

    summary = {"layer": layer, "layer_key": layer_key}

    # --- Step 2: Extract activations ---
    if 2 not in skip:
        print(f"\n  [Layer {layer}] Step 2: Extracting activations...")
        t0 = time.time()
        extract_activations(
            pairs=pairs,
            output_dir=activations_dir,
            subject_model=args.subject_model,
            layers=[layer],
            layer_key=layer_key,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            scenarios_meta=scenarios_meta,
        )
        summary["step2_time"] = round(time.time() - t0, 1)
    else:
        print(f"\n  [Layer {layer}] Step 2: Skipped")

    # --- Step 3: Train SAE ---
    if 3 not in skip:
        print(f"\n  [Layer {layer}] Step 3: Training SAE...")
        t0 = time.time()
        sae_path = train_sae_step(
            activations_dir=activations_dir,
            checkpoint_dir=sae_checkpoint_dir,
            d_sae=args.d_sae,
            learning_rate=3e-4,
            batch_size=args.sae_batch_size,
            l1_coefficient=5e-3,
            total_steps=None,
            num_epochs=args.sae_epochs,
            resume_from=None,
            auto_scale_steps=True,
        )
        summary["step3_time"] = round(time.time() - t0, 1)
        summary["sae_checkpoint"] = sae_path

        # Load feature health
        health_path = Path(sae_checkpoint_dir) / "feature_health.json"
        if health_path.exists():
            with open(health_path) as f:
                summary["feature_health"] = json.load(f)
    else:
        print(f"\n  [Layer {layer}] Step 3: Skipped")
        sae_path = str(Path(sae_checkpoint_dir) / "sae_final.pt")

    # --- Step 4: Contrastive features ---
    if 4 not in skip:
        print(f"\n  [Layer {layer}] Step 4: Identifying contrastive features...")
        t0 = time.time()
        identify_contrastive_features(
            pairs=pairs,
            sae_checkpoint=sae_path,
            subject_model=args.subject_model,
            layers=[layer],
            layer_key=layer_key,
            batch_size=args.batch_size,
            top_k=200,
            output_dir=activations_dir,
            scenarios_meta=scenarios_meta,
        )
        summary["step4_time"] = round(time.time() - t0, 1)

        cf_path = Path(activations_dir) / "contrastive_features.json"
        if cf_path.exists():
            with open(cf_path) as f:
                cf = json.load(f)
            s = cf.get("_summary", {})
            summary["contrastive_summary"] = {
                "unique_features": s.get("unique_features", 0),
                "dedup_ratio": s.get("dedup_ratio", 0),
            }
    else:
        print(f"\n  [Layer {layer}] Step 4: Skipped")

    # --- Step 5: Feature interpretation (requires LLM) ---
    if 5 not in skip:
        print(f"\n  [Layer {layer}] Step 5: Interpreting features...")
        t0 = time.time()

        from analysis.feature_interpreter import (
            collect_max_activating_examples,
            generate_explanation_report,
            label_features_via_llm,
            load_activations_from_shards,
        )

        cf_path = Path(activations_dir) / "contrastive_features.json"
        with open(cf_path) as f:
            contrastive = json.load(f)

        feature_indices = sorted(
            {
                feat["feature_index"]
                for ct_info in contrastive.values()
                for feat in ct_info.get("top_features", [])
            }
        )

        prompts, activations = load_activations_from_shards(activations_dir)
        feature_examples = collect_max_activating_examples(
            prompts=prompts,
            activations=activations,
            sae_checkpoint=sae_path,
            feature_indices=feature_indices,
        )
        del activations

        feature_labels = label_features_via_llm(
            feature_examples=feature_examples,
            qwen_model=args.qwen_model,
            tp_size=args.generation_tp_size,
            max_model_len=args.max_model_len,
            output_dir=activations_dir,
        )

        generate_explanation_report(
            contrastive_features_path=str(cf_path),
            feature_examples=feature_examples,
            feature_labels=feature_labels,
            output_dir=activations_dir,
        )
        summary["step5_time"] = round(time.time() - t0, 1)
    else:
        print(f"\n  [Layer {layer}] Step 5: Skipped")

    # --- Step 6: Fuzzing evaluation (requires LLM) ---
    if 6 not in skip:
        print(f"\n  [Layer {layer}] Step 6: Fuzzing evaluation...")
        t0 = time.time()

        from transformers import AutoTokenizer

        from analysis.fuzzing_evaluator import (
            build_fuzzing_examples,
            compute_fuzzing_metrics,
            evaluate_fuzzing,
            extract_per_token_activations,
            save_fuzzing_report,
        )
        from data.scenario import default_scenario
        from extraction.extractor import build_agent_prompt

        desc_path = Path(activations_dir) / "feature_descriptions.json"
        with open(desc_path) as f:
            feature_descriptions = json.load(f)

        _default_sc = default_scenario()
        prompt_to_scenario: dict[str, str] = {}
        for pair in pairs:
            prompt_to_scenario[pair.anchor_prompt] = pair.scenario_name
            prompt_to_scenario[pair.contrast_prompt] = pair.scenario_name

        all_prompts: list[str] = []
        seen: set[str] = set()
        for desc in feature_descriptions.values():
            for prompt in desc.get("top_examples", []) + desc.get("bottom_examples", []):
                if prompt not in seen:
                    seen.add(prompt)
                    all_prompts.append(prompt)

        tokenizer = AutoTokenizer.from_pretrained(args.subject_model, trust_remote_code=True)

        formatted_prompts = []
        for req in all_prompts:
            sc_name = prompt_to_scenario.get(req, "")
            sc = scenarios_meta.get(sc_name, _default_sc)
            formatted_prompts.append(
                build_agent_prompt(sc.system_prompt, sc.tools, req, tokenizer=tokenizer)
            )

        token_strings_list, token_activations_list = extract_per_token_activations(
            prompts=all_prompts,
            formatted_prompts=formatted_prompts,
            subject_model=args.subject_model,
            layers=[layer],
            layer_key=layer_key,
            batch_size=64,
        )

        prompt_to_idx = {p: i for i, p in enumerate(all_prompts)}

        examples = build_fuzzing_examples(
            feature_descriptions=feature_descriptions,
            prompt_to_idx=prompt_to_idx,
            token_strings_list=token_strings_list,
            token_activations_list=token_activations_list,
            sae_checkpoint=sae_path,
            tokenizer=tokenizer,
        )
        del token_strings_list, token_activations_list

        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = evaluate_fuzzing(
            examples=examples,
            feature_descriptions=feature_descriptions,
            qwen_model=args.qwen_model,
            tp_size=args.generation_tp_size,
            max_model_len=args.max_model_len,
            output_dir=activations_dir,
        )

        per_feature, fuzzing_summary = compute_fuzzing_metrics(results, feature_descriptions)
        save_fuzzing_report(per_feature, fuzzing_summary, results, activations_dir)
        summary["step6_time"] = round(time.time() - t0, 1)
        summary["fuzzing_summary"] = fuzzing_summary
    else:
        print(f"\n  [Layer {layer}] Step 6: Skipped")

    return summary


def build_comparison_report(summaries: list[dict], output_path: Path) -> None:
    """Build a comparison table across layers and save as JSON."""
    rows = []
    for s in summaries:
        row: dict = {"layer": s["layer"]}

        fh = s.get("feature_health", {})
        if fh:
            row["alive_pct"] = fh.get("alive_pct", None)
            row["dead_pct"] = fh.get("dead_pct", None)
            l0 = fh.get("l0", {})
            row["l0_mean"] = l0.get("mean", None)
            row["l0_sem"] = l0.get("sem", None)
            mse = fh.get("reconstruction_mse", {})
            row["recon_mse"] = mse.get("mean", None)

        cs = s.get("contrastive_summary", {})
        if cs:
            row["unique_contrastive_features"] = cs.get("unique_features", None)
            row["dedup_ratio"] = cs.get("dedup_ratio", None)

        fz = s.get("fuzzing_summary", {})
        if fz:
            combined = fz.get("combined_score", {})
            row["combined_score_mean"] = combined.get("mean", None)
            row["combined_score_sem"] = combined.get("sem", None)
            row["combined_score_p_value"] = combined.get("p_value_vs_baseline", None)
            ta = fz.get("token_level_accuracy", {})
            row["token_accuracy_mean"] = ta.get("mean", None)
            qt = fz.get("quality_tiers", {})
            row["excellent_pct"] = qt.get("excellent_above_0.8", {}).get("proportion", None)
            row["good_pct"] = qt.get("good_0.6_to_0.8", {}).get("proportion", None)
            row["poor_pct"] = qt.get("poor_below_0.6", {}).get("proportion", None)

        rows.append(row)

    report = {
        "layers_tested": [s["layer"] for s in summaries],
        "comparison": rows,
        "full_summaries": summaries,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("  Layer Sweep Comparison")
    print("=" * 90)

    # Header
    has_fuzzing = any("combined_score_mean" in r for r in rows)

    header = f"  {'Layer':>5}  {'Alive%':>7}  {'Dead%':>6}  {'L0':>8}  {'MSE':>10}  {'Features':>8}"
    if has_fuzzing:
        header += f"  {'Combined':>8}  {'p-value':>8}  {'Excl%':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in rows:
        line = (
            f"  {r['layer']:>5}"
            f"  {r.get('alive_pct', '-'):>7}"
            f"  {r.get('dead_pct', '-'):>6}"
            f"  {r.get('l0_mean', '-'):>8}"
            f"  {r.get('recon_mse', '-'):>10}"
            f"  {r.get('unique_contrastive_features', '-'):>8}"
        )
        if has_fuzzing:
            cs = r.get("combined_score_mean")
            pv = r.get("combined_score_p_value")
            ep = r.get("excellent_pct")
            line += f"  {cs if cs is not None else '-':>8}"
            line += f"  {pv if pv is not None else '-':>8}"
            line += f"  {ep if ep is not None else '-':>6}"
        print(line)

    print("=" * 90)

    # Identify best layer
    scored = [(r["layer"], r.get("combined_score_mean") or r.get("alive_pct", 0)) for r in rows]
    best_layer, best_score = max(scored, key=lambda x: x[1] if x[1] is not None else 0)
    print(f"  Best layer: {best_layer} (score: {best_score})")
    print(f"  Report saved: {output_path}")
    print("=" * 90)


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Layer Sweep")
    print(f"  Layers          : {args.layers}")
    print(f"  Pairs           : {args.pairs_dir}")
    print(f"  Output base     : {args.base_output_dir}")
    print(f"  Skip steps      : {args.skip_steps or 'none'}")
    print(f"  Subject model   : {args.subject_model}")
    print("=" * 70)

    summaries: list[dict] = []
    total_t0 = time.time()

    for layer in args.layers:
        layer_dir = base_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"  Layer {layer}")
        print(f"{'#' * 70}")

        t0 = time.time()
        summary = run_layer(layer, args, layer_dir)
        summary["total_time"] = round(time.time() - t0, 1)
        summaries.append(summary)

        # Save per-layer summary
        with open(layer_dir / "layer_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    total_elapsed = time.time() - total_t0
    print(f"\n  Total sweep time: {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)")

    # Build comparison
    report_path = base_dir / "layer_comparison.json"
    build_comparison_report(summaries, report_path)


if __name__ == "__main__":
    main()
