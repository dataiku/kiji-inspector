"""
Feature ablation experiment (causal intervention).

Tests whether SAE features are causally involved in tool-selection decisions
by zeroing them out during the forward pass and measuring whether the model's
tool prediction changes.

For each contrast type:
  1. Ablate top-K contrastive features -> measure flip rate + CATE (with bootstrap CI)
  2. Ablate K random features as control -> measure flip rate + CATE (with bootstrap CI)
  3. Fisher's exact test comparing the two
  4. One-sided Wilcoxon signed-rank test on probability shifts

Usage:
    python ablation.py \
        --sae-checkpoint output/sae/sae_final.pt \
        --contrastive-features output/activations/contrastive_features.json \
        --pairs-dir output/pairs/ \
        --output-dir ablation_output/ \
        [--n-features 10] [--n-prompts-per-type 100] [--layer 20] \
        [--model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kiji_inspector.utils.stats import clopper_pearson_ci as _clopper_pearson_ci

# ---------------------------------------------------------------------------
# Tool prediction
# ---------------------------------------------------------------------------


def build_tool_token_map(
    tokenizer,
    tools: list[dict[str, str]],
) -> tuple[dict[int, str], dict[str, int]]:
    """Map between tool names and their first token ID.

    The prompt ends with "I'll use the " so the next token starts with a
    space followed by the tool name.  We try multiple tokenization variants
    to handle different tokenizer behaviors.

    Returns:
        (token_id_to_tool, tool_to_token_id)
    """
    token_to_tool: dict[int, str] = {}
    tool_to_token: dict[str, int] = {}
    for tool in tools:
        name = tool["name"]
        # Try multiple variants: " name", "name", " Name"
        candidates = [f" {name}", name, f" {name.replace('_', ' ')}"]
        for variant in candidates:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids and ids[0] not in token_to_tool:
                token_to_tool[ids[0]] = name
                if name not in tool_to_token:
                    tool_to_token[name] = ids[0]
    return token_to_tool, tool_to_token


def get_tool_prediction(
    model,
    tokenizer,
    prompt: str,
    input_device: torch.device,
    token_to_tool: dict[int, str],
    contrast_tool: str | None = None,  # NEW
    top_k: int = 10,
) -> tuple[str, int, float]:
    """Run the full model and return (tool_name, token_id, prob_contrast_tool)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Find predicted tool (unchanged logic)
    topk_ids = logits.topk(top_k, dim=-1).indices[0].tolist()
    predicted_tool = "unknown"
    predicted_tid = topk_ids[0]
    for tid in topk_ids:
        if tid in token_to_tool:
            predicted_tool = token_to_tool[tid]
            predicted_tid = tid
            break

    # NEW: probability of the contrast tool
    prob_contrast = 0.0
    if contrast_tool is not None:
        for tid, name in token_to_tool.items():
            if name == contrast_tool:
                prob_contrast = float(probs[tid])
                break

    return predicted_tool, predicted_tid, prob_contrast


# ---------------------------------------------------------------------------
# Ablation hook
# ---------------------------------------------------------------------------


def make_ablation_hook(
    sae,
    feature_indices: list[int] | None = None,
    decision_token_only: bool = True,
):
    """Create a forward pre-hook that ablates specific SAE features.

    The hook intercepts the layer's *input* — the residual stream entering
    the layer — encodes it through the SAE, zeros out the specified
    features, decodes back, and returns the modified activation as the
    layer's new input. This must be the layer's input rather than its
    output: the SAE is trained on activations extracted via vLLM's
    aux-hidden-state mechanism, which records the residual entering layer N
    (see activation_extractor.py's ``_make_hook`` for the full convention).
    Hooking the layer's output would intervene one layer downstream of
    where the SAE's features actually live.

    When feature_indices is None (or empty), this becomes a reconstruction-only
    hook: the activation is encoded and decoded through the SAE without zeroing
    any features.  This measures the baseline distortion from the SAE round-trip.

    Args:
        sae: Trained JumpReLUSAE (already on the correct device).
        feature_indices: SAE feature indices to zero out, or None for
            reconstruction-only baseline.
        decision_token_only: If True, only patch the last (decision) token.
    """
    feat_set = set(feature_indices) if feature_indices else set()
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    def hook(module, args, kwargs):
        if args:
            hidden = args[0]
            rest_args = args[1:]
            hidden_from_kwargs = False
        else:
            hidden = kwargs["hidden_states"]
            rest_args = None
            hidden_from_kwargs = True

        orig_device = hidden.device
        orig_dtype = hidden.dtype

        with torch.no_grad():
            _rms = sae.rms_scale if (sae.rms_scale is not None and sae.rms_scale > 0) else None
            if decision_token_only:
                # Only modify the last token position
                last_tok = hidden[:, -1:, :]  # (1, 1, d_model)
                flat = last_tok.reshape(-1, last_tok.shape[-1]).to(
                    device=sae_device, dtype=sae_dtype
                )
                if _rms is not None:
                    flat = flat / _rms
                features = sae.encode(flat)
                for idx in feat_set:
                    features[:, idx] = 0.0
                modified = sae.decode(features)
                if _rms is not None:
                    modified = modified * _rms
                modified = modified.reshape(last_tok.shape).to(device=orig_device, dtype=orig_dtype)
                hidden = torch.cat([hidden[:, :-1, :], modified], dim=1)
            else:
                flat = hidden.reshape(-1, hidden.shape[-1]).to(device=sae_device, dtype=sae_dtype)
                if _rms is not None:
                    flat = flat / _rms
                features = sae.encode(flat)
                for idx in feat_set:
                    features[:, idx] = 0.0
                modified = sae.decode(features)
                if _rms is not None:
                    modified = modified * _rms
                hidden = modified.reshape(hidden.shape).to(device=orig_device, dtype=orig_dtype)

        if hidden_from_kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden
            return args, new_kwargs
        return (hidden,) + rest_args, kwargs

    return hook


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Metrics (Enhanced with Causal Inference + Wilcoxon + Aggregate Stats)
# ---------------------------------------------------------------------------


def compute_ablation_metrics(
    per_contrast: dict[str, dict],
) -> dict:
    """Compute aggregate metrics across contrast types, including causal CATE
    (Conditional Average Treatment Effect per contrast type), bootstrap CIs,
    and Wilcoxon signed-rank test statistics (both per contrast type and in aggregate).
    """
    import numpy as np
    from scipy.stats import fisher_exact, wilcoxon

    def bootstrap_ci(data: list[float], n_boot: int = 5000, alpha: float = 0.05):
        """Non-parametric bootstrap 95% CI.

        Returns:
        mean : original sample mean (point estimate)
        lower, upper : bootstrap percentile confidence interval."""
        if not data:
            return 0.0, 0.0, 0.0
        data = np.array(data)
        boot_means = np.array(
            [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
        )
        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
        mean = float(np.mean(data))
        return mean, lower, upper

    wilcoxon_p_values = []  # for aggregate statistics

    for _ct_key, info in per_contrast.items():
        n = info.get("n_tested", 0)
        if n == 0:
            continue

        c_flips = info["contrastive_flips"]
        c_directed = info["contrastive_directed_flips"]
        r_flips = info["random_flips"]
        recon_flips = info.get("reconstruction_flips", 0)

        # === Existing flip-rate metrics (backward compatible) ===
        info["contrastive_ablation"] = {
            "features_ablated": info["contrastive_feature_indices"],
            "n_features": len(info["contrastive_feature_indices"]),
            "flip_rate": round(c_flips / n, 4),
            "flip_rate_ci_95": [round(v, 4) for v in _clopper_pearson_ci(c_flips, n)],
            "directed_flip_rate": round(c_directed / n, 4),
            "directed_flip_rate_ci_95": [round(v, 4) for v in _clopper_pearson_ci(c_directed, n)],
        }
        info["random_ablation"] = {
            "n_features": info.get("n_random_features", 0),
            "flip_rate": round(r_flips / n, 4),
            "flip_rate_ci_95": [round(v, 4) for v in _clopper_pearson_ci(r_flips, n)],
        }
        info["reconstruction_baseline"] = {
            "flip_rate": round(recon_flips / n, 4),
            "flip_rate_ci_95": [round(v, 4) for v in _clopper_pearson_ci(recon_flips, n)],
        }

        # Fisher's exact test
        table = [[c_flips, n - c_flips], [r_flips, n - r_flips]]
        _, p_value = fisher_exact(table, alternative="greater")
        info["fisher_exact_p_value"] = round(float(p_value), 6)

        # === Causal metrics using probability deltas ===
        deltas = info.get("prob_deltas", {})
        contrastive_deltas = deltas.get("contrastive", [])

        if contrastive_deltas:
            cate, ci_lower, ci_upper = bootstrap_ci(contrastive_deltas)
            info["contrastive_ablation"].update(
                {
                    "CATE": round(cate, 4),
                    "CATE_ci_95_lower": round(ci_lower, 4),
                    "CATE_ci_95_upper": round(ci_upper, 4),
                }
            )

            # Wilcoxon signed-rank test (per contrast type)
            if np.count_nonzero(contrastive_deltas) >= 10:
                try:
                    _, wilcoxon_p = wilcoxon(
                        contrastive_deltas, alternative="greater", zero_method="wilcox"
                    )
                    wilcoxon_p = float(wilcoxon_p)
                except ValueError:
                    wilcoxon_p = None
            else:
                wilcoxon_p = None

            info["contrastive_ablation"]["wilcoxon_p_value"] = (
                round(wilcoxon_p, 6) if wilcoxon_p is not None else None
            )

            # Collect for aggregate
            if wilcoxon_p is not None:
                wilcoxon_p_values.append(wilcoxon_p)

        # Optional CATE for random / reconstruction
        if deltas.get("random"):
            random_cate, _, _ = bootstrap_ci(deltas["random"])
            info["random_ablation"]["CATE"] = round(random_cate, 4)
        if deltas.get("reconstruction"):
            recon_cate, _, _ = bootstrap_ci(deltas["reconstruction"])
            info["reconstruction_baseline"]["CATE"] = round(recon_cate, 4)

        # Clean up
        for key in (
            "contrastive_flips",
            "contrastive_directed_flips",
            "random_flips",
            "reconstruction_flips",
            "contrastive_feature_indices",
            "n_random_features",
        ):
            info.pop(key, None)

    # === Aggregate across all contrast types ===
    tested = [v for v in per_contrast.values() if v.get("n_tested", 0) > 0]
    if tested:
        agg = {
            "n_contrast_types": len(tested),
            "mean_contrastive_flip_rate": round(
                np.mean([v["contrastive_ablation"]["flip_rate"] for v in tested]), 4
            ),
            "mean_random_flip_rate": round(
                np.mean([v["random_ablation"]["flip_rate"] for v in tested]), 4
            ),
            "mean_directed_flip_rate": round(
                np.mean([v["contrastive_ablation"]["directed_flip_rate"] for v in tested]), 4
            ),
            "mean_reconstruction_flip_rate": round(
                np.mean([v["reconstruction_baseline"]["flip_rate"] for v in tested]), 4
            ),
            "mean_contrastive_CATE": round(
                np.mean([v["contrastive_ablation"].get("CATE", 0) for v in tested]), 4
            ),
        }

        # === NEW: Wilcoxon aggregate statistics ===
        if wilcoxon_p_values:
            agg["mean_wilcoxon_p_value"] = round(np.mean(wilcoxon_p_values), 6)
            significant_count = sum(1 for p in wilcoxon_p_values if p < 0.05)
            agg["wilcoxon_significant_count"] = significant_count
            agg["wilcoxon_significant_rate"] = round(significant_count / len(wilcoxon_p_values), 4)
        else:
            agg["mean_wilcoxon_p_value"] = None
            agg["wilcoxon_significant_count"] = 0
            agg["wilcoxon_significant_rate"] = 0.0

    else:
        agg = {}

    return {"per_contrast_type": per_contrast, "aggregate": agg}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_ablation_experiment(
    sae_checkpoint: str,
    contrastive_features_path: str,
    pairs_dir: str,
    output_dir: str = "ablation_output",
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    layer: int = 20,
    n_features: int = 10,
    n_prompts_per_type: int = 100,
    seed: int = 42,
) -> dict:
    """Run the full ablation experiment.

    Args:
        sae_checkpoint: Path to trained SAE checkpoint.
        contrastive_features_path: Path to contrastive_features.json from step 4.
        pairs_dir: Path to contrastive pairs (shard_*.parquet).
        output_dir: Where to save ablation_report.json.
        model_name: HuggingFace model name for the subject model.
        layer: Transformer layer where the SAE was trained.
        n_features: Number of top contrastive features to ablate.
        n_prompts_per_type: Max prompts to test per contrast type.
        seed: Random seed.
    """
    from kiji_inspector.core.sae_core import JumpReLUSAE
    from kiji_inspector.data.contrastive_dataset import ContrastiveDataset
    from kiji_inspector.data.scenario import load_scenarios_meta
    from kiji_inspector.extraction.activation_extractor import ActivationConfig, ActivationExtractor
    from kiji_inspector.extraction.extractor import build_agent_prompt

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load contrastive features
    with open(contrastive_features_path) as f:
        contrastive = json.load(f)

    # Load pairs
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    scenarios = load_scenarios_meta(pairs_dir)

    # Load SAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(sae_checkpoint, device=device)
    sae.eval()

    # Load subject model
    print(f"Loading subject model: {model_name}")
    config = ActivationConfig(
        model_name=model_name,
        layers=[layer],
        dtype=torch.bfloat16,
    )
    extractor = ActivationExtractor(config=config)
    model = extractor.model
    tokenizer = extractor.tokenizer
    input_device = extractor._input_device

    # Get the layer module for hook registration
    model_layers = extractor._get_model_layers()
    target_layer = model_layers[layer]

    # Build tool token maps per scenario
    scenario_tool_maps: dict[str, tuple[dict, dict]] = {}
    for name, sc in scenarios.items():
        maps = build_tool_token_map(tokenizer, sc.tools)
        scenario_tool_maps[name] = maps

    # Diagnostic: print tool token mappings
    _printed_scenarios: set[str] = set()
    for name, (t2t, _t2tok) in scenario_tool_maps.items():
        if name not in _printed_scenarios:
            _printed_scenarios.add(name)
            print(f"  Tool tokens for '{name}': {len(t2t)} mappings")
            for tid, tname in sorted(t2t.items(), key=lambda x: x[1]):
                decoded = tokenizer.decode([tid])
                print(f"    {tname:30s} -> token {tid:6d} ({decoded!r})")

    # Collect all unique SAE feature indices (for random control sampling)
    all_contrastive_indices: set[int] = set()
    for ct_key, ct_info in contrastive.items():
        if ct_key.startswith("_"):
            continue
        for feat in ct_info.get("top_features", []):
            all_contrastive_indices.add(feat["feature_index"])

    d_sae = sae.d_sae
    non_contrastive_indices = list(set(range(d_sae)) - all_contrastive_indices)

    print(f"\n{'=' * 70}")
    print("  Feature Ablation Experiment")
    print(f"  SAE checkpoint    : {sae_checkpoint}")
    print(f"  Layer             : {layer}")
    print(f"  Features to ablate: {n_features}")
    print(f"  Prompts per type  : {n_prompts_per_type}")
    print(f"  Contrast types    : {sum(1 for k in contrastive if not k.startswith('_'))}")
    print(f"{'=' * 70}\n")

    per_contrast: dict[str, dict] = {}

    from kiji_inspector.data.scenario import default_scenario

    default_sc = default_scenario()

    for ct_key, ct_info in contrastive.items():
        if ct_key.startswith("_"):
            continue

        top_features = ct_info.get("top_features", [])
        if len(top_features) < n_features:
            print(f"  Skipping {ct_key}: only {len(top_features)} features (need {n_features})")
            continue

        contrastive_indices = [f["feature_index"] for f in top_features[:n_features]]
        random_indices = random.sample(
            non_contrastive_indices, min(n_features, len(non_contrastive_indices))
        )

        # Get pairs for this contrast type
        ct_pairs = dataset.get_by_contrast_type(ct_key)
        if not ct_pairs:
            print(f"  Skipping {ct_key}: no pairs found")
            continue
        random.shuffle(ct_pairs)
        ct_pairs = ct_pairs[:n_prompts_per_type]

        print(
            f"  {ct_key}: testing {len(ct_pairs)} pairs, "
            f"ablating features {contrastive_indices[:3]}..."
        )

        # === NEW: collect probability deltas for causal metrics ===
        prob_deltas = {
            "contrastive": [],
            "random": [],
            "reconstruction": [],
        }

        n_tested = 0
        n_baseline_mismatches = 0
        n_unknown_baseline = 0
        contrastive_flips = 0
        contrastive_directed_flips = 0
        random_flips = 0
        reconstruction_flips = 0

        for pair in tqdm(ct_pairs, desc=f"    {ct_key}", unit="pair", leave=False):
            sc = scenarios.get(pair.scenario_name, default_sc)
            token_to_tool, tool_to_token = scenario_tool_maps.get(
                pair.scenario_name,
                build_tool_token_map(tokenizer, sc.tools),
            )

            prompt = build_agent_prompt(
                system_prompt=sc.system_prompt,
                tools=sc.tools,
                user_request=pair.anchor_prompt,
                tokenizer=tokenizer,
            )

            expected_tool = pair.anchor_tool.split(",")[0].strip()
            contrast_tool = pair.contrast_tool.split(",")[0].strip()

            # Baseline (no ablation)
            baseline_tool, baseline_tid, baseline_prob_contrast = get_tool_prediction(
                model, tokenizer, prompt, input_device, token_to_tool, contrast_tool=contrast_tool
            )

            if baseline_tool == "unknown":
                n_unknown_baseline += 1
                continue
            if baseline_tool != expected_tool:
                n_baseline_mismatches += 1
                continue

            n_tested += 1

            # Reconstruction-only baseline
            hook_handle = target_layer.register_forward_pre_hook(
                make_ablation_hook(sae, feature_indices=None), with_kwargs=True
            )
            recon_tool, recon_tid, recon_prob_contrast = get_tool_prediction(
                model, tokenizer, prompt, input_device, token_to_tool, contrast_tool=contrast_tool
            )
            hook_handle.remove()
            if recon_tid != baseline_tid:
                reconstruction_flips += 1

            # Contrastive ablation
            hook_handle = target_layer.register_forward_pre_hook(
                make_ablation_hook(sae, contrastive_indices), with_kwargs=True
            )
            ablated_tool, ablated_tid, ablated_prob_contrast = get_tool_prediction(
                model, tokenizer, prompt, input_device, token_to_tool, contrast_tool=contrast_tool
            )
            hook_handle.remove()
            if ablated_tid != baseline_tid:
                contrastive_flips += 1
                if ablated_tool == contrast_tool:
                    contrastive_directed_flips += 1

            # Random ablation
            hook_handle = target_layer.register_forward_pre_hook(
                make_ablation_hook(sae, random_indices), with_kwargs=True
            )
            rand_tool, rand_tid, rand_prob_contrast = get_tool_prediction(
                model, tokenizer, prompt, input_device, token_to_tool, contrast_tool=contrast_tool
            )
            hook_handle.remove()
            if rand_tid != baseline_tid:
                random_flips += 1

            # === NEW: store probability shifts ===
            prob_deltas["contrastive"].append(ablated_prob_contrast - baseline_prob_contrast)
            prob_deltas["random"].append(rand_prob_contrast - baseline_prob_contrast)
            prob_deltas["reconstruction"].append(recon_prob_contrast - baseline_prob_contrast)

        # Store results (including new deltas)
        per_contrast[ct_key] = {
            "n_pairs_available": len(ct_pairs),
            "n_tested": n_tested,
            "contrastive_flips": contrastive_flips,
            "contrastive_directed_flips": contrastive_directed_flips,
            "random_flips": random_flips,
            "reconstruction_flips": reconstruction_flips,
            "contrastive_feature_indices": contrastive_indices,
            "n_random_features": len(random_indices),
            "prob_deltas": prob_deltas,  # ← this is the key new data
        }

        # ... (the rest of the diagnostics print remains unchanged)

        # Diagnostics
        total_tried = n_tested + n_baseline_mismatches + n_unknown_baseline
        print(
            f"    Baseline filter: {total_tried} pairs tried, "
            f"{n_tested} correct ({100 * n_tested / max(1, total_tried):.0f}%), "
            f"{n_baseline_mismatches} wrong tool, "
            f"{n_unknown_baseline} unknown token"
        )

        if n_tested > 0:
            print(
                f"    Results (n={n_tested}): "
                f"recon_only={reconstruction_flips}/{n_tested} "
                f"({100 * reconstruction_flips / n_tested:.1f}%), "
                f"contrastive={contrastive_flips}/{n_tested} "
                f"({100 * contrastive_flips / n_tested:.1f}%), "
                f"directed={contrastive_directed_flips}/{n_tested} "
                f"({100 * contrastive_directed_flips / n_tested:.1f}%), "
                f"random={random_flips}/{n_tested} "
                f"({100 * random_flips / n_tested:.1f}%)"
            )
        else:
            print("    No prompts with correct baseline prediction")

    # Compute metrics and save
    report = compute_ablation_metrics(per_contrast)

    report_path = output_dir / "ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    agg = report.get("aggregate", {})
    print(f"\n{'=' * 70}")
    print("  Ablation Results")
    if agg:
        print(f"  Contrast types tested : {agg['n_contrast_types']}")
        print(f"  Mean recon-only flip  : {agg.get('mean_reconstruction_flip_rate', 0):.1%}")
        print(f"  Mean contrastive flip : {agg['mean_contrastive_flip_rate']:.1%}")
        print(f"  Mean directed flip    : {agg['mean_directed_flip_rate']:.1%}")
        print(f"  Mean random flip      : {agg['mean_random_flip_rate']:.1%}")
        recon_rate = agg.get("mean_reconstruction_flip_rate", 0)
        if recon_rate > 0.3:
            print(f"\n  WARNING: Reconstruction-only flip rate is {recon_rate:.1%}.")
            print("  The SAE round-trip itself disrupts predictions significantly.")
            print("  Ablation results should be interpreted relative to this baseline.")
    print(f"  Report saved          : {report_path}")
    print(f"{'=' * 70}")

    # Cleanup
    extractor.cleanup()

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature ablation experiment")
    parser.add_argument("--sae-checkpoint", required=True, help="Path to SAE checkpoint")
    parser.add_argument(
        "--contrastive-features", required=True, help="Path to contrastive_features.json"
    )
    parser.add_argument("--pairs-dir", required=True, help="Path to contrastive pairs directory")
    parser.add_argument("--output-dir", default="ablation_output", help="Output directory")
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="Subject model name",
    )
    parser.add_argument("--layer", type=int, default=20, help="Transformer layer for ablation")
    parser.add_argument("--n-features", type=int, default=10, help="Number of features to ablate")
    parser.add_argument(
        "--n-prompts-per-type", type=int, default=100, help="Max prompts per contrast type"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_ablation_experiment(
        sae_checkpoint=args.sae_checkpoint,
        contrastive_features_path=args.contrastive_features,
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        layer=args.layer,
        n_features=args.n_features,
        n_prompts_per_type=args.n_prompts_per_type,
        seed=args.seed,
    )
