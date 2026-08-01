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

    The prompt ends with "I'll use the" (no trailing space), so the next token
    is the space-prefixed tool name — "▁ticket" under SentencePiece,
    "Ġticket" under BPE. The " {name}" variant below is therefore the one that
    normally matches; the others are fallbacks for tokenizers that split
    differently.

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
# Random-control sampling
# ---------------------------------------------------------------------------

# Log-spaced firing-rate bins for frequency matching. A feature's bin is
# determined by how often it fires on the training corpus; the matched
# control draws a feature from the same bin as each contrastive feature.
FIRING_RATE_BIN_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.01]

# Firing rate above which a feature counts as "alive" — same threshold the
# trainer's analyze_feature_health uses.
ALIVE_RATE_THRESHOLD = 1e-3


def compute_firing_rates(
    sae,
    activations_dir: str | Path,
    target_rows: int = 1_600_000,
    chunk_size: int = 8192,
    device: str = "cuda",
) -> np.ndarray:
    """Compute per-feature firing rates by encoding cached activations.

    The ablation's random control must be frequency-matched to the
    contrastive features being ablated — a uniform draw over the dictionary
    lands overwhelmingly on features that never fire (50-80% of a trained
    dictionary is near-silent), making the control a no-op. That requires
    per-feature firing rates, which the trainer computes for
    feature_health.json but historically did not persist; this recomputes
    them the same way, from the step 1 activation shards already on disk.

    Chunks are sampled at evenly spaced offsets across each shard (rows are
    written in prompt order, so reading only shard heads would bias the
    sample toward early prompts).

    Args:
        sae: Loaded JumpReLUSAE (provides encode + normalize_input).
        activations_dir: Directory containing shard_*.npy files.
        target_rows: Total rows to sample across all shards.
        chunk_size: Rows per encoded batch.
        device: Device to run the SAE on.

    Returns:
        Array of shape (d_sae,) with each feature's firing rate in [0, 1].
    """
    shard_files = sorted(Path(activations_dir).glob("shard_*.npy"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.npy files in {activations_dir}")

    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    rows_per_shard = max(chunk_size, target_rows // len(shard_files))
    fire_count = torch.zeros(sae.d_sae, device=sae_device)
    total = 0

    for path in shard_files:
        arr = np.load(path, mmap_mode="r")
        n_rows = arr.shape[0]
        if n_rows < chunk_size:
            starts = [0]
        else:
            n_chunks = max(1, rows_per_shard // chunk_size)
            starts = np.linspace(0, n_rows - chunk_size, n_chunks).astype(int).tolist()
        for s in starts:
            block = np.asarray(arr[s : s + chunk_size], dtype=np.float32)
            block = block[np.isfinite(block).all(axis=1)]
            if block.shape[0] == 0:
                continue
            x = torch.from_numpy(block).to(device=sae_device, dtype=sae_dtype)
            with torch.inference_mode():
                feats = sae.encode(sae.normalize_input(x))
                fire_count += (feats.abs() > 0).float().sum(dim=0)
            total += x.shape[0]
        del arr

    return (fire_count / max(total, 1)).cpu().numpy()


def compute_type_conditional_rates(
    sae,
    memmaps: list[np.memmap],
    cum_offsets: np.ndarray,
    pair_indices: np.ndarray,
    max_rows: int = 400,
    chunk_size: int = 512,
) -> np.ndarray:
    """Per-feature firing rates conditioned on one contrast type's prompts.

    Corpus-level firing rates are the wrong yardstick for the random
    control: a feature firing on 5% of corpus tokens usually does not fire
    on any particular prompt, while the contrastive features under test fire
    on nearly all of their type's prompts by construction. Matching on
    corpus rates therefore still leaves the control a near-no-op (measured:
    ~87% of prompts showed random deltas identical to the reconstruction
    baseline). This computes rates over the type's own anchor rows (row
    2*i for pair index i, the step 1 shard layout), so the control can be
    matched on activity where it matters.

    Args:
        sae: Loaded JumpReLUSAE.
        memmaps / cum_offsets: From shard_io.open_layer_shards.
        pair_indices: Global pair indices for this contrast type.
        max_rows: Cap on rows sampled (evenly spaced across the type).
        chunk_size: Rows per encode batch.

    Returns:
        Array (d_sae,) of firing fractions on this type's anchor prompts.
    """
    from kiji_inspector.analysis.shard_io import gather_rows

    idx = np.asarray(pair_indices, dtype=np.int64)
    if len(idx) > max_rows:
        idx = idx[np.linspace(0, len(idx) - 1, max_rows).astype(int)]
    anchor_rows = 2 * idx

    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    fire_count = torch.zeros(sae.d_sae, device=sae_device)
    total = 0
    for s in range(0, len(anchor_rows), chunk_size):
        block = gather_rows(memmaps, cum_offsets, anchor_rows[s : s + chunk_size]).astype(
            np.float32
        )
        block = block[np.isfinite(block).all(axis=1)]
        if block.shape[0] == 0:
            continue
        x = torch.from_numpy(block).to(device=sae_device, dtype=sae_dtype)
        with torch.inference_mode():
            feats = sae.encode(sae.normalize_input(x))
            fire_count += (feats.abs() > 0).float().sum(dim=0)
        total += x.shape[0]

    return (fire_count / max(total, 1)).cpu().numpy()


def sample_frequency_matched(
    rng: random.Random,
    contrastive_indices: list[int],
    firing_rates: np.ndarray,
    excluded: set[int],
) -> list[int]:
    """Draw control features frequency-matched to the contrastive set.

    For each contrastive feature, picks an unused, non-excluded feature from
    the same firing-rate bin (FIRING_RATE_BIN_EDGES), walking outward to the
    nearest non-empty bin when the target bin is exhausted. This makes the
    "random ablation" arm intervene on features comparably active to the
    ones under test, instead of near-silent ones.

    Args:
        rng: Dedicated random.Random instance (reproducible, isolated from
            other random consumers).
        contrastive_indices: The features being ablated in the treatment arm.
        firing_rates: Per-feature firing rates from compute_firing_rates.
        excluded: Feature indices never eligible as controls (typically the
            union of all contrastive features across contrast types).

    Returns:
        List of matched control indices, same length as contrastive_indices
        (shorter only if the entire eligible pool is smaller than that).
    """
    n_bins = len(FIRING_RATE_BIN_EDGES) - 1
    bin_of = np.digitize(firing_rates, FIRING_RATE_BIN_EDGES) - 1
    bin_of = np.clip(bin_of, 0, n_bins - 1)

    pool_by_bin: dict[int, list[int]] = {
        b: [int(i) for i in np.where(bin_of == b)[0] if int(i) not in excluded]
        for b in range(n_bins)
    }

    matched: list[int] = []
    used: set[int] = set()
    for ci in contrastive_indices:
        target_bin = int(bin_of[ci])
        # Walk outward: 0, +1, -1, +2, -2, ...
        for delta in [0] + [d for k in range(1, n_bins) for d in (k, -k)]:
            b = target_bin + delta
            if b < 0 or b >= n_bins:
                continue
            candidates = [i for i in pool_by_bin[b] if i not in used]
            if candidates:
                pick = candidates[rng.randrange(len(candidates))]
                matched.append(pick)
                used.add(pick)
                break
    return matched


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
            # normalize_input/denormalize_output apply the full transform the
            # SAE was trained under — (x - mean_vec) / rms_scale and its
            # inverse. Scaling by rms_scale alone leaves the activation offset
            # in place, which for a residual stream dwarfs the signal.
            if decision_token_only:
                # Only modify the last token position
                last_tok = hidden[:, -1:, :]  # (1, 1, d_model)
                flat = last_tok.reshape(-1, last_tok.shape[-1]).to(
                    device=sae_device, dtype=sae_dtype
                )
                features = sae.encode(sae.normalize_input(flat))
                for idx in feat_set:
                    features[:, idx] = 0.0
                modified = sae.denormalize_output(sae.decode(features))
                modified = modified.reshape(last_tok.shape).to(device=orig_device, dtype=orig_dtype)
                hidden = torch.cat([hidden[:, :-1, :], modified], dim=1)
            else:
                flat = hidden.reshape(-1, hidden.shape[-1]).to(device=sae_device, dtype=sae_dtype)
                features = sae.encode(sae.normalize_input(flat))
                for idx in feat_set:
                    features[:, idx] = 0.0
                modified = sae.denormalize_output(sae.decode(features))
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


def _wilcoxon_greater(deltas: list[float], min_nonzero: int = 10) -> float | None:
    """One-sided Wilcoxon signed-rank p-value, or None when undertested.

    Returns None when fewer than min_nonzero deltas are nonzero (the test
    discards zeros via zero_method="wilcox", so it has no power below that)
    or when scipy rejects the input. Callers must count None results in
    their denominators explicitly — silently dropping them inflates any
    significance rate.
    """
    from scipy.stats import wilcoxon

    if np.count_nonzero(deltas) < min_nonzero:
        return None
    try:
        _, p = wilcoxon(deltas, alternative="greater", zero_method="wilcox")
        return float(p)
    except ValueError:
        return None


def compute_ablation_metrics(
    per_contrast: dict[str, dict],
) -> dict:
    """Compute aggregate metrics across contrast types, including causal CATE
    (Conditional Average Treatment Effect per contrast type), bootstrap CIs,
    and Wilcoxon signed-rank test statistics (both per contrast type and in
    aggregate, with Benjamini-Hochberg correction across contrast types).
    """
    import numpy as np
    from scipy.stats import combine_pvalues, false_discovery_control, fisher_exact

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

    # (ct_key, p) for BH correction; excluded types tracked so significance
    # denominators can count them instead of silently dropping them.
    wilcoxon_tested: list[tuple[str, float]] = []
    wilcoxon_excluded: list[str] = []

    for ct_key, info in per_contrast.items():
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

            wilcoxon_p = _wilcoxon_greater(contrastive_deltas)
            info["contrastive_ablation"]["wilcoxon_p_value"] = (
                round(wilcoxon_p, 6) if wilcoxon_p is not None else None
            )
            if wilcoxon_p is not None:
                wilcoxon_tested.append((ct_key, wilcoxon_p))
            else:
                wilcoxon_excluded.append(ct_key)
        else:
            wilcoxon_excluded.append(ct_key)

        # Random / reconstruction arms: CATE plus the same Wilcoxon test.
        # Under a healthy frequency-matched control these p-values should be
        # unremarkable; a control arm that comes out "significant" is itself
        # a red flag worth surfacing.
        if deltas.get("random"):
            random_cate, _, _ = bootstrap_ci(deltas["random"])
            info["random_ablation"]["CATE"] = round(random_cate, 4)
            p = _wilcoxon_greater(deltas["random"])
            info["random_ablation"]["wilcoxon_p_value"] = (
                round(p, 6) if p is not None else None
            )
        if deltas.get("reconstruction"):
            recon_cate, _, _ = bootstrap_ci(deltas["reconstruction"])
            info["reconstruction_baseline"]["CATE"] = round(recon_cate, 4)
            p = _wilcoxon_greater(deltas["reconstruction"])
            info["reconstruction_baseline"]["wilcoxon_p_value"] = (
                round(p, 6) if p is not None else None
            )

        # Preserve raw counts (previously popped, which made reports
        # impossible to re-analyze post hoc).
        info["raw_counts"] = {
            key: info.pop(key)
            for key in (
                "contrastive_flips",
                "contrastive_directed_flips",
                "random_flips",
                "reconstruction_flips",
                "contrastive_feature_indices",
                "n_random_features",
            )
            if key in info
        }

    # Benjamini-Hochberg correction across contrast types (one family per
    # report). Uncorrected p-values are kept alongside for comparison.
    adjusted_ps: list[float] = []
    if wilcoxon_tested:
        adjusted_ps = [
            float(p)
            for p in false_discovery_control([p for _, p in wilcoxon_tested], method="bh")
        ]
        for (ct_key, _), p_adj in zip(wilcoxon_tested, adjusted_ps, strict=True):
            per_contrast[ct_key]["contrastive_ablation"]["wilcoxon_p_value_bh"] = round(p_adj, 6)

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

        # Wilcoxon aggregates. Two significance rates are reported: over
        # tested types only, and over all types with n_tested > 0 (excluded
        # types counted as not-significant) — the first is what a naive
        # reading produces, the second is the honest denominator. BH-adjusted
        # counts control the false-discovery rate across the type family.
        # Fisher's method replaces the former mean-of-p-values (which has no
        # inferential meaning); it assumes independent tests, which contrast
        # types only approximately satisfy.
        n_tested_types = len(wilcoxon_tested)
        n_excluded_types = len(wilcoxon_excluded)
        agg["wilcoxon_tested_types"] = n_tested_types
        agg["wilcoxon_excluded_types"] = sorted(wilcoxon_excluded)
        if wilcoxon_tested:
            raw_ps = [p for _, p in wilcoxon_tested]
            sig_raw = sum(1 for p in raw_ps if p < 0.05)
            sig_bh = sum(1 for p in adjusted_ps if p < 0.05)
            denom_all = n_tested_types + n_excluded_types
            agg["wilcoxon_significant_count"] = sig_raw
            agg["wilcoxon_significant_count_bh"] = sig_bh
            agg["wilcoxon_significant_rate_tested"] = round(sig_raw / n_tested_types, 4)
            agg["wilcoxon_significant_rate_all"] = round(sig_raw / denom_all, 4)
            agg["wilcoxon_significant_rate_bh_tested"] = round(sig_bh / n_tested_types, 4)
            agg["wilcoxon_significant_rate_bh_all"] = round(sig_bh / denom_all, 4)
            _, fisher_p = combine_pvalues(raw_ps, method="fisher")
            agg["fisher_combined_p_value"] = float(fisher_p)
        else:
            agg["wilcoxon_significant_count"] = 0
            agg["wilcoxon_significant_count_bh"] = 0
            agg["wilcoxon_significant_rate_tested"] = 0.0
            agg["wilcoxon_significant_rate_all"] = 0.0
            agg["wilcoxon_significant_rate_bh_tested"] = 0.0
            agg["wilcoxon_significant_rate_bh_all"] = 0.0
            agg["fisher_combined_p_value"] = None

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
    min_baseline_pass_rate: float = 0.2,
    random_control: str = "frequency-matched",
    activations_dir: str | None = None,
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
        min_baseline_pass_rate: Abort if the fraction of prompts whose
            unablated prediction matches the expected tool falls below this.
            Guards against publishing a report built on a handful of prompts.
            Set to 0 to disable.
        random_control: How the random-control features are drawn.
            "frequency-matched" (default) matches each contrastive feature's
            firing-rate bin, redrawn per prompt; "alive" draws uniformly from
            features firing on >0.1% of inputs; "legacy" reproduces the old
            behavior (uniform over the whole dictionary, one draw per
            contrast type) — known to be a near-no-op on sparse dictionaries.
        activations_dir: Directory with step 1 shard_*.npy files, used to
            compute firing rates when no persisted firing_rates.npy exists
            next to the SAE checkpoint. Defaults to the directory containing
            contrastive_features_path.
    """
    from kiji_inspector.core.sae_core import JumpReLUSAE
    from kiji_inspector.data.contrastive_dataset import ContrastiveDataset
    from kiji_inspector.data.scenario import load_scenarios_meta
    from kiji_inspector.extraction.activation_extractor import ActivationConfig, ActivationExtractor
    from kiji_inspector.extraction.extractor import build_agent_prompt
    from kiji_inspector.extraction.vllm_activation_extractor import recommended_chat_template_kwargs

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load contrastive features
    with open(contrastive_features_path) as f:
        contrastive = json.load(f)

    # Load pairs. Excludes anchor_tool == contrast_tool pairs, matching
    # pipeline.py's _load_pairs() (used by steps 1/3/4/5) -- for those pairs
    # the expected tool is identical either way, so "did ablation flip toward
    # the contrast tool" is degenerate. Step 3's feature selection never saw
    # these (its pair count is validated against step 1's filtered shards),
    # so leaving them in here would test features against noise the features
    # weren't selected to explain.
    dataset = ContrastiveDataset.from_parquet(pairs_dir)
    total_pairs = len(dataset.pairs)
    dataset.pairs = [p for p in dataset.pairs if p.anchor_tool != p.contrast_tool]
    excluded_pairs = total_pairs - len(dataset.pairs)
    print(f"  Loaded {len(dataset.pairs)} pairs from {pairs_dir}")
    if excluded_pairs:
        print(f"  Excluded {excluded_pairs} pairs where anchor_tool == contrast_tool")
    scenarios = load_scenarios_meta(pairs_dir)

    # Load SAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(sae_checkpoint, device=device)
    sae.eval()

    # Firing rates for the random control. Without them, a uniform draw over
    # the dictionary lands overwhelmingly on near-silent features and the
    # "random ablation" arm degenerates into the reconstruction baseline.
    firing_rates: np.ndarray | None = None
    acts_dir = activations_dir or str(Path(contrastive_features_path).parent)
    if random_control != "legacy":
        rates_path = Path(sae_checkpoint).parent / "firing_rates.npy"
        if rates_path.exists():
            firing_rates = np.load(rates_path)
            print(f"  Loaded firing rates from {rates_path}")
        else:
            print(f"  Computing firing rates from {acts_dir} (no persisted firing_rates.npy)...")
            firing_rates = compute_firing_rates(sae, acts_dir, device=device)
        n_alive = int((firing_rates > ALIVE_RATE_THRESHOLD).sum())
        n_dead = int((firing_rates == 0).sum())
        n_ultra_rare = int(((firing_rates > 0) & (firing_rates < 1e-4)).sum())
        print(
            f"  Firing rates: {n_alive} alive (> {ALIVE_RATE_THRESHOLD:g}), "
            f"{n_ultra_rare} ultra-rare, {n_dead} dead of {len(firing_rates)}"
        )

    # For frequency matching, corpus rates are not enough: the contrastive
    # features fire on nearly all of their own type's prompts, while a
    # corpus-rate-matched feature usually fires on none of them. Matching on
    # rates conditioned on each type's own prompts needs the step 1 shards
    # (row 2*i is pair i's anchor — the same layout step 3 validates).
    type_rate_memmaps = None
    type_rate_offsets = None
    pair_indices_by_type: dict[str, np.ndarray] = {}
    if random_control == "frequency-matched":
        from kiji_inspector.analysis.shard_io import open_layer_shards

        try:
            type_rate_memmaps, type_rate_offsets = open_layer_shards(Path(acts_dir))
            total_rows = int(type_rate_offsets[-1])
            if total_rows != 2 * len(dataset.pairs):
                print(
                    f"  WARNING: shard rows ({total_rows}) != 2x pairs "
                    f"({2 * len(dataset.pairs)}); falling back to corpus-rate matching."
                )
                type_rate_memmaps = None
            else:
                by_type: dict[str, list[int]] = {}
                for i, pair in enumerate(dataset.pairs):
                    by_type.setdefault(pair.contrast_type, []).append(i)
                pair_indices_by_type = {
                    ct: np.asarray(v, dtype=np.int64) for ct, v in by_type.items()
                }
        except FileNotFoundError:
            print(f"  WARNING: no shards in {acts_dir}; falling back to corpus-rate matching.")

    # Dedicated RNG for control draws: reproducible and isolated from the
    # global random module (which also shuffles pairs).
    control_rng = random.Random(seed)

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

    # Reasoning models (e.g. Nemotron-3-Nano) open a <think> block in their
    # generation prompt by default, which would put the decision token inside
    # the reasoning channel instead of at the tool-name position -- the same
    # bug fixed for step 1 extraction, but ablation builds its own prompts
    # here and needs the same suppression applied.
    _tpl_kwargs = recommended_chat_template_kwargs(model_name, tokenizer) or None

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

        # Control features. "frequency-matched" redraws per prompt (below);
        # "alive" and "legacy" keep one draw per contrast type.
        match_rates = firing_rates
        matched_on = "corpus" if firing_rates is not None else None
        if random_control == "frequency-matched":
            # Prefer rates conditioned on this type's own prompts (the
            # honest null: features active *here* but not contrastively
            # selected); corpus rates only as fallback.
            if type_rate_memmaps is not None and ct_key in pair_indices_by_type:
                match_rates = compute_type_conditional_rates(
                    sae, type_rate_memmaps, type_rate_offsets, pair_indices_by_type[ct_key]
                )
                matched_on = "type-conditional"
            random_indices = sample_frequency_matched(
                control_rng, contrastive_indices, match_rates, all_contrastive_indices
            )
        elif random_control == "alive":
            alive_pool = [
                i
                for i in np.where(firing_rates > ALIVE_RATE_THRESHOLD)[0].tolist()
                if i not in all_contrastive_indices
            ]
            random_indices = control_rng.sample(alive_pool, min(n_features, len(alive_pool)))
        else:  # legacy
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
                chat_template_kwargs=_tpl_kwargs,
                close_think_block=bool(_tpl_kwargs),
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

            # Random ablation. Frequency-matched mode redraws per prompt so
            # the control is an average over many draws rather than hostage
            # to one; zero extra forward passes (this arm already runs one).
            if random_control == "frequency-matched":
                random_indices = sample_frequency_matched(
                    control_rng, contrastive_indices, match_rates, all_contrastive_indices
                )
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
            # Recorded so the baseline-pass-rate guard below (and anyone reading
            # the report) can tell "the features did nothing" apart from "almost
            # no prompt reached the intervention".
            "n_baseline_mismatches": n_baseline_mismatches,
            "n_unknown_baseline": n_unknown_baseline,
            "contrastive_flips": contrastive_flips,
            "contrastive_directed_flips": contrastive_directed_flips,
            "random_flips": random_flips,
            "reconstruction_flips": reconstruction_flips,
            "contrastive_feature_indices": contrastive_indices,
            "n_random_features": len(random_indices),
            "prob_deltas": prob_deltas,  # ← this is the key new data
            "random_control": {
                "mode": random_control,
                "matched_on": matched_on,
                "n_features": len(random_indices),
                "firing_rate_bins_of_contrastive": (
                    [
                        int(np.clip(b - 1, 0, len(FIRING_RATE_BIN_EDGES) - 2))
                        for b in np.digitize(
                            match_rates[contrastive_indices], FIRING_RATE_BIN_EDGES
                        )
                    ]
                    if match_rates is not None
                    else None
                ),
            },
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

    # Refuse to publish a report the baseline filter has already invalidated.
    # A prompt the model does not answer with a tool name yields n_tested ~= 0
    # per contrast type, and every downstream statistic then degenerates
    # silently: CATE and directed-flip come out as exactly 0.0, Wilcoxon p as
    # None, and flip-rate CIs as [0, 0.975] -- a report that looks complete but
    # measures nothing. The usual cause is a prompt whose decision token is not
    # a tool name (see _DEFAULT_ASSISTANT_PREFILL in extraction/extractor.py),
    # or a --model that does not match the SAE's subject model.
    kept = sum(v.get("n_tested", 0) for v in per_contrast.values())
    tried = sum(
        v.get("n_tested", 0) + v.get("n_baseline_mismatches", 0) + v.get("n_unknown_baseline", 0)
        for v in per_contrast.values()
    )
    if tried and kept / tried < min_baseline_pass_rate:
        raise RuntimeError(
            f"Baseline filter kept only {kept}/{tried} prompts "
            f"({100 * kept / tried:.1f}%), below the {100 * min_baseline_pass_rate:.0f}% "
            f"minimum — the model is not predicting tool names at the decision "
            f"token, so ablation would measure nothing. Check that --model "
            f"({model_name}) matches the SAE's subject model and that the "
            f"assistant prefill leaves the tool name as the next token. "
            f"Pass min_baseline_pass_rate=0 to override."
        )

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
    parser.add_argument(
        "--min-baseline-pass-rate",
        type=float,
        default=0.2,
        help="Abort if fewer than this fraction of prompts predict the expected "
        "tool before ablation (default: 0.2; 0 disables). Protects against a "
        "report computed from a handful of prompts.",
    )
    parser.add_argument(
        "--random-control",
        choices=["frequency-matched", "alive", "legacy"],
        default="frequency-matched",
        help="Random-control sampling: 'frequency-matched' (default) matches "
        "each contrastive feature's firing-rate bin, redrawn per prompt; "
        "'alive' draws uniformly from features firing on >0.1%% of inputs; "
        "'legacy' reproduces the old uniform-over-dictionary draw (a "
        "near-no-op on sparse dictionaries, kept for comparison).",
    )
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="Directory with step 1 shard_*.npy files for computing firing "
        "rates (default: directory containing --contrastive-features). Only "
        "used when no firing_rates.npy exists next to the SAE checkpoint.",
    )

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
        min_baseline_pass_rate=args.min_baseline_pass_rate,
        random_control=args.random_control,
        activations_dir=args.activations_dir,
    )
