"""
Feature interpretation pipeline (Step 5).

5a: Extract activations for all prompts via Nemotron (batched).
5b: Encode through SAE, collect top/bottom activating prompts per feature.
5c: Auto-label features via LLM.
5d: Generate user-facing explanation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 5a: Extract activations for all unique prompts
# ---------------------------------------------------------------------------


def load_activations_from_shards(
    activations_dir: str | Path,
) -> tuple[list[str], np.ndarray]:
    """Load activations and prompt texts from Step 2 output.

    Step 2 saves ``prompts.json`` (user request per activation vector)
    alongside the numpy shards.  We load both, deduplicate prompts, and
    return one activation per unique prompt.

    Args:
        activations_dir: Directory containing shard_*.npy, metadata.json,
            and prompts.json.

    Returns:
        prompts: Ordered list of unique user request strings.
        activations: numpy array of shape (N, d_model), float32.
    """
    activations_dir = Path(activations_dir)

    # Load metadata
    metadata_path = activations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {activations_dir}. Run step 2 first.")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load prompt texts
    prompts_path = activations_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"prompts.json not found in {activations_dir}. Re-run step 2 to generate it."
        )
    with open(prompts_path) as f:
        all_prompts: list[str] = json.load(f)

    print(f"  Loading activation shards from {activations_dir}")
    print(
        f"  Model: {metadata['model']}, layer: {metadata['layer']}, d_model: {metadata['d_model']}"
    )

    # Load and concatenate all shards
    shard_paths = sorted(activations_dir.glob("shard_*.npy"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npy files found in {activations_dir}")

    shards = []
    for sp in tqdm(shard_paths, desc="[5a] Loading shards", unit="shard"):
        shards.append(np.load(sp))

    all_activations = np.concatenate(shards, axis=0)  # (total_tokens, d_model), float16
    del shards
    print(f"  Loaded {all_activations.shape[0]} activation vectors, shape {all_activations.shape}")

    # Deduplicate: keep first activation for each unique prompt
    seen: set[str] = set()
    unique_prompts: list[str] = []
    unique_activations: list[np.ndarray] = []

    for prompt, act in zip(all_prompts, all_activations, strict=True):
        if prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)
            unique_activations.append(act)

    del all_activations

    activations = np.stack(unique_activations, axis=0).astype(np.float32)
    del unique_activations

    print(f"  Unique prompts: {len(unique_prompts)}, activations shape: {activations.shape}")

    return unique_prompts, activations


# ---------------------------------------------------------------------------
# 5b: Encode through SAE, collect top/bottom activating prompts
# ---------------------------------------------------------------------------


def collect_max_activating_examples(
    prompts: list[str],
    activations: np.ndarray,
    sae_checkpoint: str,
    feature_indices: list[int],
    top_n: int = 20,
    bottom_n: int = 10,
) -> dict[int, dict]:
    """For each feature, find the prompts with highest and lowest activation.

    Args:
        prompts: List of user request strings (same order as activations).
        activations: (N, d_model) float32 numpy array.
        sae_checkpoint: Path to trained SAE checkpoint.
        feature_indices: Which SAE features to analyze.
        top_n: Number of top-activating examples to collect.
        bottom_n: Number of near-zero examples to collect.

    Returns:
        Dict mapping feature_index -> {
            "top": [{"prompt": str, "activation": float}, ...],
            "bottom": [{"prompt": str, "activation": float}, ...],
        }
    """
    from sae.model import JumpReLUSAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(sae_checkpoint, device=device)
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype

    print(f"  Encoding {len(prompts)} prompts through SAE (d_sae={sae.d_sae})...")

    # Encode all activations through SAE in chunks to avoid OOM
    chunk_size = 4096
    all_features: list[torch.Tensor] = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(activations), chunk_size), desc="[5b] SAE encoding", unit="chunk"
        ):
            chunk = torch.from_numpy(activations[i : i + chunk_size]).to(
                device=device, dtype=sae_dtype
            )
            features = sae.encode(chunk)  # (chunk, d_sae)
            all_features.append(features.cpu())

    # Concatenate on CPU
    feature_matrix = torch.cat(all_features, dim=0)  # (N, d_sae), on CPU
    del all_features

    results: dict[int, dict] = {}

    for feat_idx in tqdm(feature_indices, desc="[5b] Collecting examples", unit="feature"):
        col = feature_matrix[:, feat_idx]  # (N,)

        # Top activating
        topk_vals, topk_ids = col.topk(min(top_n, len(col)))
        top_examples = []
        for val, idx in zip(topk_vals.tolist(), topk_ids.tolist(), strict=True):
            top_examples.append({"prompt": prompts[idx], "activation": round(val, 6)})

        # Bottom (near-zero / zero)
        bottomk_vals, bottomk_ids = col.topk(min(bottom_n, len(col)), largest=False)
        bottom_examples = []
        for val, idx in zip(bottomk_vals.tolist(), bottomk_ids.tolist(), strict=True):
            bottom_examples.append({"prompt": prompts[idx], "activation": round(val, 6)})

        results[feat_idx] = {
            "top": top_examples,
            "bottom": bottom_examples,
            "mean_activation": round(col.mean().item(), 6),
            "max_activation": round(col.max().item(), 6),
            "frac_nonzero": round((col > 0).float().mean().item(), 4),
        }

    del feature_matrix
    return results


# ---------------------------------------------------------------------------
# 5c: Auto-label features via LLM (runs in subprocess for GPU isolation)
# ---------------------------------------------------------------------------

_LABEL_PROMPT_TEMPLATE = """You are analyzing features learned by a Sparse Autoencoder (SAE) trained on an AI agent's internal activations at the moment it decides which tool to use.

Each feature corresponds to a specific concept or pattern the agent uses when making tool-selection decisions.

For feature #{feature_index}, here are the prompts that MOST activate this feature:
{top_prompts}

And here are prompts where this feature is INACTIVE (near-zero activation):
{bottom_prompts}

This feature is active in {frac_nonzero_pct}% of all prompts, with mean activation {mean_activation} and max {max_activation}.

Based on these examples, provide:
1. A short label (3-8 words) describing what this feature detects
2. A one-sentence description explaining the concept
3. Your confidence (high/medium/low) in this interpretation

Output as JSON:
{{"label": "...", "description": "...", "confidence": "high|medium|low"}}"""


def _format_label_prompt(feat_idx: int, examples: dict) -> str:
    top_lines = "\n".join(
        f"  [{ex['activation']:.4f}] {ex['prompt']}" for ex in examples["top"][:15]
    )
    bottom_lines = "\n".join(
        f"  [{ex['activation']:.4f}] {ex['prompt']}" for ex in examples["bottom"][:8]
    )
    return _LABEL_PROMPT_TEMPLATE.format(
        feature_index=feat_idx,
        top_prompts=top_lines,
        bottom_prompts=bottom_lines,
        frac_nonzero_pct=round(examples["frac_nonzero"] * 100, 1),
        mean_activation=examples["mean_activation"],
        max_activation=examples["max_activation"],
    )


def _run_labeling_subprocess(
    label_prompts: list[tuple[int, str]],
    qwen_model: str,
    tp_size: int,
    max_model_len: int,
    output_path: str,
) -> None:
    """Child process: load vLLM, label all features, save results, exit."""

    from vllm import LLM, SamplingParams

    print(f"  [subprocess] Loading vLLM model: {qwen_model}")
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
        temperature=0.3,
        top_p=0.9,
        max_tokens=500,
    )

    # Build ChatML prompts
    system = (
        "You are an expert at interpreting neural network features. "
        "Output only valid JSON, no markdown fences."
    )
    chatml_prompts = []
    for _feat_idx, user_content in label_prompts:
        chatml_prompts.append(
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    print(f"  [subprocess] Labeling {len(chatml_prompts)} features...")
    outputs = llm.generate(chatml_prompts, sampling_params)

    labels: dict[str, dict] = {}
    for (feat_idx, _), output in zip(label_prompts, outputs, strict=True):
        raw = output.outputs[0].text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(raw)
            labels[str(feat_idx)] = {
                "label": parsed.get("label", "unknown"),
                "description": parsed.get("description", ""),
                "confidence": parsed.get("confidence", "low"),
            }
        except (json.JSONDecodeError, KeyError):
            labels[str(feat_idx)] = {
                "label": "parse_error",
                "description": raw[:200],
                "confidence": "low",
            }

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  [subprocess] Saved {len(labels)} labels to {output_path}")


def label_features_via_llm(
    feature_examples: dict[int, dict],
    qwen_model: str,
    tp_size: int,
    max_model_len: int,
    output_dir: str | Path,
) -> dict[int, dict]:
    """Label features using an LLM in a subprocess (GPU memory isolation).

    Returns:
        Dict mapping feature_index -> {"label": str, "description": str, "confidence": str}
    """
    import multiprocessing as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = str(output_dir / "_labels_temp.json")

    # Build prompts
    label_prompts = []
    for feat_idx, examples in feature_examples.items():
        prompt = _format_label_prompt(feat_idx, examples)
        label_prompts.append((feat_idx, prompt))

    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_run_labeling_subprocess,
        args=(label_prompts, qwen_model, tp_size, max_model_len, labels_path),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Labeling subprocess failed with exit code {p.exitcode}")

    with open(labels_path) as f:
        raw_labels = json.load(f)

    # Convert string keys back to int
    labels = {int(k): v for k, v in raw_labels.items()}

    # Clean up temp file
    Path(labels_path).unlink(missing_ok=True)

    return labels


# ---------------------------------------------------------------------------
# 5d: Generate user-facing explanation report
# ---------------------------------------------------------------------------


def generate_explanation_report(
    contrastive_features_path: str | Path,
    feature_examples: dict[int, dict],
    feature_labels: dict[int, dict],
    output_dir: str | Path,
) -> Path:
    """Combine contrastive features, examples, and labels into a report.

    Produces ``feature_descriptions.json`` (per-feature details) and
    ``decision_report.json`` (per-contrast-type explanations).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load contrastive features from step 4
    with open(contrastive_features_path) as f:
        contrastive = json.load(f)

    # Build per-feature descriptions
    feature_descriptions: dict[str, dict] = {}
    for feat_idx, label_info in feature_labels.items():
        examples = feature_examples.get(feat_idx, {})
        feature_descriptions[str(feat_idx)] = {
            **label_info,
            "mean_activation": examples.get("mean_activation", 0),
            "max_activation": examples.get("max_activation", 0),
            "frac_nonzero": examples.get("frac_nonzero", 0),
            "top_examples": [ex["prompt"] for ex in examples.get("top", [])[:10]],
            "bottom_examples": [ex["prompt"] for ex in examples.get("bottom", [])[:10]],
        }

    desc_path = output_dir / "feature_descriptions.json"
    with open(desc_path, "w") as f:
        json.dump(feature_descriptions, f, indent=2)
    print(f"  Saved feature descriptions: {desc_path}")

    # Build per-contrast-type explanations
    decision_report: dict[str, dict] = {}
    for ct_value, ct_info in contrastive.items():
        if ct_value.startswith("_"):
            continue
        explained_features = []
        for feat_info in ct_info.get("top_features", [])[:10]:
            idx = feat_info["feature_index"]
            label = feature_labels.get(idx, {})
            explained_features.append(
                {
                    "feature_index": idx,
                    "label": label.get("label", "unlabeled"),
                    "description": label.get("description", ""),
                    "confidence": label.get("confidence", "low"),
                    "mean_abs_diff": feat_info["mean_abs_diff"],
                    "anchor_mean_activation": feat_info["anchor_mean_activation"],
                    "contrast_mean_activation": feat_info["contrast_mean_activation"],
                }
            )

        decision_report[ct_value] = {
            "num_pairs": ct_info["num_pairs"],
            "explanation": _build_plain_language_explanation(ct_value, explained_features),
            "key_features": explained_features,
        }

    report_path = output_dir / "decision_report.json"
    with open(report_path, "w") as f:
        json.dump(decision_report, f, indent=2)
    print(f"  Saved decision report: {report_path}")

    # Print summary
    print("\n  Decision explanations:")
    for ct_value, info in decision_report.items():
        print(f"    {ct_value}:")
        print(f"      {info['explanation']}")

    return report_path


def _build_plain_language_explanation(contrast_type: str, features: list[dict]) -> str:
    """Build a plain-language explanation from labeled features."""
    labeled = [f for f in features if f["label"] not in ("unlabeled", "parse_error")]
    if not labeled:
        return f"No interpretable features identified for {contrast_type}."

    # Pick top 3 labeled features
    top = labeled[:3]
    parts = []
    for f in top:
        direction = (
            "anchor" if f["anchor_mean_activation"] > f["contrast_mean_activation"] else "contrast"
        )
        parts.append(f'"{f["label"]}" (stronger in {direction} prompt)')

    feature_str = ", ".join(parts[:-1])
    if len(parts) > 1:
        feature_str += f", and {parts[-1]}"
    else:
        feature_str = parts[0]

    ct_readable = contrast_type.replace("_vs_", " vs ").replace("_", " ")
    return (
        f"When deciding between {ct_readable} tools, the model relies on "
        f"features like {feature_str}."
    )
