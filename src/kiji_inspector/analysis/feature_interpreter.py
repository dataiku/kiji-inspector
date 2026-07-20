"""
Feature interpretation pipeline (Step 5).

5a: Load activations from Step 2 numpy shards.
5b: Encode through SAE, collect top/bottom activating prompts per feature.
5c: Auto-label features via LLM (generator model).
5d: Generate user-facing explanation report.
"""

from __future__ import annotations

import json
import re
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
    for sp in tqdm(shard_paths, desc="[4a] Loading shards", unit="shard"):
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

    activations = np.stack(unique_activations, axis=0)
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
    from kiji_inspector.core.sae_core import JumpReLUSAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_pretrained(sae_checkpoint, device=device)
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype

    n_features = len(feature_indices)
    n_prompts = len(activations)
    feat_idx_t = torch.tensor(feature_indices, device=device, dtype=torch.long)

    print(
        f"  Encoding {n_prompts} prompts through SAE (d_sae={sae.d_sae}), "
        f"streaming top-{top_n}/bottom-{bottom_n} for {n_features} features..."
    )

    eff_top_n = min(top_n, n_prompts)
    eff_bot_n = min(bottom_n, n_prompts)

    # Running heaps on CPU — total ≈ n_features × (top_n+bottom_n) × 8B ≪ 1 GB
    top_vals = torch.full((n_features, eff_top_n), float("-inf"), dtype=torch.float32)
    top_idx = torch.full((n_features, eff_top_n), -1, dtype=torch.long)
    bot_vals = torch.full((n_features, eff_bot_n), float("inf"), dtype=torch.float32)
    bot_idx = torch.full((n_features, eff_bot_n), -1, dtype=torch.long)
    sum_vals = torch.zeros(n_features, dtype=torch.float64)
    max_vals = torch.full((n_features,), float("-inf"), dtype=torch.float32)
    nonzero_count = torch.zeros(n_features, dtype=torch.long)

    chunk_size = 4096

    with torch.no_grad():
        for i in tqdm(
            range(0, n_prompts, chunk_size),
            desc="[4b] SAE encode + streaming top-K",
            unit="chunk",
        ):
            end = min(i + chunk_size, n_prompts)
            chunk = torch.from_numpy(activations[i:end]).to(device=device, dtype=sae_dtype)
            if sae.rms_scale is not None and sae.rms_scale > 0:
                chunk = chunk / sae.rms_scale
            features = sae.encode(chunk)  # (chunk, d_sae)
            # Project to only the features we care about, move to CPU as fp32
            feat_sub = features.index_select(1, feat_idx_t).to(torch.float32).cpu()
            # Transpose so each row is one feature's values across the chunk
            feat_sub_t = feat_sub.t().contiguous()  # (n_features, chunk)
            chunk_len = feat_sub_t.shape[1]
            global_idx = torch.arange(i, end, dtype=torch.long).unsqueeze(0).expand(
                n_features, chunk_len
            )

            # Running stats
            sum_vals += feat_sub_t.to(torch.float64).sum(dim=1)
            max_vals = torch.maximum(max_vals, feat_sub_t.max(dim=1).values)
            nonzero_count += (feat_sub_t > 0).sum(dim=1)

            # Streaming top-N: combine current heap + new chunk, keep top
            combined_vals = torch.cat([top_vals, feat_sub_t], dim=1)
            combined_idx = torch.cat([top_idx, global_idx], dim=1)
            top_vals, pos = combined_vals.topk(eff_top_n, dim=1, largest=True)
            top_idx = combined_idx.gather(1, pos)

            # Streaming bottom-N (smallest values)
            combined_vals = torch.cat([bot_vals, feat_sub_t], dim=1)
            combined_idx = torch.cat([bot_idx, global_idx], dim=1)
            bot_vals, pos = combined_vals.topk(eff_bot_n, dim=1, largest=False)
            bot_idx = combined_idx.gather(1, pos)

            del features, feat_sub, feat_sub_t, combined_vals, combined_idx, pos

    results: dict[int, dict] = {}
    for j, feat_idx in enumerate(feature_indices):
        top_examples = [
            {"prompt": prompts[int(gi)], "activation": round(float(v), 6)}
            for v, gi in zip(top_vals[j].tolist(), top_idx[j].tolist(), strict=True)
            if gi >= 0
        ]
        bottom_examples = [
            {"prompt": prompts[int(gi)], "activation": round(float(v), 6)}
            for v, gi in zip(bot_vals[j].tolist(), bot_idx[j].tolist(), strict=True)
            if gi >= 0
        ]
        results[feat_idx] = {
            "top": top_examples,
            "bottom": bottom_examples,
            "mean_activation": round(sum_vals[j].item() / n_prompts, 6),
            "max_activation": round(max_vals[j].item(), 6),
            "frac_nonzero": round(nonzero_count[j].item() / n_prompts, 4),
        }

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


def _try_parse_label_json(raw: str) -> dict | None:
    """Try to extract a valid label dict from LLM output.

    Returns {"label", "description", "confidence"} or None.
    """

    def _validate(parsed: dict) -> dict | None:
        if not isinstance(parsed, dict):
            return None
        label = parsed.get("label", "")
        # Reject template placeholders the model echoed back
        if label in ("...", "", "high|medium|low"):
            return None
        return {
            "label": label,
            "description": parsed.get("description", ""),
            "confidence": parsed.get("confidence", "low"),
        }

    # Fast path: entire string is valid JSON
    try:
        result = _validate(json.loads(raw))
        if result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract first JSON object from noisy output
    match = re.search(r"\{[^{}]*\}", raw)
    if match:
        try:
            result = _validate(json.loads(match.group()))
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _run_labeling_subprocess(
    label_prompts: list[tuple[int, str]],
    judging_model: str,
    tp_size: int,
    max_model_len: int,
    output_path: str,
    gpu_ids: str | None = None,
) -> None:
    """Child process: load vLLM, label all features, save results, exit."""
    import os

    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    from vllm import LLM, SamplingParams

    print(f"  [subprocess] Loading vLLM model: {judging_model}")
    # Apply model-family engine defaults so hybrid-MoE judges (e.g. Qwen3.6)
    # load reliably for generation too — same fix as the extractor path.
    from kiji_inspector.extraction.vllm_activation_extractor import recommended_vllm_kwargs

    gen_kwargs = recommended_vllm_kwargs(judging_model)
    if gen_kwargs.get("moe_backend") == "triton":
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    llm = LLM(
        model=judging_model,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        # Hybrid (linear-attention) models need one Mamba cache block per
        # decode sequence for CUDA graph capture; the default (1024) exceeds
        # the available blocks at this memory budget.
        max_num_seqs=256,
        enable_expert_parallel=False,
        disable_log_stats=True,
        **gen_kwargs,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=500,
    )

    # Build prompts using the model's own chat template. Two guards against
    # the judge meta-reasoning in prose instead of answering (which burns the
    # whole token budget before any JSON appears):
    #   1. enable_thinking=False — Qwen3-family templates suppress the
    #      <think> block entirely.
    #   2. Pre-fill the assistant response with the start of the JSON object,
    #      so the model must continue it with content rather than preamble.
    JSON_PREFILL = '{"label": "'
    system = (
        "You are an expert at interpreting neural network features. "
        "Output only valid JSON, no markdown fences."
    )
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for _feat_idx, user_content in label_prompts:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + JSON_PREFILL
        )

    print(f"  [subprocess] Labeling {len(formatted_prompts)} features...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    labels: dict[str, dict] = {}
    for (feat_idx, _), output in zip(label_prompts, outputs, strict=True):
        # Re-attach the pre-filled JSON prefix the model continued from.
        raw = (JSON_PREFILL + output.outputs[0].text).strip()
        # Strip Qwen3 thinking blocks (<think>...</think>)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Also handle unclosed thinking block (truncated output)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
        # Strip degenerate repetitive tokens (e.g. "thought000...000")
        raw = re.sub(r"(?:thought|0{10,}|\.{10,})\S*", "", raw).strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        parsed = _try_parse_label_json(raw)
        if parsed:
            labels[str(feat_idx)] = parsed
        else:
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
    judging_model: str,
    tp_size: int,
    max_model_len: int,
    output_dir: str | Path,
    dp_size: int = 1,
) -> dict[int, dict]:
    """Label features using an LLM in a subprocess (GPU memory isolation).

    When ``dp_size > 1``, spawns N model copies on N GPUs (each with
    ``tp_size=1``) to label features in parallel.

    Returns:
        Dict mapping feature_index -> {"label": str, "description": str, "confidence": str}
    """
    import multiprocessing as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build prompts
    label_prompts = []
    for feat_idx, examples in feature_examples.items():
        prompt = _format_label_prompt(feat_idx, examples)
        label_prompts.append((feat_idx, prompt))

    ctx = mp.get_context("spawn")

    if dp_size > 1:
        # Data-parallel: split prompts across N GPUs
        chunk_size = (len(label_prompts) + dp_size - 1) // dp_size
        chunks = [
            label_prompts[i : i + chunk_size] for i in range(0, len(label_prompts), chunk_size)
        ]
        output_paths = [str(output_dir / f"_labels_temp_rank{r}.json") for r in range(len(chunks))]

        processes = []
        for rank, (chunk, out_path) in enumerate(zip(chunks, output_paths, strict=True)):
            p = ctx.Process(
                target=_run_labeling_subprocess,
                args=(chunk, judging_model, 1, max_model_len, out_path, str(rank)),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Labeling subprocess (pid={p.pid}) failed with exit code {p.exitcode}"
                )

        # Merge results from all ranks
        labels: dict[int, dict] = {}
        for out_path in output_paths:
            with open(out_path) as f:
                raw_labels = json.load(f)
            labels.update({int(k): v for k, v in raw_labels.items()})
            Path(out_path).unlink(missing_ok=True)
    else:
        labels_path = str(output_dir / "_labels_temp.json")
        p = ctx.Process(
            target=_run_labeling_subprocess,
            args=(label_prompts, judging_model, tp_size, max_model_len, labels_path),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"Labeling subprocess failed with exit code {p.exitcode}")

        with open(labels_path) as f:
            raw_labels = json.load(f)

        labels = {int(k): v for k, v in raw_labels.items()}
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
