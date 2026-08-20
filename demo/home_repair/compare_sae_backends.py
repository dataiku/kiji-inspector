#!/usr/bin/env python3
"""Evaluate modified-vLLM and optionally compare it with saved HF activations.

This is the canonical producer for the home-repair demo page: it captures the
decision-position residual streams for the base and contrast prompts, reads
the model's actual tool choice at the ``I'll use the`` position (next-token
log-probabilities over the four tool names), evaluates every trained SAE on
those activations, and writes ``vllm_native_evaluation.json`` which
``home_repair_demo.py --ui-from-evaluation`` turns into ``ui_data.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import evaluate_sae_layers as layer_eval
import home_repair_demo as demo
import numpy as np

from kiji_inspector.extraction.extractor import build_agent_prompt
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
    recommended_chat_template_kwargs,
)


def _formatted_prompts(
    tokenizer, model_name: str, include_contrasts: bool = True, include_probes: bool = True
) -> tuple[list[dict], list[str]]:
    metadata = layer_eval._prompt_metadata(
        include_contrasts=include_contrasts, include_probes=include_probes and include_contrasts
    )
    template_kwargs = recommended_chat_template_kwargs(model_name, tokenizer)
    prompts = [
        build_agent_prompt(
            system_prompt=demo._SYSTEM_PROMPT,
            tools=demo._DECISION_TOOLS,
            user_request=item["request"],
            tokenizer=tokenizer,
            chat_template_kwargs=template_kwargs,
            close_think_block=bool(template_kwargs),
        )
        for item in metadata
    ]
    return metadata, prompts


def _token_ids(tokenizer, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=True)
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded.input_ids)


def _read_tool_choices(
    extractor, prompts: list[str], metadata: list[dict], chunk_size: int = 1
) -> dict:
    """Read the model's tool choice at the decision position via vLLM logprobs.

    Uses the extractor's already-loaded engine (``extractor.llm``) so no second
    model load is needed.  The request carries the same connector arguments as
    the extractor's own sampling params (a scratch ``hidden_states_path``) so
    the hidden-states connector, which writes a file per request, stays
    consistent; the scratch files are removed afterwards.  Exact per-token
    log-probabilities are requested with ``logprob_token_ids`` when the vLLM
    build supports it, otherwise the top-20 fallback is used and flagged.
    """
    from vllm import SamplingParams

    tool_to_token = demo.tool_first_token_ids(extractor.tokenizer, demo._DECISION_TOOLS)
    token_ids = sorted(tool_to_token.values())
    storage_dir = getattr(extractor, "_storage_dir", None) or "."

    def _params(index: int, exact: bool):
        path = os.path.join(storage_dir, f"readout_{index}.safetensors")
        extra = {"kv_transfer_params": {"hidden_states_path": path, "include_output_tokens": False}}
        if exact:
            return SamplingParams(
                max_tokens=6, temperature=0.0, logprob_token_ids=token_ids, extra_args=extra
            )
        return SamplingParams(max_tokens=6, temperature=0.0, logprobs=20, extra_args=extra)

    truncated = False
    try:
        params = [_params(index, exact=True) for index in range(len(prompts))]
    except TypeError:
        truncated = True
        params = [_params(index, exact=False) for index in range(len(prompts))]
    # One request per generate() call: the fork only populates per-token
    # logprobs for the first request of a batch, so batching silently drops
    # the readout for every other prompt (the sampled token is still right).
    outputs = []
    for start in range(0, len(prompts), chunk_size):
        outputs.extend(
            extractor.llm.generate(
                prompts[start : start + chunk_size],
                params[start : start + chunk_size],
                use_tqdm=False,
            )
        )

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector as connector,
        )
    except Exception:  # pragma: no cover - connector layout is fork-specific
        connector = None
    for index in range(len(prompts)):
        path = os.path.join(storage_dir, f"readout_{index}.safetensors")
        try:
            if connector is not None:
                connector.cleanup_hidden_states(path)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    logprobs_mode = None
    try:
        logprobs_mode = str(extractor.llm.llm_engine.model_config.logprobs_mode)
    except Exception:
        pass

    decisions = []
    for item, output in zip(metadata, outputs, strict=True):
        completion = output.outputs[0]
        first_logprobs = (completion.logprobs or [{}])[0]
        logprobs = {int(tid): float(obj.logprob) for tid, obj in first_logprobs.items()}
        sampled = int(completion.token_ids[0]) if completion.token_ids else None
        decision = demo.decision_from_logprobs(
            logprobs,
            tool_to_token,
            sampled_id=sampled,
            completion=completion.text,
            truncated=truncated,
        )
        decision["step"] = item["step"]
        decision["topTokens"] = [
            {
                "tokenId": int(tid),
                "logprob": round(float(obj.logprob), 4),
                "text": getattr(obj, "decoded_token", None),
            }
            for tid, obj in sorted(first_logprobs.items(), key=lambda kv: -kv[1].logprob)[:8]
        ]
        decisions.append(decision)
    return {"decisions": decisions, "logprobs_mode": logprobs_mode, "truncated": truncated}


def _extract_vllm(
    model_name: str,
    layers: list[int],
    batch_size: int,
    gpu_memory_utilization: float,
    include_contrasts: bool = True,
    read_tool_choices: bool = True,
    limit_prompts: int | None = None,
    include_probes: bool = True,
) -> tuple[list[dict], dict[int, np.ndarray], dict, dict | None]:
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    metadata, prompts = _formatted_prompts(
        hf_tokenizer, model_name, include_contrasts, include_probes
    )
    if limit_prompts:
        metadata, prompts = metadata[:limit_prompts], prompts[:limit_prompts]
    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=model_name,
            layers=layers,
            token_positions="decision",
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
            max_num_seqs=max(16, batch_size),
        )
    )
    readout: dict | None = None
    try:
        _, vllm_formatted = _formatted_prompts(
            extractor.tokenizer, model_name, include_contrasts, include_probes
        )
        if limit_prompts:
            vllm_formatted = vllm_formatted[:limit_prompts]
        text_equal = [left == right for left, right in zip(prompts, vllm_formatted, strict=True)]
        token_equal = [
            _token_ids(hf_tokenizer, prompt) == _token_ids(extractor.tokenizer, prompt)
            for prompt in prompts
        ]
        activations = extractor.extract_batch(prompts, batch_size=batch_size)
        if read_tool_choices:
            try:
                readout = _read_tool_choices(extractor, prompts, metadata)
                for decision in readout["decisions"]:
                    print(
                        f"  Tool choice {decision['step']}: {decision['display']} "
                        f"(p={decision['prob']:.2f}, coverage={decision['coverage']:.2f})"
                    )
            except Exception:
                traceback.print_exc()
                print("  WARNING: tool-choice readout failed; the UI will hide it.")
                readout = None
    finally:
        extractor.cleanup()

    by_layer = {
        layer: np.stack([item[f"residual_{layer}"] for item in activations]).astype(np.float32)
        for layer in layers
    }
    prompt_parity = {
        "all_formatted_text_equal": all(text_equal),
        "all_token_ids_equal": all(token_equal),
        "formatted_text_equal": text_equal,
        "token_ids_equal": token_equal,
    }
    return metadata, by_layer, prompt_parity, readout


def _vector_parity(hf: np.ndarray, vllm: np.ndarray) -> dict:
    if hf.shape != vllm.shape:
        raise ValueError(f"HF shape {hf.shape} != vLLM shape {vllm.shape}")
    cosine = []
    relative_l2 = []
    rms_ratio = []
    for hf_row, vllm_row in zip(hf, vllm, strict=True):
        cosine.append(1.0 - layer_eval._cosine_distance(hf_row, vllm_row))
        hf_norm = float(np.linalg.norm(hf_row))
        relative_l2.append(float(np.linalg.norm(vllm_row - hf_row) / hf_norm) if hf_norm else 0.0)
        hf_rms = float(np.sqrt(np.mean(np.square(hf_row))))
        vllm_rms = float(np.sqrt(np.mean(np.square(vllm_row))))
        rms_ratio.append(vllm_rms / hf_rms if hf_rms else 0.0)

    hf_flat = hf.reshape(-1).astype(np.float64)
    vllm_flat = vllm.reshape(-1).astype(np.float64)
    denominator = float(np.dot(hf_flat, hf_flat))
    best_fit_scale = float(np.dot(hf_flat, vllm_flat) / denominator) if denominator else 0.0
    scaled_residual = vllm_flat - best_fit_scale * hf_flat
    vllm_norm = float(np.linalg.norm(vllm_flat))
    return {
        "mean_cosine_similarity": float(np.mean(cosine)),
        "min_cosine_similarity": float(np.min(cosine)),
        "mean_relative_l2_error": float(np.mean(relative_l2)),
        "mean_rms_ratio_vllm_over_hf": float(np.mean(rms_ratio)),
        "best_fit_scale_vllm_from_hf": best_fit_scale,
        "relative_error_after_best_fit_scale": (
            float(np.linalg.norm(scaled_residual) / vllm_norm) if vllm_norm else 0.0
        ),
        "mse": float(np.mean(np.square(vllm - hf))),
        "per_prompt": [
            {
                "cosine_similarity": cosine[index],
                "relative_l2_error": relative_l2[index],
                "rms_ratio_vllm_over_hf": rms_ratio[index],
            }
            for index in range(len(cosine))
        ],
    }


def _feature_rank_parity(hf_result: dict, vllm_result: dict) -> dict:
    if hf_result["layer"] != vllm_result["layer"]:
        raise ValueError("Cannot compare feature rankings from different layers.")
    pairs = []
    for hf_features, vllm_features in zip(
        hf_result["top_features"], vllm_result["top_features"], strict=True
    ):
        hf_ranking = [feature["index"] for feature in hf_features]
        vllm_ranking = [feature["index"] for feature in vllm_features]
        pairs.append(
            {
                "rbo": layer_eval._truncated_rbo(hf_ranking, vllm_ranking),
                "top_k_jaccard": layer_eval._jaccard(hf_ranking, vllm_ranking),
            }
        )
    return {
        "mean_rbo": float(np.mean([pair["rbo"] for pair in pairs])),
        "mean_top_k_jaccard": float(np.mean([pair["top_k_jaccard"] for pair in pairs])),
        "per_prompt": pairs,
    }


def _render_parity_markdown(report: dict) -> str:
    lines = [
        "# HF vs modified-vLLM residual-stream parity",
        "",
        f"- Formatted prompt text equal: {report['prompt_parity']['all_formatted_text_equal']}",
        f"- Token IDs equal: {report['prompt_parity']['all_token_ids_equal']}",
        "- SAE thresholds: native checkpoint values",
        "",
        "| Layer | Vector cosine | Min cosine | Relative L2 | RMS ratio vLLM/HF | Best-fit scale | Scaled residual | Feature RBO | Top-k Jaccard | vLLM L0 | vLLM norm. MSE | vLLM purity | vLLM contam. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["layers"]:
        vector = item["vector_parity"]
        rank = item["feature_rank_parity"]
        vllm = item["vllm_evaluation"]
        lines.append(
            f"| {item['layer']} | {vector['mean_cosine_similarity']:.6f} | "
            f"{vector['min_cosine_similarity']:.6f} | "
            f"{vector['mean_relative_l2_error']:.3f} | "
            f"{vector['mean_rms_ratio_vllm_over_hf']:.3f} | "
            f"{vector['best_fit_scale_vllm_from_hf']:.3f} | "
            f"{vector['relative_error_after_best_fit_scale']:.3f} | "
            f"{rank['mean_rbo']:.3f} | {rank['mean_top_k_jaccard']:.3f} | "
            f"{vllm['l0']['mean']:.2f} | "
            f"{vllm['reconstruction']['normalized_mse']:.5f} | "
            f"{vllm['scenario_specificity']['scenario_purity']:.1%} | "
            f"{vllm['scenario_specificity']['contamination_mass_share']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--hf-activations-file",
        help="Optional matching HF activations; omit for a vLLM-only evaluation.",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--results-dir", default="demo/home_repair/output/backend_parity")
    parser.add_argument("--layers", type=int, nargs="+", default=demo._TRAINED_LAYERS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-contrasts",
        action="store_true",
        help="Capture only the three base prompts (legacy report shape).",
    )
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="Skip the paraphrase and keyword-control probe prompts.",
    )
    parser.add_argument(
        "--no-tool-choice",
        action="store_true",
        help="Skip the next-token tool-choice readout.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cheap check: one prompt, layer 27 only, results under <results-dir>/smoke.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.layers = [27]
        args.results_dir = str(Path(args.results_dir) / "smoke")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    hf_activations = None
    if args.hf_activations_file:
        with np.load(args.hf_activations_file) as saved:
            hf_activations = {
                layer: saved[f"residual_{layer}"].astype(np.float32) for layer in args.layers
            }

    metadata, vllm_activations, prompt_parity, readout = _extract_vllm(
        model_name=args.model_name,
        layers=args.layers,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        include_contrasts=not args.no_contrasts,
        read_tool_choices=not args.no_tool_choice,
        limit_prompts=1 if args.smoke else None,
        include_probes=not args.no_probes,
    )
    np.savez_compressed(
        results_dir / "vllm_six_layer_demo_activations.npz",
        **{f"residual_{layer}": values for layer, values in vllm_activations.items()},
    )

    hf_results = []
    vllm_results = []
    parity_layers = []
    for layer in args.layers:
        print(f"\nEvaluating native SAE at layer {layer}...")
        contrastive_map = layer_eval._load_contrastive_map(output_dir, layer)
        vllm_result = layer_eval._evaluate_layer(
            layer,
            vllm_activations[layer],
            metadata,
            output_dir,
            threshold_offset=0.0,
            top_k=args.top_k,
            device=args.device,
            contrastive_map=contrastive_map,
        )
        vllm_results.append(vllm_result)
        if hf_activations is not None:
            if hf_activations[layer].shape[0] != len(metadata):
                raise ValueError(
                    "HF activations have a different prompt count than this run; "
                    "re-capture them with evaluate_sae_layers.py."
                )
            hf_result = layer_eval._evaluate_layer(
                layer,
                hf_activations[layer],
                metadata,
                output_dir,
                threshold_offset=0.0,
                top_k=args.top_k,
                device=args.device,
                contrastive_map=contrastive_map,
            )
            hf_results.append(hf_result)
            parity_layers.append(
                {
                    "layer": layer,
                    "vector_parity": _vector_parity(hf_activations[layer], vllm_activations[layer]),
                    "feature_rank_parity": _feature_rank_parity(hf_result, vllm_result),
                    "hf_evaluation": hf_result,
                    "vllm_evaluation": vllm_result,
                }
            )

    base = {
        "model": args.model_name,
        "backend": "vllm",
        "layers_evaluated": args.layers,
        "threshold_offset": 0.0,
        "top_k": args.top_k,
        "prompts": metadata,
        "decisions": readout["decisions"] if readout else None,
        "logprobs_mode": readout.get("logprobs_mode") if readout else None,
        "tool_choice_truncated": readout.get("truncated") if readout else None,
    }
    vllm_report = {**base, "layers": vllm_results}
    (results_dir / "vllm_native_evaluation.json").write_text(json.dumps(vllm_report, indent=2))
    (results_dir / "vllm_native_evaluation.md").write_text(layer_eval._render_markdown(vllm_report))
    if hf_activations is not None:
        hf_report = {**base, "layers": hf_results}
        parity_report = {
            "model": args.model_name,
            "prompt_parity": prompt_parity,
            "layers": parity_layers,
        }
        (results_dir / "hf_native_evaluation.json").write_text(json.dumps(hf_report, indent=2))
        (results_dir / "hf_native_evaluation.md").write_text(layer_eval._render_markdown(hf_report))
        (results_dir / "hf_vllm_parity.json").write_text(json.dumps(parity_report, indent=2))
        (results_dir / "hf_vllm_parity.md").write_text(_render_parity_markdown(parity_report))
    print(f"\nWrote vLLM evaluation reports to {results_dir}")


if __name__ == "__main__":
    main()
