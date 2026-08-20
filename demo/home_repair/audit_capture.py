#!/usr/bin/env python3
"""Capture session for the home-repair audit page (modified vLLM, canonical).

One engine session captures, for the 46 audit prompts (36 grid cells + 10
tripwire rows from ``audit_grid``):

* decision-position residual streams at every trained layer
  → ``audit_activations.npz`` (``steps`` + one ``residual_<layer>`` array
  per layer, rows in prompt order);
* the exact-logprob tool-choice readout per prompt (one ``generate()`` call
  per prompt — the fork fills per-token logprobs only for a batch's first
  request), reusing ``compare_sae_backends._read_tool_choices``;
* a longer greedy completion for each tripwire prompt, so the page can show
  whether the reply text mentions the hazard even though the tool choice
  follows the ask
  → ``audit_capture.json``.

The report is observational only; SAE encoding and the tripwire gate run on
CPU afterwards (``audit_report.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import audit_grid
import compare_sae_backends as csb
import home_repair_demo as demo
import numpy as np

from kiji_inspector.extraction.extractor import build_agent_prompt
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
    recommended_chat_template_kwargs,
)


def format_prompts(tokenizer, model_name: str, metadata: list[dict]) -> list[str]:
    """Training-style formatted prompts for arbitrary audit metadata rows."""
    template_kwargs = recommended_chat_template_kwargs(model_name, tokenizer)
    return [
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


def worklist(prompt_set: str) -> list[dict]:
    """The prompt rows a capture session runs, by named set."""
    if prompt_set == "holdout":
        return audit_grid.holdout_tripwire_prompts()
    return audit_grid.audit_prompts()


def smoke_worklist(metadata: list[dict]) -> list[dict]:
    """A cheap 3-prompt check: one grid cell plus one matched tripwire pair."""
    grid = next(item for item in metadata if item["kind"] == "grid")
    hazard = next(item for item in metadata if item["kind"] == "hazard")
    control = next(item for item in metadata if item.get("promptId") == hazard["controlId"])
    return [grid, control, hazard]


def completion_targets(metadata: list[dict]) -> list[dict]:
    """The rows that get a longer greedy completion (all tripwire rows)."""
    return [item for item in metadata if item["kind"] in ("hazard", "control")]


def assemble_report(
    metadata: list[dict],
    readout: dict | None,
    completions: dict[str, str],
    model_name: str,
    layers: list[int],
    prompt_parity: dict | None,
) -> dict:
    """Merge prompt rows, decisions (by step) and completions (by promptId)."""
    decisions = (readout or {}).get("decisions") or []
    by_step = {decision.get("step"): decision for decision in decisions}
    rows = []
    for item in metadata:
        row = dict(item)
        row["decision"] = by_step.get(item["step"])
        prompt_id = item.get("promptId")
        if prompt_id in completions:
            row["completion"] = completions[prompt_id]
        rows.append(row)
    return {
        "model": model_name,
        "layers": list(layers),
        "backend": "vllm",
        "logprobs_mode": (readout or {}).get("logprobs_mode"),
        "truncated": (readout or {}).get("truncated"),
        "prompt_parity": prompt_parity,
        "prompts": rows,
    }


def _cleanup_scratch(storage_dir: str, name: str, count: int) -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector as connector,
        )
    except Exception:  # pragma: no cover - connector layout is fork-specific
        connector = None
    for index in range(count):
        path = os.path.join(storage_dir, f"{name}_{index}.safetensors")
        try:
            if connector is not None:
                connector.cleanup_hidden_states(path)
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _long_completions(
    extractor, prompts: list[str], targets: list[dict], max_tokens: int
) -> dict[str, str]:
    """Greedy continuations from the decision position for the tripwire rows."""
    from vllm import SamplingParams

    storage_dir = getattr(extractor, "_storage_dir", None) or "."
    indices = [index for index, item in enumerate(targets)]
    params = [
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            extra_args={
                "kv_transfer_params": {
                    "hidden_states_path": os.path.join(
                        storage_dir, f"completion_{index}.safetensors"
                    ),
                    "include_output_tokens": False,
                }
            },
        )
        for index in indices
    ]
    outputs = extractor.llm.generate(prompts, params, use_tqdm=False)
    _cleanup_scratch(storage_dir, "completion", len(prompts))
    return {
        item["promptId"]: output.outputs[0].text
        for item, output in zip(targets, outputs, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--results-dir", default="demo/home_repair/output/audit")
    parser.add_argument("--layers", type=int, nargs="+", default=demo._TRAINED_LAYERS)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--completion-tokens", type=int, default=80)
    parser.add_argument(
        "--prompt-set",
        choices=("audit", "holdout"),
        default="audit",
        help="audit = 36 grid cells + 10 tripwire rows; holdout = the 18 "
        "pre-registered layer-34 validation pairs (see holdout_prereg.md).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cheap check: 3 prompts, layer 27 only, results under <results-dir>/smoke.",
    )
    args = parser.parse_args()
    if args.smoke and args.prompt_set != "audit":
        parser.error("--smoke applies to the audit prompt set only")
    if args.smoke:
        args.layers = [27]
        args.results_dir = str(Path(args.results_dir) / "smoke")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = worklist(args.prompt_set)
    if args.smoke:
        metadata = smoke_worklist(metadata)
    print(f"Capturing {len(metadata)} audit prompts at layers {args.layers}")

    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    prompts = format_prompts(hf_tokenizer, args.model_name, metadata)

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=args.model_name,
            layers=args.layers,
            token_positions="decision",
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=2048,
            max_num_seqs=max(16, args.batch_size),
        )
    )
    readout: dict | None = None
    completions: dict[str, str] = {}
    try:
        vllm_prompts = format_prompts(extractor.tokenizer, args.model_name, metadata)
        text_equal = [left == right for left, right in zip(prompts, vllm_prompts, strict=True)]
        token_equal = [
            csb._token_ids(hf_tokenizer, prompt) == csb._token_ids(extractor.tokenizer, prompt)
            for prompt in prompts
        ]
        prompt_parity = {
            "all_formatted_text_equal": all(text_equal),
            "all_token_ids_equal": all(token_equal),
        }

        activations = extractor.extract_batch(prompts, batch_size=args.batch_size)

        try:
            readout = csb._read_tool_choices(extractor, prompts, metadata)
            for item, decision in zip(metadata, readout["decisions"], strict=True):
                print(
                    f"  {item['step']}: {decision['display']} "
                    f"(p={decision['prob']:.2f}, coverage={decision['coverage']:.2f})"
                )
        except Exception:
            traceback.print_exc()
            print("  WARNING: tool-choice readout failed; the report will carry no decisions.")
            readout = None

        targets = completion_targets(metadata)
        if targets:
            index_of = {id(item): index for index, item in enumerate(metadata)}
            target_prompts = [prompts[index_of[id(item)]] for item in targets]
            try:
                completions = _long_completions(
                    extractor, target_prompts, targets, args.completion_tokens
                )
            except Exception:
                traceback.print_exc()
                print("  WARNING: completion capture failed; the report will carry none.")
                completions = {}
    finally:
        extractor.cleanup()

    arrays = {
        f"residual_{layer}": np.stack([item[f"residual_{layer}"] for item in activations]).astype(
            np.float32
        )
        for layer in args.layers
    }
    npz_path = results_dir / "audit_activations.npz"
    np.savez(npz_path, steps=np.array([item["step"] for item in metadata]), **arrays)

    report = assemble_report(
        metadata, readout, completions, args.model_name, args.layers, prompt_parity
    )
    report_path = results_dir / "audit_capture.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nWrote {npz_path} ({', '.join(sorted(arrays))}; {len(metadata)} rows)")
    print(f"Wrote {report_path}")
    for item in completion_targets(metadata):
        text = completions.get(item["promptId"], "")
        preview = " ".join(text.split())[:90]
        print(f"  completion {item['promptId']}: {preview}")


if __name__ == "__main__":
    main()
