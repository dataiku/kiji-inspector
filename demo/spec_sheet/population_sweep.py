#!/usr/bin/env python3
"""Read the model's first tool choice for EVERY unique tool_selection prompt.

The pairs parquet holds ~9.3K unique tool_selection requests, but the demo
sweep only ever read 650 of them.  This script runs the exact tool-choice
readout of ``demo/tool_selection/capture_decisions.py`` (prefill
``I'll use the``, exact first-token log-probabilities, second-token
disambiguation for ``file_read``/``file_write``) over all of them, on the
canonical modified-vLLM backend.  No new prompts are generated — these are
the training parquet's own requests.

The readout appends one JSON line per prompt to ``readout.jsonl`` and skips
prompts already present, so an interrupted run resumes where it stopped.

Usage (inside 575lab/kiji-inspector:dev):
    python demo/spec_sheet/population_sweep.py --model-name /models/... [--limit 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "demo" / "tool_selection"))
sys.path.insert(0, str(_REPO_ROOT / "demo" / "home_repair"))


def unique_tool_selection_prompts(pair_rows: list[dict]) -> list[str]:
    """All unique tool_selection requests, in first-appearance parquet order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for row in pair_rows:
        if row["scenario_name"] != "tool_selection":
            continue
        for prompt in (row["anchor_prompt"], row["contrast_prompt"]):
            if prompt not in seen:
                seen.add(prompt)
                ordered.append(prompt)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--pairs-dir", default=str(_REPO_ROOT / "output" / "pairs"))
    parser.add_argument("--output-dir", default=str(_DEMO_DIR / "output" / "population"))
    parser.add_argument(
        "--prompts",
        default=str(_DEMO_DIR / "output" / "population" / "prompt_list.json"),
        help="Pre-built prompt list (written by --write-prompts); falls back to the parquet.",
    )
    parser.add_argument(
        "--write-prompts",
        action="store_true",
        help="only write the prompt list from the parquet and exit (no model)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="smoke test: first N prompts")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if prompts_path.exists() and not args.write_prompts:
        requests = json.loads(prompts_path.read_text())
    else:
        import pandas as pd

        parquet_files = sorted(Path(args.pairs_dir).glob("shard_*.parquet"))
        pair_rows = pd.concat([pd.read_parquet(p) for p in parquet_files]).to_dict("records")
        requests = unique_tool_selection_prompts(pair_rows)
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        prompts_path.write_text(json.dumps(requests, indent=0) + "\n")
        if args.write_prompts:
            print(f"Wrote {len(requests)} prompts -> {prompts_path}")
            return

    import capture_decisions as cap
    import tool_selection_demo as demo

    from kiji_inspector.extraction.vllm_activation_extractor import (
        VLLMActivationConfig,
        VLLMActivationExtractor,
    )

    if args.limit:
        requests = requests[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    readout_path = output_dir / "readout.jsonl"
    done: set[str] = set()
    if readout_path.exists():
        with readout_path.open() as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["request"])
                except Exception:
                    pass
    pending = [r for r in requests if r not in done]
    print(f"{len(requests)} prompts, {len(done)} already done, {len(pending)} to read")
    if not pending:
        return

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=args.model_name,
            layers=[demo.SAE_LAYER],
            token_positions="decision",
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=2048,
            max_num_seqs=32,
        )
    )
    try:
        tree = demo.tool_token_tree(extractor.tokenizer)
        with readout_path.open("a") as handle:
            for start in range(0, len(pending), args.chunk_size):
                chunk = pending[start : start + args.chunk_size]
                prompts = demo.build_prompts(extractor.tokenizer, args.model_name, chunk)
                readout = cap.read_tool_choices(extractor, prompts, tree, min_shared_mass=0.005)
                for request, decision in zip(chunk, readout["decisions"], strict=True):
                    handle.write(
                        json.dumps(
                            {
                                "request": request,
                                "toolId": decision["toolId"],
                                "prob": decision["prob"],
                                "distribution": decision["distribution"],
                                "coverage": decision["coverage"],
                                "completion": decision.get("completion"),
                            }
                        )
                        + "\n"
                    )
                handle.flush()
                print(f"  {min(start + args.chunk_size, len(pending))}/{len(pending)}", flush=True)
        meta = {
            "model": args.model_name,
            "logprobsMode": readout.get("logprobs_mode"),
            "nPrompts": len(requests),
            "readout": "capture_decisions.read_tool_choices (exact logprobs, second-token read)",
        }
        (output_dir / "readout_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    finally:
        extractor.cleanup()
    print(f"Wrote {readout_path}")


if __name__ == "__main__":
    main()
