#!/usr/bin/env python3
"""Stage 2: read the model's first tool choice for many requests, batched.

``demo/home_repair/sweep_tool_choice.py`` cannot be used at dataset scale. It
drives the *modified* vLLM through ``_read_tool_choices``, whose comment reads:

    One request per generate() call: the fork only populates per-token
    logprobs for the first request of a batch, so batching silently drops
    the readout for every other prompt.

At one request per call, the full dataset's 3.2M requests is not a run anyone
finishes. The fix is that **the sweep does not need the fork at all**: it
records only ``toolId`` / ``prob`` / ``distribution`` and never touches a
hidden state, so the hidden-states connector -- the thing that forces
``chunk_size=1`` -- is pure overhead here. This script therefore uses stock
vLLM with ordinary batched generation and ``logprobs=N``, which is the same
top-k fallback ``_read_tool_choices`` already degrades to when the fork's exact
``logprob_token_ids`` is unavailable.

The readout itself is unchanged: same ``build_agent_prompt`` framing, same
``tool_first_token_ids`` / ``decision_from_logprobs`` helpers, so rows are
comparable with the published sweeps. Tools whose names share a first token
(``file_read`` / ``file_write``) are merged into one bucket ``"file_read|file_write"``,
exactly as ``sweep_tool_choice.py --scenario`` does; stage 3 skips pairs whose
two sides land in the same merged bucket, since their flip cannot be scored.

Output is **JSONL, appended and resumable**: re-running skips requests already
present, so a long sweep survives preemption. Runs one scenario at a time
because the system prompt and tool list are per scenario.

Cost. Prompts are ~250-400 tokens and only ~6 tokens are generated, so this is
prefill-bound. Measure your own rate with ``--limit 2000`` first and scale it:
the printed ``requests/sec`` times the candidate count is the whole job. Sweep
a sampled subset (stage 1 ``--per-theme``) before committing to all 3.2M.

Usage (inside 575lab/kiji-inspector:dev, or any vLLM with this model):
    python demo/steering/sweep/sweep_pairs_batched.py \\
        --model-name $MODEL --scenario tool_selection \\
        --candidates demo/steering/sweep/output/sweep_candidates/tool_selection/meta.json \\
        --results demo/steering/sweep/output/sweep_candidates/tool_selection/sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
sys.path.insert(0, str(_DEMO_DIR.parent / "home_repair"))

sys.path.insert(0, str(_DEMO_DIR))

import home_repair_demo as demo  # noqa: E402  (decision_from_logprobs / tool_display)
import tool_selection_demo as tool_demo  # noqa: E402  (token tree readout)

from kiji_inspector.extraction.extractor import build_agent_prompt  # noqa: E402
from kiji_inspector.extraction.vllm_activation_extractor import (  # noqa: E402
    recommended_chat_template_kwargs,
)


def merged_tool_tokens(tokenizer, tools: list[dict]) -> dict[str, int]:
    """Tool -> first token of ``" {name}"``, merging tools that collide.

    Kept only as the fallback for scenarios the token tree cannot express.
    Mirrors ``sweep_tool_choice.py``'s ``_lenient_ids``.
    """
    merged: dict[int, list[str]] = {}
    for tool in tools:
        token = int(tokenizer.encode(f" {tool['name']}", add_special_tokens=False)[0])
        merged.setdefault(token, []).append(tool["name"])
    if all(len(names) == 1 for names in merged.values()):
        return demo.tool_first_token_ids(tokenizer, tools)
    return {"|".join(names): token for token, names in merged.items()}


def read_with_tree(completion, tree, tool_ids: list[str]) -> dict:
    """Tool readout from one generation, using the demo's token tree.

    Two things the bare first-token readout gets wrong, both measured on a
    49k-request sweep of this model:

    * **Surface forms.** The model writes ``" API call tool"`` far more often
      than ``" api_call"``, and ``" API"`` is a different token from ``" api"``.
      Counting only the canonical form left 4.8% of requests below half
      coverage, 2,116 of them the single string ``" API"``.
    * **Shared first tokens.** ``file_read`` and ``file_write`` both start
      ``" file"``. A first-token-only readout must merge them into one
      unscorable bucket; the tree splits them on the second token.

    ``max_tokens>=2`` with logprobs gives the second-token distribution
    conditioned on the token actually sampled, which is exactly what
    ``distribution_from_tree`` needs for a shared prefix. Prefixes that were
    not sampled have no conditional available and split evenly, as that
    function documents.
    """
    first = (completion.logprobs or [{}])[0]
    first_logprobs = {int(tid): float(obj.logprob) for tid, obj in first.items()}
    second_logprobs: dict[int, dict[int, float]] = {}
    token_ids = completion.token_ids or []
    if len(token_ids) > 1 and completion.logprobs and len(completion.logprobs) > 1:
        sampled_first = int(token_ids[0])
        if sampled_first in tree["shared"]:
            second_logprobs[sampled_first] = {
                int(tid): float(obj.logprob) for tid, obj in completion.logprobs[1].items()
            }
    return tool_demo.distribution_from_tree(first_logprobs, second_logprobs, tree, tool_ids)


def load_requests(candidates_path: Path) -> list[str]:
    """Unique requests from a stage-1 ``meta.json`` (or a ``candidates.json``)."""
    payload = json.loads(candidates_path.read_text())
    requests: list[str] = []
    if isinstance(payload, list):  # meta.json
        for row in payload:
            requests.extend((row["anchor"], row["contrast"]))
    else:  # candidates.json: {group: [request, ...]}
        for group in payload.values():
            requests.extend(group)
    seen, unique = set(), []
    for request in requests:
        if request not in seen:
            seen.add(request)
            unique.append(request)
    return unique


def done_requests(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["request"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from a killed run
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--scenario", default="tool_selection")
    parser.add_argument("--candidates", required=True, help="Stage-1 meta.json or candidates.json")
    parser.add_argument("--results", required=True, help="JSONL, appended and resumable")
    parser.add_argument("--batch-size", type=int, default=2048, help="Requests per flush")
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=512,
        help="Concurrent sequences. NemotronH is a hybrid Mamba model and each decode "
        "sequence needs its own Mamba cache block, so vLLM's default of 1024 exceeds the "
        "~713 blocks that fit and engine startup fails. 512 batches well and stays under.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Sweep only the first N (timing)")
    args = parser.parse_args()

    scenario = json.loads((_REPO_ROOT / "scenarios" / f"{args.scenario}.json").read_text())
    requests = load_requests(Path(args.candidates))
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    already = done_requests(results_path)
    pending = [r for r in requests if r not in already]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(requests):,} unique requests, {len(already):,} already done, {len(pending):,} to go")
    if not pending:
        print("Nothing to do.")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()
    tool_ids = [t["name"] for t in scenario["tools"]]
    tree = fallback = None
    try:
        tree = tool_demo.tool_token_tree(tokenizer, scenario["tools"])
        shared = [t for t in tree["shared"]]
        print(f"token-tree readout: {len(tree['first'])} first tokens, {len(shared)} shared")
    except ValueError as exc:
        # Tools indistinguishable even at the second token: fall back to the
        # merged single-token buckets rather than reporting a wrong split.
        fallback = merged_tool_tokens(tokenizer, scenario["tools"])
        print(f"token tree unavailable ({exc}); merged-bucket readout: {sorted(fallback)}")
    template_kwargs = recommended_chat_template_kwargs(args.model_name, tokenizer)
    params = SamplingParams(max_tokens=6, temperature=0.0, logprobs=args.logprobs)

    started = time.time()
    written = 0
    with results_path.open("a") as handle:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            prompts = [
                build_agent_prompt(
                    system_prompt=scenario["system_prompt"],
                    tools=scenario["tools"],
                    user_request=request,
                    tokenizer=tokenizer,
                    chat_template_kwargs=template_kwargs,
                    close_think_block=bool(template_kwargs),
                )
                for request in chunk
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for request, output in zip(chunk, outputs, strict=True):
                completion = output.outputs[0]
                if tree is not None:
                    decision = read_with_tree(completion, tree, tool_ids)
                    decision["completion"] = completion.text
                else:
                    first = (completion.logprobs or [{}])[0]
                    logprobs = {int(tid): float(obj.logprob) for tid, obj in first.items()}
                    sampled = int(completion.token_ids[0]) if completion.token_ids else None
                    decision = demo.decision_from_logprobs(
                        logprobs,
                        fallback,
                        sampled_id=sampled,
                        completion=completion.text,
                        truncated=True,  # top-k logprobs, not the fork's exact set
                    )
                handle.write(
                    json.dumps(
                        {
                            "request": request,
                            "scenario": args.scenario,
                            "toolId": decision["toolId"],
                            "prob": decision["prob"],
                            "distribution": decision["distribution"],
                            "coverage": decision["coverage"],
                            "lowCoverage": decision["lowCoverage"],
                            "completion": decision["completion"],
                        }
                    )
                    + "\n"
                )
            handle.flush()
            written += len(chunk)
            rate = written / max(time.time() - started, 1e-6)
            remaining = (len(pending) - written) / rate if rate else 0
            print(
                f"  {written:,}/{len(pending):,}  {rate:.1f} req/s  "
                f"eta {remaining / 3600:.1f}h",
                flush=True,
            )

    total = time.time() - started
    print(f"\nWrote {written:,} rows to {results_path} in {total / 60:.1f} min")
    print(f"Sustained {written / max(total, 1e-6):.1f} requests/sec")


if __name__ == "__main__":
    main()
