#!/usr/bin/env python3
"""Sweep candidate user requests and read the model's first tool choice for each.

The demo's decision prompts only carry the short user request, so the
request wording decides which tool the model picks at ``I'll use the``.  This
utility loads the subject model once (modified vLLM, same extractor as the
demo) and prints the tool-choice distribution for every candidate so the
demo's base/contrast requests can be chosen from data instead of guessed.

Usage (inside 575lab/kiji-inspector:dev):
    python demo/home_repair/sweep_tool_choice.py --model-name /models/... \
        [--candidates candidates.json] [--results demo/home_repair/output/sweep/tool_choice_sweep.json]

``candidates.json`` is ``{"<group>": ["request", ...], ...}``; an entry may also
be ``{"request": ..., "history": [{"role", "content"}, ...]}`` to read a later
tool decision (prior turns inserted after the request).  Without it the
built-in candidate list below is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare_sae_backends as backend_compare
import home_repair_demo as demo

from kiji_inspector.extraction.extractor import build_agent_prompt
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
    recommended_chat_template_kwargs,
)

DEFAULT_CANDIDATES: dict[str, list[str]] = {
    "water_heater_noise (target pro_quote)": [
        "My 9-year-old gas water heater pops and rumbles when heating and the hot water has a rust tinge; should I call a professional?",
        "My gas water heater is 9 years old, makes loud popping and rumbling, and the water runs rusty; I want a licensed plumber's quote.",
        "My gas water heater rumbles loudly and smells faintly of gas near the burner; please get me professional repair quotes and costs.",
        "My 9-year-old gas water heater pops loudly while heating and hot water comes out rusty; how much would a pro charge to fix it?",
        "My old gas water heater rumbles and pops; I don't want to touch the gas burner myself, so find me a professional service quote.",
        "My gas water heater is popping and rumbling and I'm worried the tank is failing; get me quotes from licensed plumbers to replace it.",
    ],
    "water_heater_noise contrast (hazard removed)": [
        "My 2-year-old electric water heater pops a little while heating; can you find a guide to flush the sediment myself?",
        "My electric water heater makes a light popping sound when heating; show me a video tutorial for flushing sediment from the tank.",
        "My 2-year-old electric water heater pops slightly when heating and the water is clear; how much would a pro charge to flush it?",
    ],
    "dishwasher_leak (target parts_search)": [
        "My 3-year-old dishwasher leaks from the bottom because the door gasket is cracked; find me a replacement gasket with price and availability.",
        "My dishwasher door gasket is torn and leaking; search for a compatible replacement gasket, its price, and shipping time.",
        "My dishwasher's spray-arm seal is worn and leaking; I need a replacement part with pricing and when it can ship.",
        "My 3-year-old dishwasher leaks at the door because the gasket is cracked; what does a replacement door gasket cost and is it in stock?",
    ],
    "dishwasher_leak contrast (warranty)": [
        "My brand-new dishwasher, still under warranty, leaks from the bottom because the door gasket is cracked; what should I do?",
        "My dishwasher is still under manufacturer warranty and leaks at the door gasket; get me the service quote and what the warranty covers.",
    ],
    "disposal_stuck (target tutorial_search)": [
        "My garbage disposal hums but won't spin; show me a step-by-step video on how to unjam it with the hex wrench.",
        "My garbage disposal is jammed and humming; find a beginner video tutorial for clearing the jam and pressing the reset button.",
        "My garbage disposal hums without turning; I want a how-to video with difficulty rating and the tools needed to free it.",
        "My garbage disposal hums but the blades don't turn; find me a short video tutorial for freeing the flywheel from underneath.",
    ],
    "disposal_stuck contrast (hazard added)": [
        "My garbage disposal hums but won't spin, smells like burning plastic, and tripped the breaker; should I call an electrician?",
        "My garbage disposal hums, won't spin, and there is a burning smell from the motor; get me a professional repair quote.",
    ],
    "current demo requests": [
        prompt["request"] for prompt in demo.decision_prompts(include_contrasts=True)
    ],
}


def build_prompt_with_history(
    tokenizer, model_name: str, request: str, history: list[dict], prefill: str = "I'll use the"
) -> str:
    """Same system/tool framing as ``build_agent_prompt`` plus prior turns.

    ``history`` is a list of ``{"role", "content"}`` messages that follow the
    user request (e.g. the assistant's first tool call and its result), so the
    prefill lands at a *second* tool decision.
    """
    template_kwargs = recommended_chat_template_kwargs(model_name, tokenizer)
    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in demo._DECISION_TOOLS
    )
    messages = [
        {
            "role": "system",
            "content": (
                f"{demo._SYSTEM_PROMPT}\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                f"When you decide to use a tool, respond with the tool name."
            ),
        },
        {"role": "user", "content": request},
        *history,
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {})
    )
    if template_kwargs and formatted.rstrip().endswith("<think>"):
        formatted = formatted.rstrip() + "\n\n</think>\n\n"
    return formatted + prefill


def build_prompts(tokenizer, model_name: str, requests: list) -> list[str]:
    """``requests`` are strings (single-turn) or ``{"request", "history"}`` dicts."""
    template_kwargs = recommended_chat_template_kwargs(model_name, tokenizer)
    prompts = []
    for request in requests:
        if isinstance(request, dict):
            prompts.append(
                build_prompt_with_history(
                    tokenizer, model_name, request["request"], request.get("history") or []
                )
            )
            continue
        prompts.append(
            build_agent_prompt(
                system_prompt=demo._SYSTEM_PROMPT,
                tools=demo._DECISION_TOOLS,
                user_request=request,
                tokenizer=tokenizer,
                chat_template_kwargs=template_kwargs,
                close_think_block=bool(template_kwargs),
            )
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--results", default="demo/home_repair/output/sweep/tool_choice_sweep.json")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Scenario JSON (system_prompt + tools) to sweep instead of home_repair.json; "
            "tools sharing a first token are merged into one readout bucket."
        ),
    )
    args = parser.parse_args()
    if args.scenario:
        scenario = json.loads(Path(args.scenario).read_text())
        demo._SYSTEM_PROMPT = scenario["system_prompt"]
        demo._DECISION_TOOLS = scenario["tools"]
        strict_ids = demo.tool_first_token_ids

        def _lenient_ids(tokenizer, tools):
            merged: dict[int, list[str]] = {}
            for tool in tools:
                token = int(tokenizer.encode(f" {tool['name']}", add_special_tokens=False)[0])
                merged.setdefault(token, []).append(tool["name"])
            if all(len(names) == 1 for names in merged.values()):
                return strict_ids(tokenizer, tools)
            return {"|".join(names): token for token, names in merged.items()}

        demo.tool_first_token_ids = _lenient_ids

    candidates = DEFAULT_CANDIDATES
    if args.candidates:
        candidates = json.loads(Path(args.candidates).read_text())
    flat: list[tuple[str, str]] = [
        (group, request) for group, requests in candidates.items() for request in requests
    ]

    extractor = VLLMActivationExtractor(
        VLLMActivationConfig(
            model_name=args.model_name,
            layers=[demo._SAE_LAYER],
            token_positions="decision",
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=2048,
            max_num_seqs=32,
        )
    )
    try:
        prompts = build_prompts(extractor.tokenizer, args.model_name, [req for _, req in flat])
        metadata = [
            {"step": f"{group}::{index}", "toolId": None} for index, (group, _) in enumerate(flat)
        ]
        readout = backend_compare._read_tool_choices(extractor, prompts, metadata)
    finally:
        extractor.cleanup()

    rows = []
    for (group, request), decision in zip(flat, readout["decisions"], strict=True):
        rows.append(
            {
                "group": group,
                "request": request["request"] if isinstance(request, dict) else request,
                "history": request.get("history") if isinstance(request, dict) else None,
                "toolId": decision["toolId"],
                "prob": decision["prob"],
                "distribution": decision["distribution"],
                "coverage": decision["coverage"],
                "completion": decision["completion"],
            }
        )

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "logprobs_mode": readout.get("logprobs_mode"),
                "truncated": readout.get("truncated"),
                "rows": rows,
            },
            indent=2,
        )
    )

    current_group = None
    for row in rows:
        if row["group"] != current_group:
            current_group = row["group"]
            print(f"\n== {current_group}")
        dist = " ".join(f"{k[:6]}={v:.2f}" for k, v in row["distribution"].items())
        print(f"  {row['toolId']:<16} p={row['prob']:.2f}  [{dist}]  {row['request']}")
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
