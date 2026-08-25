#!/usr/bin/env python3
"""Tool-selection demo: one-cue request pairs where the model's tool flips.

The ``tool_selection`` scenario (8 tools) is the one in the SAE's training
data where the subject model's first tool choice is a genuine decision: in a
sweep of 650 training-pair requests, 194 were split (p < 0.8) and 116 of 325
one-cue pairs flipped tool.  This demo takes a handful of those pairs — two
requests that differ in one cue (cached vs live, one record vs all records,
one file vs the whole project, local vs remote, check vs update, create vs
update, direct vs delegated) — and asks what the SAE (page layer 43; 27 and
34 for comparison) says about the flip:

* which features are specific to each side of the pair (the *cue features*),
* whether switching those families off on their own side moves the tool
  probability (HF backend, against mass-matched random ablations),
* whether patching one side's cue families into the other side moves the
  decision across (the same test that failed for the home-repair asks).

Everything observational comes from the modified-vLLM backend
(``capture_decisions.py``); interventions run on HuggingFace in the same
container (``attribute_pairs.py`` for decision-token ablation / cross-patch,
``trace_pairs.py`` for the per-token cue map, position-resolved ablation,
dose-response and steered generations); ``build_ui_data`` turns all of it
into ``output/ui_data.json`` for ``index.html``.  Generic feature helpers are
imported from the home-repair demo, which lives next door.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
sys.path.insert(0, str(_DEMO_DIR.parent / "home_repair"))

import home_repair_demo as hr  # noqa: E402  (generic helpers + HF engine)

SCENARIO_NAME = "tool_selection"
_SCENARIO_PATH = _REPO_ROOT / "scenarios" / "tool_selection.json"
_SCENARIO = json.loads(_SCENARIO_PATH.read_text())
SYSTEM_PROMPT: str = _SCENARIO["system_prompt"]
TOOLS: list[dict] = _SCENARIO["tools"]
TOOL_IDS: list[str] = [tool["name"] for tool in TOOLS]
SAE_LAYER = hr._SAE_LAYER
DECISION_PREFILL = "I'll use the"

# Pairs are selected from the sweep by ``select_pairs.py`` (flip × lexical
# overlap, one per theme, ≤2 per tool combination, no tool named) and stored in
# ``pairs.json`` next to this file; ``sweep`` records what the sweep read so a
# re-capture can be checked against it.  Other scenarios select the same way
# with ``rank_flips.py --select-demo``.
_PAIRS_PATH = _DEMO_DIR / "pairs.json"
_PAIRS_FILE = json.loads(_PAIRS_PATH.read_text())
PAIRS: list[dict] = _PAIRS_FILE["pairs"]
PAIR_SELECTION: dict = {k: v for k, v in _PAIRS_FILE.items() if k != "pairs"}

SIDES = ("a", "b")

# Hand-written probes per pair side (paraphrases + one keyword control), see
# probes.json.  Pairs without an entry simply get no probe panel.
_PROBES_PATH = _DEMO_DIR / "probes.json"
PROBES: dict = {
    k: v for k, v in json.loads(_PROBES_PATH.read_text()).items() if not k.startswith("about")
}

# Where this scenario's pairs/probes live and its outputs go.  Rebound by
# ``configure`` so several scenarios can share this module without their
# capture/steering/trace directories colliding.
DEMO_DIR = _DEMO_DIR


_STEERING_ROOT = _DEMO_DIR.parent / "steering"


def _resolve_scenario_dir(scenario: str) -> Path:
    """Where a scenario's ``pairs.json``, ``probes.json`` and ``output/`` live.

    Everything built against the shipped SAEs sits under ``demo/steering/``;
    the older stand-alone demos sit directly under ``demo/``.  ``steering``
    wins so that ``--scenario tool_selection`` reads the current-SAE study
    rather than the published page, whose embedded features come from a
    different dictionary and must not be mixed with it.
    """
    candidate = _STEERING_ROOT / scenario
    return candidate if candidate.exists() else _DEMO_DIR.parent / scenario


def configure(scenario: str, demo_dir: Path | str | None = None) -> None:
    """Point this module at another scenario's config, pairs and probes.

    The four driver scripts (``capture_decisions``, ``attribute_pairs``,
    ``trace_pairs`` and this module's ``build_ui_data``) read everything
    through the module globals set here, so a single ``configure`` call at
    startup is enough to run the whole demo for a different scenario.

    ``demo_dir`` defaults to :func:`_resolve_scenario_dir` and is where
    ``pairs.json``, ``probes.json`` and ``output/`` are read and written.
    """
    global SCENARIO_NAME, _SCENARIO_PATH, _SCENARIO, SYSTEM_PROMPT, TOOLS, TOOL_IDS
    global _PAIRS_PATH, _PAIRS_FILE, PAIRS, PAIR_SELECTION, _PROBES_PATH, PROBES, DEMO_DIR

    SCENARIO_NAME = scenario
    DEMO_DIR = Path(demo_dir) if demo_dir else _resolve_scenario_dir(scenario)
    _SCENARIO_PATH = _REPO_ROOT / "scenarios" / f"{scenario}.json"
    _SCENARIO = json.loads(_SCENARIO_PATH.read_text())
    SYSTEM_PROMPT = _SCENARIO["system_prompt"]
    TOOLS = _SCENARIO["tools"]
    TOOL_IDS = [tool["name"] for tool in TOOLS]

    _PAIRS_PATH = DEMO_DIR / "pairs.json"
    _PAIRS_FILE = json.loads(_PAIRS_PATH.read_text())
    PAIRS = _PAIRS_FILE["pairs"]
    PAIR_SELECTION = {k: v for k, v in _PAIRS_FILE.items() if k != "pairs"}

    _PROBES_PATH = DEMO_DIR / "probes.json"
    PROBES = (
        {
            k: v
            for k, v in json.loads(_PROBES_PATH.read_text()).items()
            if not k.startswith("about")
        }
        if _PROBES_PATH.exists()
        else {}
    )
    # ``tool_token_tree`` / ``distribution_from_tree`` take TOOLS / TOOL_IDS as
    # *default arguments*, which bind at def time; every caller omits them, so
    # rebinding the globals alone would silently keep the old scenario's tools.
    tool_token_tree.__defaults__ = (TOOLS,)
    distribution_from_tree.__defaults__ = (TOOL_IDS,)
    use_scenario_in_home_repair_module()


def decision_prompts(include_probes: bool = False) -> list[dict]:
    """Both sides of every pair, in a fixed order (A then B within a pair).

    With ``include_probes`` the paraphrases (``{pid}_{S}_P1``, ``_P2``) and
    keyword controls (``{pid}_{S}_K``) follow, after all base prompts, each
    carrying ``kind`` and ``base`` (the step it probes).
    """
    prompts = []
    for pair in PAIRS:
        for side in SIDES:
            prompts.append(
                {
                    "step": f"{pair['id']}_{side.upper()}",
                    "pair": pair["id"],
                    "side": side,
                    "kind": "base",
                    "request": pair[side]["request"],
                }
            )
    if not include_probes:
        return prompts
    for pair in PAIRS:
        probes = PROBES.get(pair["id"]) or {}
        for side in SIDES:
            block = probes.get(side) or {}
            base = f"{pair['id']}_{side.upper()}"
            for i, text in enumerate(block.get("paraphrases") or [], start=1):
                prompts.append(
                    {
                        "step": f"{base}_P{i}",
                        "pair": pair["id"],
                        "side": side,
                        "kind": "paraphrase",
                        "base": base,
                        "request": text,
                    }
                )
            keyword = block.get("keyword")
            if keyword:
                prompts.append(
                    {
                        "step": f"{base}_K",
                        "pair": pair["id"],
                        "side": side,
                        "kind": "keyword",
                        "base": base,
                        "cue": keyword.get("cue"),
                        "request": keyword["request"],
                    }
                )
    return prompts


def build_prompts(tokenizer, model_name: str, requests: list[str]) -> list[str]:
    """Training-style decision prompts for this scenario (same builder as training)."""
    from kiji_inspector.extraction.extractor import build_agent_prompt
    from kiji_inspector.extraction.vllm_activation_extractor import (
        recommended_chat_template_kwargs,
    )

    template_kwargs = recommended_chat_template_kwargs(model_name, tokenizer)
    return [
        build_agent_prompt(
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
            user_request=request,
            tokenizer=tokenizer,
            chat_template_kwargs=template_kwargs,
            close_think_block=bool(template_kwargs),
            assistant_prefill=DECISION_PREFILL,
        )
        for request in requests
    ]


def use_scenario_in_home_repair_module() -> None:
    """Point the home-repair HF engine / prompt builder at this scenario."""
    hr._SYSTEM_PROMPT = SYSTEM_PROMPT
    hr._DECISION_TOOLS = TOOLS


# ---------------------------------------------------------------------------
# Tool readout over a token-prefix tree
# ---------------------------------------------------------------------------


def tool_surface_forms(name: str) -> list[str]:
    """Ways the model writes a tool name after ``I'll use the``.

    Besides the canonical ``api_call`` the model also writes ``API call`` /
    ``api call``; all forms count toward the same tool.
    """
    parts = name.split("_")
    spaced = " ".join(parts)
    forms = [name, spaced, parts[0].upper() + " " + " ".join(parts[1:]), spaced.capitalize()]
    seen: list[str] = []
    for form in forms:
        form = form.strip()
        if form and form not in seen:
            seen.append(form)
    return seen


def tool_token_tree(tokenizer, tools: list[dict] = TOOLS) -> dict:
    """Token layout of the tool names (all surface forms) after ``I'll use the``.

    Returns ``{"first": {t1: {tool: [t2, ...]}}, "shared": [t1, ...]}``: for
    every first token ``t1`` the tools it can start and, when more than one
    tool shares ``t1`` (``file_read`` / ``file_write`` share ``" file"``), the
    second tokens that disambiguate them.  A readout needs the first-token
    log-probabilities plus, for each shared ``t1``, the next-token
    log-probabilities after it.
    """
    first: dict[int, dict[str, set[int]]] = {}
    for tool in tools:
        name = tool["name"]
        first_word = name.split("_")[0].lower()
        covered = False
        for form in tool_surface_forms(name):
            ids = tokenizer.encode(f" {form}", add_special_tokens=False)
            if not ids:
                continue
            # Only forms whose first token is the whole first word count
            # (" API", " Api", " api"); fragments such as " W" or " IN" would
            # attribute unrelated next-token mass to a tool.
            if tokenizer.decode([int(ids[0])]).strip().lower() != first_word:
                continue
            bucket = first.setdefault(int(ids[0]), {})
            bucket.setdefault(name, set())
            if len(ids) > 1:
                bucket[name].add(int(ids[1]))
            covered = True
        if not covered:
            # The tokenizer splits this tool's first word, so no surface form
            # starts with a whole-word token (``escalation_system`` ->
            # " escal" + "ation").  Fall back to that prefix token and record
            # the second token, which is exactly what the shared-prefix
            # machinery below already disambiguates on.  Without this the tool
            # never enters the tree and silently reads p = 0 forever, while
            # generations that plainly name it are attributed to whichever
            # other tool holds residual mass.
            for form in tool_surface_forms(name):
                ids = tokenizer.encode(f" {form}", add_special_tokens=False)
                if len(ids) < 2:
                    continue
                piece = tokenizer.decode([int(ids[0])]).strip().lower()
                # A real prefix, long enough not to be a stray fragment.
                if len(piece) < 3 or not first_word.startswith(piece):
                    continue
                bucket = first.setdefault(int(ids[0]), {})
                bucket.setdefault(name, set())
                bucket[name].add(int(ids[1]))
                covered = True
        if not covered:
            raise ValueError(
                f"Tool {name!r} has no readable first token: no surface form of it starts "
                f"with a whole-word or >=3-character prefix token. Rename the tool or add a "
                f"surface form; leaving it out would silently report p=0 for it."
            )
    shared = []
    for token, names in first.items():
        if len(names) > 1:
            shared.append(token)
            seconds = [names[n] for n in names]
            if any(not s for s in seconds):
                raise ValueError(f"Tools {list(names)} share token {token} with no second token")
            for i, a in enumerate(seconds):
                for b in seconds[i + 1 :]:
                    if a & b:
                        raise ValueError(
                            f"Tools {list(names)} are not distinct at the second token"
                        )
    return {
        "first": {t: {n: sorted(s) for n, s in names.items()} for t, names in first.items()},
        "shared": sorted(shared),
    }


def second_token_ids(tree: dict, first_token: int) -> list[int]:
    return sorted({t2 for seconds in tree["first"][first_token].values() for t2 in seconds})


def distribution_from_tree(
    first_logprobs: dict[int, float],
    second_logprobs: dict[int, dict[int, float]],
    tree: dict,
    tools: list[str] = TOOL_IDS,
) -> dict:
    """Combine first-token (and, where needed, second-token) logprobs into p(tool).

    ``first_logprobs`` maps first-token id -> log p at the decision position;
    ``second_logprobs[t1]`` maps second-token id -> log p after emitting ``t1``.
    Shared first tokens split their mass by the conditional second-token
    probabilities (renormalised over the tools that share the prefix; evenly
    when no second-token readout is available).
    """
    import math

    raw: dict[str, float] = dict.fromkeys(tools, 0.0)
    for token, names in tree["first"].items():
        p_first = math.exp(first_logprobs[token]) if token in first_logprobs else 0.0
        if len(names) == 1:
            (name,) = names
            raw[name] = raw.get(name, 0.0) + p_first
            continue
        conditional = second_logprobs.get(token, {})
        weights = {
            name: sum(math.exp(conditional[t2]) for t2 in seconds if t2 in conditional)
            for name, seconds in names.items()
        }
        total = sum(weights.values())
        for name in names:
            raw[name] = raw.get(name, 0.0) + p_first * (
                weights[name] / total if total > 0 else 1.0 / len(names)
            )
    coverage = float(sum(raw.values()))
    distribution = {t: (raw.get(t, 0.0) / coverage if coverage > 0 else 0.0) for t in tools}
    best = max(distribution, key=distribution.get) if coverage > 0 else None
    return {
        "toolId": best,
        "display": best,
        "prob": round(distribution[best], 4) if best else 0.0,
        "distribution": {t: round(p, 4) for t, p in distribution.items()},
        "raw": {t: round(raw.get(t, 0.0), 6) for t in tools},
        "coverage": round(coverage, 4),
        "lowCoverage": coverage < 0.5,
    }


# ---------------------------------------------------------------------------
# Feature analysis of a pair
# ---------------------------------------------------------------------------


def side_specific_rows(
    this_active: list[tuple[int, float]],
    other_active: list[tuple[int, float]],
    labels: dict | None,
    top_n: int = 6,
) -> list[dict]:
    """Features stronger on this side than on the other, merged into families.

    ``delta`` is this-side minus other-side activation (absence = 0); rows are
    ranked by it, near-duplicate labels merged (``merged`` lists the twins),
    and the top ``top_n`` returned.  These are the *cue features* of a side.
    """
    this_map = {int(i): float(a) for i, a in this_active if a > 0}
    other_map = {int(i): float(a) for i, a in other_active if a > 0}
    max_activation = max(this_map.values(), default=0.0) or 1.0
    rows = []
    for index, activation in this_map.items():
        delta = activation - other_map.get(index, 0.0)
        if delta <= 0:
            continue
        rows.append(
            {
                "index": index,
                "label": hr._label_for(labels, index),
                "activation": round(activation, 4),
                "other": round(other_map.get(index, 0.0), 4),
                "delta": round(delta, 4),
                "share": round(activation / max_activation, 4),
            }
        )
    rows.sort(key=lambda row: (-row["delta"], -row["activation"]))
    rows = hr.dedupe_feature_rows(rows)
    rows.sort(key=lambda row: (-row["delta"], -row["activation"]))
    for row in rows:
        row.setdefault("merged", [])
        # dedupe keeps the strongest twin's activation/delta; keep ``other`` consistent.
        row["other"] = round(row["activation"] - row["delta"], 4)
    return rows[:top_n]


def shared_rows(
    a_active: list[tuple[int, float]],
    b_active: list[tuple[int, float]],
    labels: dict | None,
    top_n: int = 5,
) -> list[dict]:
    """Features strong on both sides (ranked by the smaller activation)."""
    a_map = {int(i): float(a) for i, a in a_active if a > 0}
    b_map = {int(i): float(a) for i, a in b_active if a > 0}
    rows = [
        {
            "index": index,
            "label": hr._label_for(labels, index),
            "a": round(a_map[index], 4),
            "b": round(b_map[index], 4),
            "minActivation": round(min(a_map[index], b_map[index]), 4),
        }
        for index in set(a_map) & set(b_map)
    ]
    rows.sort(key=lambda row: -row["minActivation"])
    rows = hr.dedupe_feature_rows(rows)
    rows.sort(key=lambda row: -row["minActivation"])
    return rows[:top_n]


def pair_feature_analysis(
    a_active: list[tuple[int, float]],
    b_active: list[tuple[int, float]],
    labels: dict | None,
    top_n: int = 6,
) -> dict:
    return {
        "aFeatures": side_specific_rows(a_active, b_active, labels, top_n),
        "bFeatures": side_specific_rows(b_active, a_active, labels, top_n),
        "shared": shared_rows(a_active, b_active, labels),
        "overlap": hr.active_overlap(a_active, b_active),
        "numActive": {
            "a": sum(1 for _, v in a_active if v > 0),
            "b": sum(1 for _, v in b_active if v > 0),
        },
    }


def other_tool_for(target_tool: str, other_distribution: dict, this_distribution: dict) -> str:
    """The tool a side's decision is weighed against.

    The other side's choice when it differs from ``target_tool``; otherwise
    (the pair sharpens rather than flips) this side's own runner-up.
    """
    ranked_other = sorted(other_distribution.items(), key=lambda kv: -kv[1])
    if ranked_other and ranked_other[0][0] != target_tool:
        return ranked_other[0][0]
    ranked_this = sorted(this_distribution.items(), key=lambda kv: -kv[1])
    for tool, _ in ranked_this:
        if tool != target_tool:
            return tool
    return target_tool


# ---------------------------------------------------------------------------
# UI payload
# ---------------------------------------------------------------------------


def layer_block(report: dict, layer: int | None = None) -> dict:
    """The per-layer block (``active_features`` …) of a capture report."""
    layer = report.get("layer", SAE_LAYER) if layer is None else layer
    for block in report.get("layers", []):
        if int(block.get("layer")) == int(layer):
            return block
    if "active_features" in report and int(report.get("layer", SAE_LAYER)) == int(layer):
        return report  # single-layer report shape
    raise ValueError(f"Capture report has no SAE layer {layer}.")


def _labels_from_report(report: dict, layer: int | None = None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for rows in layer_block(report, layer).get("active_features", []):
        for row in rows:
            labels[str(row["index"])] = row.get("label", "")
    return labels


def _active_by_step(report: dict, layer: int | None = None) -> dict[str, list[tuple[int, float]]]:
    rows_by_prompt = layer_block(report, layer)["active_features"]
    return {
        prompt["step"]: [(int(row["index"]), float(row["activation"])) for row in rows]
        for prompt, rows in zip(report["prompts"], rows_by_prompt, strict=True)
    }


def check_report_prompts(report: dict) -> None:
    """The report must hold exactly the base prompts, optionally followed by the probes."""
    found = [(p["step"], p["request"]) for p in report.get("prompts", [])]
    base = [(p["step"], p["request"]) for p in decision_prompts()]
    full = [(p["step"], p["request"]) for p in decision_prompts(include_probes=True)]
    if found not in (base, full):
        raise ValueError(
            "The capture report's prompts do not match the current pair/probe definitions; "
            "rerun capture_decisions.py."
        )


def probe_evidence(
    rows: list[dict],
    base_active: list[tuple[int, float]],
    other_active: list[tuple[int, float]],
    probe_active: list[tuple[int, float]],
    base_choice: dict | None,
    probe_choice: dict | None,
    top: int = 4,
) -> dict:
    """How one probe prompt relates to its base side's cue features.

    ``rows`` are the side's cue families (page rows); a family *fires* when
    any member is active on the probe.  Returns the per-family activation of
    the first ``top`` families (probe vs base), the firing count, the
    active-set cosine to the base and to the other side, and the tool choice.
    """
    probe_map = {int(i): float(v) for i, v in probe_active if v > 0}
    base_map = {int(i): float(v) for i, v in base_active if v > 0}
    firing = 0
    fam_rows = []
    for k, row in enumerate(rows):
        family = [int(row["index"])] + [int(i) for i in row.get("merged") or []]
        act = sum(probe_map.get(i, 0.0) for i in family)
        base_act = sum(base_map.get(i, 0.0) for i in family)
        fires = act > 0
        firing += int(fires)
        if k < top:
            fam_rows.append(
                {
                    "index": int(row["index"]),
                    "label": row["label"],
                    "activation": round(act, 3),
                    "base": round(base_act, 3),
                    "fires": fires,
                }
            )
    same_tool = None
    if base_choice and probe_choice:
        same_tool = base_choice.get("display") == probe_choice.get("display")
    return {
        "familiesFiring": firing,
        "familiesTotal": len(rows),
        "families": fam_rows,
        "cosineToBase": hr.active_overlap(base_active, probe_active).get("cosine"),
        "cosineToOther": hr.active_overlap(other_active, probe_active).get("cosine"),
        "modelChoice": probe_choice,
        "sameTool": same_tool,
        "numActive": len(probe_map),
    }


_MIN_CAUSAL_EFFECT = 0.02  # below 2 pp a family is "descriptive" whatever the random band


def attach_causal(rows: list[dict], block: dict | None) -> None:
    """Attach per-family ablation results (``attribute_pairs.py``) to rows, in place.

    A row is *descriptive* unless its ablation moves the target probability by
    more than both the mass-matched random band and ``_MIN_CAUSAL_EFFECT``.
    """
    if not block:
        return
    by_index = {int(entry["index"]): entry for entry in block.get("rows", [])}
    threshold = max(float(block.get("controlThreshold") or 0.0), _MIN_CAUSAL_EFFECT)
    for row in rows:
        entry = by_index.get(int(row["index"]))
        if not entry:
            continue
        delta = entry.get("deltaTarget")
        row["causal"] = {
            "deltaTarget": delta,
            "deltaOther": entry.get("deltaOther"),
            "targetTool": block.get("targetTool"),
            "otherTool": block.get("otherTool"),
            "hfActivation": entry.get("hfActivation"),
            "inactiveUnderHf": bool(entry.get("inactiveUnderHf")),
            "intervened": entry.get("intervened"),
            "argmaxChanged": entry.get("argmaxChanged"),
            "descriptive": delta is None or abs(float(delta)) <= threshold,
        }


def _cross_patch_with_parity(block: dict | None, decisions: dict, steps: dict) -> dict | None:
    """Flag cross-patch directions whose HF baseline (on the patched side) differs from vLLM."""
    if not block:
        return None
    out = {}
    for key, inj in block.items():
        into_step = steps.get(inj.get("intoSide"))
        vllm_choice = (decisions.get(into_step) or {}).get("display") if into_step else None
        hf_choice = inj.get("intoBaselineChoice")
        entry = dict(inj)
        entry["intoBaselineVllmChoice"] = vllm_choice
        entry["intoBaselineMismatch"] = bool(hf_choice and vllm_choice and hf_choice != vllm_choice)
        out[key] = entry
    return out


def trace_for_pair(trace: dict | None, pid: str, steps: dict) -> dict:
    """Slice ``trace_results.json`` (``trace_pairs.py``) for one pair.

    Returns ``{"positions": {side: ...}, "dose": {...}, "generations": {...}}``
    with empty/None parts where the trace has nothing for this pair.
    """
    if not trace:
        return {"positions": {}, "dose": None, "generations": None}
    positions = {}
    for side, step in steps.items():
        entry = (trace.get("positions") or {}).get(step)
        if entry:
            positions[side] = entry
    return {
        "positions": positions,
        "dose": (trace.get("dose") or {}).get(pid) or None,
        "generations": (trace.get("generations") or {}).get(pid) or None,
    }


def _probe_block(
    pid: str,
    side: str,
    rows: list[dict],
    active: dict,
    active_by_step: dict,
    decisions: dict,
    base_choice: dict | None,
    probe_prompts: list[dict],
) -> dict | None:
    """Paraphrase / keyword-control evidence for one pair side (None if not captured)."""
    other = "b" if side == "a" else "a"
    mine = [p for p in probe_prompts if p["pair"] == pid and p["side"] == side]
    if not mine or not rows:
        return None
    out: dict = {"paraphrases": [], "keyword": None}
    for p in mine:
        evidence = probe_evidence(
            rows,
            active[side],
            active[other],
            active_by_step.get(p["step"], []),
            base_choice,
            decisions.get(p["step"]),
        )
        evidence.update({"step": p["step"], "request": p["request"], "kind": p["kind"]})
        if p["kind"] == "keyword":
            evidence["cue"] = p.get("cue")
            out["keyword"] = evidence
        else:
            out["paraphrases"].append(evidence)
    return out


def build_ui_data(
    report: dict,
    steering: dict | None = None,
    model_name: str = "",
    layer: int | None = None,
    trace: dict | None = None,
) -> dict:
    """Assemble ``ui_data.json`` from the vLLM capture (+ optional HF attribution / trace)."""
    check_report_prompts(report)
    layer = int(
        layer
        if layer is not None
        else (steering or {}).get("layer") or report.get("layer", SAE_LAYER)
    )
    if steering and steering.get("layer") is not None and int(steering["layer"]) != layer:
        raise ValueError(
            f"Steering results are for layer {steering['layer']} but the page is built for {layer}."
        )
    if trace and trace.get("layer") is not None and int(trace["layer"]) != layer:
        raise ValueError(
            f"Trace results are for layer {trace['layer']} but the page is built for {layer}."
        )
    block = layer_block(report, layer)
    labels = _labels_from_report(report, layer)
    active_by_step = _active_by_step(report, layer)
    decisions = {d["step"]: d for d in report.get("decisions") or [] if d}
    attribution = (steering or {}).get("attribution") or {}
    cross = (steering or {}).get("crossPatch") or {}
    captured_steps = {p["step"] for p in report.get("prompts", [])}
    probe_prompts = [
        p
        for p in decision_prompts(include_probes=True)
        if p["kind"] != "base" and p["step"] in captured_steps
    ]

    pairs_out = []
    for pair in PAIRS:
        pid = pair["id"]
        steps = {side: f"{pid}_{side.upper()}" for side in SIDES}
        active = {side: active_by_step.get(steps[side], []) for side in SIDES}
        analysis = pair_feature_analysis(active["a"], active["b"], labels)
        sides_out = {}
        for side in SIDES:
            other = "b" if side == "a" else "a"
            rows = analysis[f"{side}Features"]
            causal_block = (attribution.get(pid) or {}).get(side)
            withheld = None
            choice = decisions.get(steps[side])
            if causal_block:
                hf_choice = causal_block.get("hfChoice")
                vllm_choice = (choice or {}).get("display")
                if hf_choice and vllm_choice and hf_choice != vllm_choice:
                    withheld = {
                        "reason": "hf_baseline_disagrees",
                        "hfChoice": hf_choice,
                        "vllmChoice": vllm_choice,
                    }
                else:
                    attach_causal(rows, causal_block)
            sides_out[side] = {
                "step": steps[side],
                "request": pair[side]["request"],
                "cueWords": list(pair[side].get("cueWords", [])),
                "sweep": pair[side].get("sweep"),
                "modelChoice": choice,
                "numActive": analysis["numActive"][side],
                "features": rows,
                "otherSide": other,
            }
            if causal_block and withheld is None:
                sides_out[side]["attribution"] = {
                    k: v for k, v in causal_block.items() if k != "rows"
                }
            elif withheld:
                sides_out[side]["causalWithheld"] = withheld
            probes = _probe_block(
                pid, side, rows, active, active_by_step, decisions, choice, probe_prompts
            )
            if probes:
                sides_out[side]["probes"] = probes
        pairs_out.append(
            {
                "id": pid,
                "title": pair["title"],
                "cue": pair["cue"],
                "a": sides_out["a"],
                "b": sides_out["b"],
                "shared": analysis["shared"],
                "overlap": analysis["overlap"],
                "flipped": (
                    (decisions.get(steps["a"]) or {}).get("display")
                    != (decisions.get(steps["b"]) or {}).get("display")
                    if decisions.get(steps["a"]) and decisions.get(steps["b"])
                    else None
                ),
                "crossPatch": _cross_patch_with_parity(cross.get(pid), decisions, steps),
                **trace_for_pair(trace, pid, steps),
            }
        )

    run_metadata = {
        "model": Path(model_name or report.get("model", "")).name,
        "saeLayer": layer,
        "backend": report.get("backend", "vllm"),
        "logprobsMode": report.get("logprobs_mode"),
        "dSae": block.get("d_sae"),
        "availableLayers": [b["layer"] for b in report.get("layers", [])] or [layer],
    }
    hf_summary = hr.hf_parity_summary(steering)
    if hf_summary:
        run_metadata["hf"] = hf_summary
    if trace:
        run_metadata["trace"] = {
            "compareLayers": list(trace.get("compareLayers") or []),
            "scales": list(trace.get("scales") or []),
            "genTokens": trace.get("genTokens"),
            "controlDraws": trace.get("controlDraws"),
            "hfFastPath": trace.get("hfFastPath"),
        }
    return {
        "runMetadata": run_metadata,
        "pairSelection": PAIR_SELECTION,
        "scenario": {
            "systemPrompt": SYSTEM_PROMPT,
            "tools": [{"id": t["name"], "description": t["description"]} for t in TOOLS],
            "prefill": DECISION_PREFILL,
        },
        "pairs": pairs_out,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario to build for (default: tool_selection); reads demo/<scenario>/.",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--steering", default=None)
    parser.add_argument("--trace", default=None, help="trace_results.json from trace_pairs.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()
    if args.scenario:
        configure(args.scenario)
    report_path = args.report or DEMO_DIR / "output" / "capture" / "evaluation.json"
    output_path = args.output or DEMO_DIR / "output" / "ui_data.json"
    report = json.loads(Path(report_path).read_text())
    steering = json.loads(Path(args.steering).read_text()) if args.steering else None
    trace = json.loads(Path(args.trace).read_text()) if args.trace else None
    ui_data = build_ui_data(report, steering, layer=args.layer, trace=trace)
    ui_data = hr.attach_spec_sheet(
        ui_data, SCENARIO_NAME, int(ui_data["runMetadata"]["saeLayer"])
    )
    args.output = str(output_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(ui_data, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
