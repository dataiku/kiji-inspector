#!/usr/bin/env python3
"""
Home Repair Agent Demo with SAE Activation Analysis.

Orchestrates a home repair advisor using NVIDIA Nemotron-3.5-Nano-30B:
  1. For each repair problem (dishwasher, disposal, water heater), capture one
     training-compatible initial tool-choice representation, then gather and
     analyze four scripted evidence sources
  2. After all data is gathered, an evidence-grounded final recommendation is
     assembled from the tool records
  3. Tool-choice prompts are formatted exactly like the training prompts and
     activations are captured entering each trained transformer layer
  4. Activations are mapped through a trained SAE to explain the reasoning

Uses HuggingFace transformers for both generation and activation capture
(no vLLM required). Single model instance serves both purposes.

By default, the demo uses the completed local six-layer experiment when it is
available under ``output/``. Otherwise it falls back to the matching
575-lab HuggingFace repository.

Prerequisites:
    pip install 'kiji-inspector[huggingface]'
    huggingface-cli login

Usage:
    uv run python demo/home_repair/home_repair_demo.py
    uv run python demo/home_repair/home_repair_demo.py \
      --model-name /path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-no-mtp
    uv run python demo/home_repair/home_repair_demo.py --youtube-api-key YOUR_KEY
    uv run python demo/home_repair/home_repair_demo.py --sae-layer 27
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import torch

from kiji_inspector.core.sae import SAE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
_SCENARIO_PATH = _DEMO_DIR / "home_repair.json"
_EXPERIMENT_OUTPUT_DIR = _REPO_ROOT / "output"

_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16"
_SAE_REPO_ID = "575-lab/kiji-inspector-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_SAE_LAYER = 27
_TRAINED_LAYERS = [6, 13, 20, 27, 34, 43]
_HF_THRESHOLD_OFFSET = 1.12890625

with open(_SCENARIO_PATH) as _scenario_file:
    _SCENARIO = json.load(_scenario_file)

_SYSTEM_PROMPT = _SCENARIO["system_prompt"]
_DECISION_TOOLS = _SCENARIO["tools"]

_DEFAULT_SAE_LOCAL_DIR = (
    str(_EXPERIMENT_OUTPUT_DIR)
    if (
        _EXPERIMENT_OUTPUT_DIR / f"layer_{_SAE_LAYER}" / "sae_checkpoints" / "sae_final.pt"
    ).is_file()
    else None
)


def _strip_thinking(text: str) -> str:
    """Remove complete or prompt-opened reasoning blocks from decoded output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


_PROBLEMS = [
    {
        "id": "dishwasher_leak",
        "summary": "Dishwasher leaking water from the bottom",
        "appliance": "Bosch 500 Series dishwasher",
        "age": "3 years",
        "details": (
            "Water pools under the front of the unit about 15 minutes into "
            "the wash cycle. No error codes on the display."
        ),
        # Requests state the situation and make an explicit ask.  The model's
        # first tool choice follows the ask (verified with sweep_tool_choice.py:
        # open questions go to manual_check regardless of hazard), so the base
        # request asks for what a homeowner in that situation would ask for,
        # and the contrast keeps the situation but changes the ask so the
        # decision flips.
        "initial_decision": {
            "tool": "PartsSearch",
            "request": (
                "My 3-year-old dishwasher leaks from the bottom because the door "
                "gasket is cracked; find me a replacement gasket with price and "
                "availability."
            ),
        },
        "contrast": {
            "changed": "ask: professional repair quote instead of a replacement part",
            "request": (
                "My 3-year-old dishwasher leaks from the bottom because the door "
                "gasket is cracked; get me a professional repair quote for it."
            ),
        },
        # The situation alone, no ask: the one prompt where the tool choice is
        # an open decision rather than a restatement of the request.
        "open": {
            "request": (
                "My 3-year-old dishwasher leaks from the bottom because the door gasket is cracked."
            ),
        },
        # Same meaning, different words (no "gasket", "replacement", "find",
        # "price", "availability"): do the same features fire?
        "paraphrases": [
            "The rubber seal on my three-year-old dishwasher door has split and water "
            "pools on the floor; look up a new seal for it, what it costs, and whether "
            "it's in stock.",
            "Water escapes from under my dishwasher because the door's rubber lining is "
            "torn; source a matching new seal with pricing and how soon it can ship.",
            "My dishwasher drips from the bottom; the seal around the door is cracked, "
            "so how much would a new door seal cost and can I order one now?",
        ],
        # The word without the meaning: do the features that word names fire?
        "controls": [
            {
                "direction": "added",
                "keyword": "warranty",
                "targetWords": ("warranty", "warranties"),
                "note": "mentions the warranty (expired) without asking for warranty service",
                "request": (
                    "My 3-year-old dishwasher leaks from the bottom because the door "
                    "gasket is cracked and the warranty expired ages ago; find me a "
                    "replacement gasket with price and availability."
                ),
            },
            {
                "direction": "added",
                "keyword": "professional",
                "targetWords": ("professional", "quote", "quotes"),
                "note": "mentions a professional without asking for professional repair",
                "request": (
                    "My 3-year-old dishwasher leaks from the bottom because the door "
                    "gasket is cracked, as a professional plumber confirmed; find me a "
                    "replacement gasket with price and availability."
                ),
            },
        ],
    },
    {
        "id": "disposal_stuck",
        "summary": "Garbage disposal hums but won't spin",
        "appliance": "InSinkErator Badger 5",
        "age": "2 years",
        "details": (
            "Motor hums when the switch is flipped but blades don't turn. "
            "Was working fine yesterday. No unusual smell."
        ),
        "initial_decision": {
            "tool": "TutorialSearch",
            "request": (
                "My garbage disposal hums but won't spin; show me a step-by-step "
                "video on how to unjam it with the hex wrench."
            ),
        },
        "contrast": {
            "changed": "ask: professional repair quote instead of a DIY video",
            "request": (
                "My garbage disposal hums but won't spin; get me a professional "
                "repair quote for clearing the jam."
            ),
        },
        # The situation alone, no ask: the one prompt where the tool choice is
        # an open decision rather than a restatement of the request.
        "open": {
            "request": ("My garbage disposal hums but won't spin."),
        },
        "paraphrases": [
            "The food disposer under my sink buzzes when switched on but nothing turns; "
            "walk me through a how-to clip for freeing it with the Allen key from "
            "underneath.",
            "My sink's waste grinder makes a droning noise and the blades are stuck; "
            "I'd like a tutorial that shows how to release it by hand.",
            "Garbage disposal is jammed, motor noise but no rotation; find a "
            "beginner-friendly instructional video for clearing it.",
        ],
        "controls": [
            {
                "direction": "added",
                "keyword": "gas",
                "targetWords": ("gas",),
                "note": "adds the word gas in an unrelated sense",
                "request": (
                    "My garbage disposal hums but won't spin; it started right after the "
                    "gas company visited, so show me a step-by-step video on how to unjam "
                    "it with the hex wrench."
                ),
            },
            {
                "direction": "added",
                "keyword": "professional",
                "targetWords": ("professional", "quote", "quotes"),
                "note": "mentions a professional without asking for professional repair",
                "request": (
                    "My garbage disposal hums but won't spin; my neighbour is a "
                    "professional plumber but he's away, so show me a step-by-step video "
                    "on how to unjam it with the hex wrench."
                ),
            },
        ],
    },
    {
        "id": "water_heater_noise",
        "summary": "Water heater making loud popping and rumbling sounds",
        "appliance": "Rheem 50-gallon gas water heater",
        "age": "9 years",
        "details": (
            "Loud popping when heating up, especially in the morning. "
            "Hot water takes longer to reach faucets. Slight rust tinge "
            "in the first few seconds of hot water."
        ),
        "initial_decision": {
            "tool": "ProQuote",
            "request": (
                "My gas water heater is 9 years old, pops and rumbles loudly, and the "
                "water runs rusty; I want a licensed plumber's quote."
            ),
        },
        "contrast": {
            "changed": "ask: DIY flushing video instead of a plumber's quote",
            "request": (
                "My gas water heater is 9 years old, pops and rumbles loudly, and the "
                "water runs rusty; show me a video on flushing it myself."
            ),
        },
        # The situation alone, no ask: the one prompt where the tool choice is
        # an open decision rather than a restatement of the request.
        "open": {
            "request": (
                "My gas water heater is 9 years old, pops and rumbles loudly, and the "
                "water runs rusty."
            ),
        },
        "paraphrases": [
            "Our nine-year-old gas-fired hot water tank bangs and growls when it fires "
            "up and the hot water comes out brown; I'd like a certified plumber to price "
            "the repair.",
            "Our basement's gas hot water tank is nearly a decade old, makes loud knocking "
            "noises, and the water is discoloured; how much would a professional charge to "
            "fix it?",
            "My gas hot water heater (9 yrs) thumps and grumbles while heating and the "
            "water has an orange tinge; get me an estimate from a licensed tradesperson.",
        ],
        "controls": [
            {
                "direction": "removed",
                "keyword": "gas",
                "targetWords": ("gas",),
                "note": "drops the word gas; the situation is otherwise identical",
                "request": (
                    "My water heater is 9 years old, pops and rumbles loudly, and the "
                    "water runs rusty; I want a licensed plumber's quote."
                ),
            },
            {
                "direction": "added",
                "keyword": "video",
                "targetWords": ("video", "videos", "guide", "guides", "tutorial", "tutorials"),
                "note": "mentions a video without asking for one",
                "request": (
                    "My gas water heater is 9 years old, pops and rumbles loudly, and the "
                    "water runs rusty; I watched a video about it last night, so I want a "
                    "licensed plumber's quote."
                ),
            },
        ],
    },
]

# Tool names are snake_case in the scenario/prompt (``parts_search``) but the
# demo's mock-data registry and UI use CamelCase (``PartsSearch``).
_TOOL_ID_TO_DISPLAY = {
    "manual_check": "ManualCheck",
    "parts_search": "PartsSearch",
    "tutorial_search": "TutorialSearch",
    "pro_quote": "ProQuote",
}
_DISPLAY_TO_TOOL_ID = {display: tool_id for tool_id, display in _TOOL_ID_TO_DISPLAY.items()}


def tool_display(tool_id: str) -> str:
    """Map a scenario tool id (``parts_search``) to the UI name (``PartsSearch``)."""
    return _TOOL_ID_TO_DISPLAY.get(tool_id, tool_id)


def decision_prompts(include_contrasts: bool = False, include_probes: bool = False) -> list[dict]:
    """Return the decision prompts the demo captures, in a fixed order.

    Base prompts come first (one ``{pid}_InitialDecision`` per problem), then
    optionally one ``{pid}_Contrast`` variant per problem, then optionally the
    probes: paraphrases (``{pid}_Paraphrase{n}``, same meaning in other words)
    and keyword controls (``{pid}_Control{n}``, a word added or removed without
    changing the meaning), and finally the situation-only request
    (``{pid}_Open``: no ask at all).  ``tool``/``toolId`` name the tool the
    request's ask describes (``None`` for the open request) — they are an input
    to the design, not a prediction; the model's choice is read out at run
    time.  Every producer (HF demo, vLLM evaluation) and the UI builder use
    this single definition so the captured activations, tool-choice readouts
    and the page stay aligned.
    """
    prompts = []
    for problem in _PROBLEMS:
        tool_name, request = _initial_tool_decision(problem)
        prompts.append(
            {
                "step": f"{problem['id']}_InitialDecision",
                "problem": problem["id"],
                "kind": "base",
                "tool": tool_name,
                "toolId": _DISPLAY_TO_TOOL_ID.get(tool_name, tool_name),
                "request": request,
            }
        )
    if include_contrasts:
        for problem in _PROBLEMS:
            contrast = problem.get("contrast")
            if not contrast:
                continue
            tool_name, _ = _initial_tool_decision(problem)
            prompts.append(
                {
                    "step": f"{problem['id']}_Contrast",
                    "problem": problem["id"],
                    "kind": "contrast",
                    "tool": tool_name,
                    "toolId": _DISPLAY_TO_TOOL_ID.get(tool_name, tool_name),
                    "request": contrast["request"],
                    "changed": contrast["changed"],
                }
            )
    if include_probes:
        for problem in _PROBLEMS:
            tool_name, _ = _initial_tool_decision(problem)
            for number, request in enumerate(problem.get("paraphrases", []), start=1):
                prompts.append(
                    {
                        "step": f"{problem['id']}_Paraphrase{number}",
                        "problem": problem["id"],
                        "kind": "paraphrase",
                        "tool": tool_name,
                        "toolId": _DISPLAY_TO_TOOL_ID.get(tool_name, tool_name),
                        "request": request,
                    }
                )
        for problem in _PROBLEMS:
            tool_name, _ = _initial_tool_decision(problem)
            for number, control in enumerate(problem.get("controls", []), start=1):
                prompts.append(
                    {
                        "step": f"{problem['id']}_Control{number}",
                        "problem": problem["id"],
                        "kind": "control",
                        "tool": tool_name,
                        "toolId": _DISPLAY_TO_TOOL_ID.get(tool_name, tool_name),
                        "request": control["request"],
                        "direction": control["direction"],
                        "keyword": control["keyword"],
                        "targetWords": list(control["targetWords"]),
                        "note": control["note"],
                    }
                )
        for problem in _PROBLEMS:
            open_request = problem.get("open")
            if not open_request:
                continue
            prompts.append(
                {
                    "step": f"{problem['id']}_Open",
                    "problem": problem["id"],
                    "kind": "open",
                    "tool": None,
                    "toolId": None,
                    "request": open_request["request"],
                }
            )
    return prompts


# ---------------------------------------------------------------------------
# Section A: HuggingFace generation + extraction engine
# ---------------------------------------------------------------------------


def hf_fast_path_status(model) -> bool | None:
    """Whether the loaded HF model runs its Mamba layers on the fused kernels.

    transformers' Mamba/NemotronH modeling modules expose a module-level
    ``is_fast_path_available`` flag that is True only when the ``causal-conv1d``
    and ``mamba-ssm`` kernels resolved (via the ``kernels`` hub package or the
    pip packages). Without them the naive PyTorch scan runs, and for this model
    the residual stream drifts from vLLM (cosine ≈ 0.92–0.95 vs ≈ 0.99+ with the
    kernels). Returns ``None`` for architectures without such a flag.
    """
    import importlib
    import sys

    seen: set[str] = set()
    for module in model.modules():
        name = type(module).__module__
        if name in seen:
            continue
        seen.add(name)
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
        flag = getattr(mod, "is_fast_path_available", None)
        if isinstance(flag, bool):
            return flag
    return None


def hf_parity_summary(steering: dict | None) -> dict | None:
    """Headline HF-vs-vLLM parity numbers from a steering results file.

    Returns ``{"fastPath", "prompts", "cosineMean", "cosineMin",
    "baselineAgreement"}`` (agreement = prompts whose HF and vLLM argmax tools
    coincide) or ``None`` when there is no parity block.
    """
    if not steering or not steering.get("parity"):
        return None
    cosines: list[float] = []
    agree = 0
    compared = 0
    for entry in steering["parity"].values():
        if entry.get("cosine") is not None:
            cosines.append(float(entry["cosine"]))
        hf = entry.get("baselineDistributionHf") or {}
        vllm = entry.get("baselineDistributionVllm") or {}
        if hf and vllm:
            compared += 1
            agree += int(max(hf, key=hf.get) == max(vllm, key=vllm.get))
    return {
        "fastPath": steering.get("hfFastPath"),
        "prompts": len(steering["parity"]),
        "cosineMean": round(sum(cosines) / len(cosines), 4) if cosines else None,
        "cosineMin": round(min(cosines), 4) if cosines else None,
        "baselineAgreement": agree if compared else None,
        "baselineCompared": compared,
    }


_SPEC_SHEET_UI = Path(__file__).resolve().parent.parent / "spec_sheet" / "output" / "ui_data.json"


def spec_sheet_note(spec: dict | None, scenario: str, probe_layer: int) -> dict | None:
    """Condense ``demo/spec_sheet`` ui_data into the strip a demo page shows.

    Returns None when nothing relevant is present, so pages built before the
    spec sheet ran render unchanged.
    """
    if not spec:
        return None
    note: dict = {}
    depth = spec.get("depth") or {}
    if depth:
        note["depth"] = [
            {
                "layer": int(layer),
                "beyond": entry.get("sidesAllBeyondControl"),
                "sides": entry.get("nSides"),
                "crossFlips": entry.get("crossFlips"),
            }
            for layer, entry in sorted(depth.items(), key=lambda kv: int(kv[0]))
        ]
    matching = (
        ((spec.get("transfer") or {}).get("layers") or {}).get("43", {}).get("matching") or {}
    ).get("joint->joint_seed123") or {}
    cut = matching.get("rateAtLeast0.01")
    if cut:
        note["stability"] = {
            "decoderFrac07": cut["decoder"]["fracAtLeast07"],
            "functionalFrac07": cut["functional"]["fracAtLeast07"],
        }
    workbench = spec.get("workbench") or {}
    probes = (
        ((workbench.get("layers") or {}).get(str(probe_layer)) or {})
        .get("probes", {})
        .get(scenario)
    )
    bow = (workbench.get("bow") or {}).get(scenario)
    if probes and bow:
        note["probes"] = {
            "layer": probe_layer,
            "sae": probes["saeFeatures"]["accuracy"],
            "residual": probes["residual"]["accuracy"],
            "bow": bow["accuracy"],
        }
    if scenario == "tool_selection":
        population = (spec.get("population") or {}).get("overall")
        if population:
            note["population"] = {
                "pairs": population["pairs"],
                "flipping": population["flipping"],
                "flipAtLeast06": population["flipAtLeast06"],
            }
        robustness = spec.get("robustness") or {}
        if robustness:
            note["robustness"] = [
                {
                    "dictionary": name,
                    "beyond": entry.get("sidesAllBeyondControl"),
                    "sides": entry.get("nSides"),
                    "crossFlips": entry.get("crossFlips"),
                }
                for name, entry in sorted(robustness.items())
            ]
    if not note:
        return None
    note["link"] = "../spec_sheet/index.html"
    return note


def attach_spec_sheet(
    ui_data: dict, scenario: str, probe_layer: int, path: Path | None = None
) -> dict:
    """Attach ``specSheet`` to ``ui_data`` when the spec-sheet results exist."""
    spec_path = Path(path) if path else _SPEC_SHEET_UI
    if spec_path.exists():
        note = spec_sheet_note(json.loads(spec_path.read_text()), scenario, probe_layer)
        if note:
            ui_data["specSheet"] = note
    return ui_data


class HFEngine:
    """Single HuggingFace model for text generation AND activation extraction.

    Generation uses model.generate(). Activation extraction uses a separate
    forward pass through the transformer body with hooks registered on target
    layers. Hooks are only active during extraction, not during generation.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 400,
        allow_thinking: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from kiji_inspector.extraction.vllm_activation_extractor import (
            recommended_chat_template_kwargs,
        )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.prompt_log: list[tuple[str, str]] = []

        torch_dtype = getattr(torch, dtype)

        print(f"  Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_template_kwargs = (
            {} if allow_thinking else recommended_chat_template_kwargs(model_name, self.tokenizer)
        )

        load_kwargs: dict = {
            "dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        self._input_device = self._find_input_device()

        # Resolve hidden_size (some models nest under text_config)
        self.hidden_size = (
            getattr(self.model.config, "hidden_size", None)
            or getattr(self.model.config, "text_config", self.model.config).hidden_size
        )

        self.fast_path = hf_fast_path_status(self.model)
        if self.fast_path is False:
            print(
                "  WARNING: the HuggingFace Mamba fast path is unavailable (naive fallback). "
                "Residuals and decisions will drift from vLLM; install the fused kernels "
                '(`pip install "kernels>=0.15.2,<0.16"`) before trusting interventions.'
            )

        print(f"  Model ready on {self._input_device} ({torch_dtype})")
        print(f"  hidden_size: {self.hidden_size}")
        if self.fast_path is not None:
            print(f"  mamba fast path: {self.fast_path}")

    # --- Device helpers ---

    def _find_input_device(self) -> torch.device:
        for attr_path in (
            "language_model.model.embed_tokens",
            "language_model.embed_tokens",
            "model.embed_tokens",
            "model.language_model.embed_tokens",
            "backbone.embed_tokens",
            "backbone.embedding",
            "transformer.wte",
            "model.embed_in",
        ):
            obj = self.model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return next(obj.parameters()).device
            except (AttributeError, StopIteration):
                continue
        return next(self.model.parameters()).device

    # --- Layer resolution (mirrors ActivationExtractor) ---

    def _get_model_layers(self):
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
            if hasattr(lm, "layers"):
                return lm.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            inner_lm = self.model.model.language_model
            if hasattr(inner_lm, "layers"):
                return inner_lm.layers
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone
            if hasattr(backbone, "layers"):
                return backbone.layers
            if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "layers"):
                return backbone.decoder.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            if hasattr(self.model.model.decoder, "layers"):
                return self.model.model.decoder.layers
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            return self.model.decoder.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError(f"Cannot locate transformer layers for {type(self.model).__name__}")

    def _get_inner_model(self):
        """Get transformer body, skipping lm_head to avoid logit allocation."""
        for attr in ("language_model", "model", "backbone", "transformer"):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        return self.model

    # --- Chat template ---

    def _build_prompt(self, system: str, user: str) -> str:
        """Build prompt via the tokenizer's chat template."""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.generation_template_kwargs,
            )
        except Exception:
            # Some models may not support system role -- prepend to user message
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.generation_template_kwargs,
            )

    def record_tool_decision(self, step_label: str, user_request: str) -> None:
        """Record a training-compatible prompt for later SAE extraction.

        The SAE was trained on the final token of prompts ending in
        ``I'll use the``. Reusing the pipeline prompt builder keeps the demo's
        system message, tool inventory, chat template, and decision position
        aligned with that training distribution.
        """
        from kiji_inspector.extraction.extractor import build_agent_prompt
        from kiji_inspector.extraction.vllm_activation_extractor import (
            recommended_chat_template_kwargs,
        )

        template_kwargs = recommended_chat_template_kwargs(self.model_name, self.tokenizer)

        prompt = build_agent_prompt(
            system_prompt=_SYSTEM_PROMPT,
            tools=_DECISION_TOOLS,
            user_request=user_request,
            tokenizer=self.tokenizer,
            chat_template_kwargs=template_kwargs,
            close_think_block=bool(template_kwargs),
        )
        self.prompt_log.append((step_label, prompt))

    # --- Generation ---

    def generate(self, prompt: str, step_label: str, max_tokens: int | None = None) -> str:
        """Generate text. Tool-decision prompts are recorded separately."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Some reasoning templates represent the opening <think> marker as a
        # prompt token that is omitted by skip_special_tokens, while the model
        # still emits a literal closing marker. Remove either a complete block
        # or an unmatched reasoning prefix so it cannot leak into demo output.
        response = _strip_thinking(response)
        print(f"  [{step_label}] Generated {len(new_tokens)} tokens")
        return response

    # --- Activation extraction ---

    def extract_all_prompts(self, layers: list[int]) -> list[tuple[str, dict[str, np.ndarray]]]:
        """Extract last-token activations for every logged prompt.

        Registers forward pre-hooks on the target layers, runs a single forward
        pass per prompt through the transformer body, then removes all hooks.

        A pre-hook captures the residual stream entering layer N, matching the
        vLLM auxiliary hidden-state convention used to train the SAEs. Capturing
        layer N's output would be off by one layer.
        """
        model_layers = self._get_model_layers()
        activations: dict[str, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def _make_hook(name: str):
            def hook(module, args, kwargs):
                act = args[0] if args else kwargs["hidden_states"]
                activations[name] = act.detach().cpu().to(torch.float32)

            return hook

        for idx in layers:
            if not 0 <= idx < len(model_layers):
                print(
                    f"  WARNING: layer {idx} requested but model has "
                    f"{len(model_layers)} layers; skipping"
                )
                continue
            h = model_layers[idx].register_forward_pre_hook(
                _make_hook(f"residual_{idx}"), with_kwargs=True
            )
            hooks.append(h)

        if not hooks:
            raise ValueError("None of the requested activation layers exist in the model")

        inner = self._get_inner_model()
        results: list[tuple[str, dict[str, np.ndarray]]] = []

        for step_label, prompt in self.prompt_log:
            activations.clear()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
            with torch.inference_mode():
                inner(**inputs)

            result = {name: act[:, -1, :].squeeze(0).numpy() for name, act in activations.items()}
            results.append((step_label, result))
            print(f"    {step_label}: {len(result)} layers extracted")

        for h in hooks:
            h.remove()

        return results

    # --- Cleanup ---

    def cleanup(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Model released.")


# ---------------------------------------------------------------------------
# Section B: Mock tool data + optional YouTube API
# ---------------------------------------------------------------------------

_MANUAL_DATA = {
    "dishwasher_leak": {
        "model": "Bosch 500 Series SHP65T55UC",
        "possible_causes": [
            "Door gasket worn or cracked -- inspect for debris buildup",
            "Pump seal failure (common at 2-3 years of heavy use)",
            "Water inlet valve connections loose or corroded",
            "Spray arm seal worn -- water escapes during rotation",
        ],
        "diy_difficulty": "Moderate",
        "safety": "Disconnect power at breaker and shut water supply valve before inspection.",
        "tools_needed": ["Phillips screwdriver", "Torx T20 driver", "flashlight", "towels"],
    },
    "disposal_stuck": {
        "model": "InSinkErator Badger 5",
        "possible_causes": [
            "Jammed flywheel -- foreign object wedged between impellers",
            "Thermal overload tripped (red reset button on bottom of unit)",
            "Capacitor failure -- motor hums but cannot start rotation (less common)",
        ],
        "diy_difficulty": "Easy",
        "safety": "NEVER put hand inside disposal. Ensure power is OFF at breaker before clearing a jam.",
        "tools_needed": ["1/4-inch Allen wrench (hex key)", "flashlight", "tongs or pliers"],
        "quick_fix": (
            "Insert 1/4-inch Allen wrench into the hex socket on the bottom center "
            "of the unit. Rotate back and forth to free the jam. Remove debris with "
            "tongs. Press the red reset button. Restore power and test."
        ),
    },
    "water_heater_noise": {
        "model": "Rheem Performance 50-Gal Gas XR50T06EC36U1",
        "possible_causes": [
            "Sediment buildup on tank bottom -- mineral deposits from hard water heat and pop",
            "Anode rod depleted -- sacrificial rod no longer protecting tank lining",
            "Scale buildup on burner assembly reducing heat transfer efficiency",
            "Possible tank corrosion if popping is severe and rust is present in water",
        ],
        "diy_difficulty": "Moderate to Difficult",
        "safety": (
            "GAS APPLIANCE: Risk of scalding burns from hot water and gas leaks if "
            "connections are disturbed. Turn off gas supply valve before any work. "
            "If you smell gas, leave immediately and call your gas utility."
        ),
        "tools_needed": [
            "Garden hose",
            "1-1/16 inch anode rod socket",
            "pipe wrench",
            "Teflon tape",
        ],
    },
}

_PARTS_DATA = {
    "dishwasher_leak": {
        "parts": [
            {
                "name": "Door Gasket Seal (OEM)",
                "part_no": "00744367",
                "price": 42.99,
                "in_stock": True,
            },
            {
                "name": "Drain Pump Assembly",
                "part_no": "00631200",
                "price": 89.50,
                "in_stock": True,
            },
            {"name": "Water Inlet Valve", "part_no": "00622058", "price": 55.75, "in_stock": True},
            {"name": "Spray Arm Seal Kit", "part_no": "00165259", "price": 12.99, "in_stock": True},
        ],
        "diy_cost_range": "$13 - $90 depending on which part has failed",
    },
    "disposal_stuck": {
        "parts": [
            {
                "name": "Self-Service Wrench Kit",
                "part_no": "WRN-00",
                "price": 7.99,
                "in_stock": True,
            },
            {
                "name": "Badger 5 Replacement Unit (if motor failed)",
                "part_no": "?",
                "price": 99.00,
                "in_stock": True,
            },
        ],
        "diy_cost_range": "$0 - $8 if jam clears; $99 + install if motor is dead",
    },
    "water_heater_noise": {
        "parts": [
            {
                "name": "Aluminum Anode Rod (Rheem-compatible)",
                "part_no": "SP11526",
                "price": 29.99,
                "in_stock": True,
            },
            {
                "name": "Tank Flush Kit (hose + valve adapter)",
                "part_no": "FK-100",
                "price": 14.99,
                "in_stock": True,
            },
            {
                "name": "Drain Valve Replacement",
                "part_no": "SP12112",
                "price": 11.49,
                "in_stock": True,
            },
            {
                "name": "Rheem 50-Gal Replacement Unit (if tank is corroded)",
                "part_no": "XG50T06EC36U1",
                "price": 649.00,
                "in_stock": True,
            },
        ],
        "diy_cost_range": "$15 - $45 for maintenance parts; $649+ if tank replacement needed",
    },
}

_TUTORIAL_DATA = {
    "dishwasher_leak": {
        "source": "mock",
        "results": [
            {
                "title": "How to Fix a Leaking Dishwasher - 5 Most Common Causes",
                "channel": "RepairClinic",
                "views": "1.2M",
                "duration": "12:34",
                "difficulty": "Beginner-Intermediate",
            },
            {
                "title": "Bosch Dishwasher Door Gasket Replacement",
                "channel": "AppliancePartsPros",
                "views": "340K",
                "duration": "8:15",
                "difficulty": "Beginner",
            },
            {
                "title": "Dishwasher Pump Seal: When to Replace vs Repair",
                "channel": "FixItHome",
                "views": "89K",
                "duration": "15:02",
                "difficulty": "Intermediate",
            },
        ],
    },
    "disposal_stuck": {
        "source": "mock",
        "results": [
            {
                "title": "Garbage Disposal Humming But Not Working? Easy Fix!",
                "channel": "HomeRepairTutor",
                "views": "2.8M",
                "duration": "4:22",
                "difficulty": "Beginner",
            },
            {
                "title": "How to Unjam a Garbage Disposal in 60 Seconds",
                "channel": "ThisOldHouse",
                "views": "1.5M",
                "duration": "3:10",
                "difficulty": "Beginner",
            },
            {
                "title": "InSinkErator Reset Button and Allen Wrench Fix",
                "channel": "DIYWithMike",
                "views": "620K",
                "duration": "5:45",
                "difficulty": "Beginner",
            },
        ],
    },
    "water_heater_noise": {
        "source": "mock",
        "results": [
            {
                "title": "Water Heater Making Noise? Here's Why and How to Fix It",
                "channel": "RogerWakefield",
                "views": "890K",
                "duration": "18:30",
                "difficulty": "Intermediate-Advanced",
            },
            {
                "title": "How to Flush a Water Heater (Step by Step)",
                "channel": "ThisOldHouse",
                "views": "3.1M",
                "duration": "10:15",
                "difficulty": "Intermediate",
            },
            {
                "title": "Replacing a Water Heater Anode Rod - Is It Worth It?",
                "channel": "TechDIY",
                "views": "450K",
                "duration": "14:20",
                "difficulty": "Intermediate",
            },
        ],
    },
}

_PRO_QUOTE_DATA = {
    "dishwasher_leak": {
        "diagnosis_fee": 89,
        "repair_estimates": [
            {
                "repair": "Door gasket replacement",
                "labor": 120,
                "parts": 43,
                "total": 163,
                "time": "1 hour",
            },
            {
                "repair": "Pump seal replacement",
                "labor": 180,
                "parts": 90,
                "total": 270,
                "time": "1.5 hours",
            },
            {
                "repair": "Inlet valve replacement",
                "labor": 150,
                "parts": 56,
                "total": 206,
                "time": "1 hour",
            },
        ],
        "warranty_on_repair": "90-day parts and labor",
        "urgency": "Moderate -- continued use risks water damage to flooring",
        "next_available": "2-3 business days",
    },
    "disposal_stuck": {
        "diagnosis_fee": 75,
        "repair_estimates": [
            {"repair": "Clear jam + reset", "labor": 75, "parts": 0, "total": 75, "time": "30 min"},
            {
                "repair": "Full unit replacement (Badger 5)",
                "labor": 150,
                "parts": 99,
                "total": 249,
                "time": "1.5 hours",
            },
        ],
        "warranty_on_repair": "90-day labor, manufacturer warranty on new unit",
        "urgency": "Low -- disposal is non-essential; sink still drains",
        "next_available": "3-5 business days",
    },
    "water_heater_noise": {
        "diagnosis_fee": 95,
        "repair_estimates": [
            {
                "repair": "Tank flush + anode rod replacement",
                "labor": 200,
                "parts": 45,
                "total": 245,
                "time": "2 hours",
            },
            {
                "repair": "Full unit replacement (50-gal gas)",
                "labor": 450,
                "parts": 649,
                "total": 1099,
                "time": "4-6 hours",
            },
        ],
        "warranty_on_repair": "1-year labor, 6-year tank on new unit",
        "urgency": "Moderate-High -- sediment reduces efficiency; tank corrosion risk increases with age",
        "next_available": "1-2 business days (prioritized for gas appliances)",
    },
}

# Tool registry: name -> (data_source, description)
_TOOLS = {
    "ManualCheck": (_MANUAL_DATA, "appliance troubleshooting guide"),
    "PartsSearch": (_PARTS_DATA, "replacement parts and pricing"),
    "TutorialSearch": (_TUTORIAL_DATA, "repair video tutorials"),
    "ProQuote": (_PRO_QUOTE_DATA, "professional repair quotes"),
}


def _initial_tool_decision(problem: dict) -> tuple[str, str]:
    """Return the expected first tool and its training-style user request."""
    decision = problem["initial_decision"]
    return decision["tool"], decision["request"]


def _recommended_path(problem_id: str) -> str:
    """Derive the recommendation category from the authoritative manual."""
    manual = _MANUAL_DATA[problem_id]
    safety = manual["safety"]
    difficulty = manual["diy_difficulty"]
    if "GAS APPLIANCE" in safety or "Difficult" in difficulty:
        return (
            "Hire a professional for diagnosis and any gas-system or tank repair; "
            "only an experienced homeowner should attempt a basic tank flush."
        )
    if difficulty == "Easy":
        return "Try the documented DIY jam-clearing and reset procedure first."
    return (
        "Start with a powered-off DIY inspection and replace only an identified, "
        "accessible part; hire a professional if the leak source is unclear."
    )


_GROUNDED_RATIONALES = {
    "dishwasher_leak": (
        "A leak that begins during the wash cycle can come from the door gasket, "
        "pump seal, inlet-valve connection, or spray-arm seal. Inspect while power "
        "and water are off, then replace only the confirmed failed part."
    ),
    "disposal_stuck": (
        "Humming without rotation most strongly indicates a jammed flywheel; the "
        "manual's first-line fix is the bottom hex socket followed by the reset "
        "button. At two years old, replacement is premature unless that fails."
    ),
    "water_heater_noise": (
        "At nine years, popping plus slower heating strongly suggests sediment, "
        "while the initial rusty water raises anode depletion or tank-corrosion "
        "concerns. The gas and scalding hazards justify professional inspection."
    ),
}


def _render_grounded_final_section(problem: dict) -> str:
    """Render a complete conclusion from the authoritative tool records."""
    pid = problem["id"]
    manual = _MANUAL_DATA[pid]
    parts = _PARTS_DATA[pid]
    quote = _PRO_QUOTE_DATA[pid]
    professional_costs = "; ".join(
        f"{estimate['repair']}: ${estimate['total']}" for estimate in quote["repair_estimates"]
    )
    return (
        f"## {problem['summary']}\n\n"
        f"**Recommendation:** {_recommended_path(pid)}\n\n"
        f"**Why:** {_GROUNDED_RATIONALES[pid]}\n\n"
        f"**Estimated costs:** DIY {parts['diy_cost_range']}. Professional diagnosis "
        f"fee: ${quote['diagnosis_fee']}; {professional_costs}.\n\n"
        f"**Safety:** {manual['safety']}\n\n"
        f"**Urgency:** {quote['urgency']}."
    )


def _render_priority_order() -> str:
    """Order the fixed demo problems using the quote's stated urgency."""
    severity = {"Low": 1, "Moderate": 2, "Moderate-High": 3}
    ordered = sorted(
        _PROBLEMS,
        key=lambda problem: severity.get(
            _PRO_QUOTE_DATA[problem["id"]]["urgency"].split(" --", 1)[0], 0
        ),
        reverse=True,
    )
    lines = []
    for index, problem in enumerate(ordered, start=1):
        urgency = _PRO_QUOTE_DATA[problem["id"]]["urgency"]
        lines.append(f"{index}. **{problem['summary']}** — {urgency}.")
    return "\n".join(lines)


def _fetch_youtube_tutorials(query: str, api_key: str) -> dict:
    """Search YouTube Data API v3 for repair tutorials (real API)."""
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 3,
            "key": api_key,
        }
    )
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())

    return {
        "source": "youtube_api",
        "results": [
            {
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "video_id": item["id"]["videoId"],
                "url": f"https://youtube.com/watch?v={item['id']['videoId']}",
                "description": item["snippet"]["description"][:200],
            }
            for item in data.get("items", [])
            if item.get("id", {}).get("videoId")
        ],
    }


def _get_tool_result(tool_name: str, problem_id: str, youtube_api_key: str | None = None) -> str:
    """Return tool result as JSON. Uses real YouTube API if key is provided."""
    if tool_name == "TutorialSearch" and youtube_api_key:
        problem = next(p for p in _PROBLEMS if p["id"] == problem_id)
        query = f"{problem['appliance']} {problem['summary']} repair tutorial"
        try:
            result = _fetch_youtube_tutorials(query, youtube_api_key)
            print(f"    (YouTube API: {len(result['results'])} results)")
            return json.dumps(result, indent=2)
        except Exception as e:
            print(f"    YouTube API failed ({e}), using mock data")

    source = _TOOLS[tool_name][0]
    data = source.get(problem_id, {"error": f"No data for '{problem_id}'"})
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Section C: Scripted multi-step agent orchestration
# ---------------------------------------------------------------------------


def run_home_repair_analysis(
    engine: HFEngine, youtube_api_key: str | None = None
) -> tuple[str, dict[str, str]]:
    """Run the full multi-step home repair analysis.

    For each problem, calls each tool and asks the model to analyze the results.
    Then assembles a tool-grounded recommendation across all three problems.

    Returns:
        (final_recommendation, per_problem_analyses)
    """
    per_problem_analyses: dict[str, str] = {}
    for problem in _PROBLEMS:
        pid = problem["id"]
        print(f"\n  --- Analyzing: {problem['summary']} ---")
        problem_context = ""

        # Capture one genuine initial tool-choice representation for the
        # problem. The four calls below are scripted evidence gathering, not
        # four separate decisions, so attaching a different SAE explanation to
        # each of them would misrepresent what the activation means.
        expected_tool, decision_request = _initial_tool_decision(problem)
        engine.record_tool_decision(f"{pid}_InitialDecision", decision_request)
        print(f"    Initial tool-choice target: {expected_tool}")

        for tool_name, (_, tool_desc) in _TOOLS.items():
            step_label = f"{pid}_{tool_name}"

            tool_result = _get_tool_result(tool_name, pid, youtube_api_key)

            user_msg = (
                f"A homeowner needs help with: {problem['summary']}\n"
                f"Appliance: {problem['appliance']} ({problem['age']} old)\n"
                f"Details: {problem['details']}\n\n"
            )
            if problem_context:
                user_msg += f"Your analysis so far:\n{problem_context}\n\n"
            user_msg += (
                f"Here is the {tool_desc} data:\n"
                f"```json\n{tool_result}\n```\n\n"
                f"Analyze this {tool_desc} data. Highlight key takeaways, "
                f"safety concerns, and whether this points toward DIY or "
                f"professional repair. Be specific with numbers."
            )

            prompt = engine._build_prompt(_SYSTEM_PROMPT, user_msg)
            analysis = engine.generate(prompt, step_label, max_tokens=300)
            problem_context += f"\n[{tool_name}] {analysis.strip()}\n"

        per_problem_analyses[pid] = problem_context

    # Render every factual decision field from authoritative tool data. The
    # model's per-tool analyses remain available separately, but are not fed
    # back into a free-form synthesis call where facts could be lost or changed.
    print("\n  --- Final Recommendation ---")
    final_sections = [_render_grounded_final_section(problem) for problem in _PROBLEMS]
    rendered_sections = "\n\n".join(final_sections)
    final_rec = f"{rendered_sections}\n\n## Priority order\n\n{_render_priority_order()}"

    return final_rec, per_problem_analyses


# ---------------------------------------------------------------------------
# Section D: Post-run activation extraction + SAE analysis
# ---------------------------------------------------------------------------


def _load_contrastive_feature_map(output_dir: str, layer: int) -> dict[int, list[dict]]:
    """Load contrastive_features.json and build a feature_index -> themes mapping.

    Returns a dict mapping feature index to a list of
    {"theme": str, "rank": int, "cohens_d": float, "direction": str} entries.
    """
    layer_dir = Path(output_dir) / f"layer_{layer}"
    report_paths = (
        layer_dir / "activations" / "contrastive_features.json",
        layer_dir / "contrastive_features.json",  # legacy layout
    )
    report_path = next((path for path in report_paths if path.is_file()), None)
    if report_path is None:
        print(f"  No contrastive feature map found under {layer_dir}")
        return {}

    with open(report_path) as f:
        report = json.load(f)

    feature_map: dict[int, list[dict]] = {}
    for theme, info in report.items():
        if theme.startswith("_"):  # skip _summary
            continue
        for feat in info.get("top_features", []):
            idx = feat["feature_index"]
            anchor_act = feat.get("anchor_mean_activation", 0)
            contrast_act = feat.get("contrast_mean_activation", 0)
            direction = "anchor" if anchor_act > contrast_act else "contrast"
            entry = {
                "theme": theme,
                "rank": feat["rank"],
                "cohens_d": feat["cohens_d"],
                "direction": direction,
            }
            feature_map.setdefault(idx, []).append(entry)

    num_themes = sum(not theme.startswith("_") for theme in report)
    print(
        f"  Loaded contrastive feature map: {len(feature_map)} features across {num_themes} themes"
    )
    return feature_map


def _load_sae_local(
    output_dir: str,
    layer: int,
    device: str = "cpu",
) -> tuple[SAE | None, dict | None]:
    """Load SAE and feature descriptions from local pipeline output."""
    from kiji_inspector.core.sae_core import JumpReLUSAE

    layer_dir = Path(output_dir) / f"layer_{layer}"
    checkpoint = layer_dir / "sae_checkpoints" / "sae_final.pt"
    if not checkpoint.exists():
        print(f"  No local SAE checkpoint at {checkpoint}")
        return None, None

    print(f"  Loading local SAE from {checkpoint}")
    sae = JumpReLUSAE.from_pretrained(str(checkpoint), device=device)
    sae.eval()

    feature_descriptions = None
    desc_path = layer_dir / "activations" / "feature_descriptions.json"
    if desc_path.exists():
        with open(desc_path) as f:
            feature_descriptions = json.load(f)
        print(f"  Loaded {len(feature_descriptions)} feature labels from {desc_path}")

    return sae, feature_descriptions


def _load_sae_from_hub(
    repo_id: str,
    layer: int,
    device: str = "cpu",
) -> tuple[SAE | None, dict | None]:
    """Load SAE and feature descriptions from HuggingFace Hub."""
    try:
        sae, feature_descriptions = SAE.from_pretrained(
            repo_id=repo_id,
            layer=layer,
            device=device,
        )
        return sae, feature_descriptions
    except Exception as e:
        print(f"  Could not load SAE from {repo_id} layer {layer}: {e}")
        return None, None


def analyze_activations(
    activation_log: list[tuple[str, dict[str, np.ndarray]]],
    sae_repo_id: str,
    sae_layer: int,
    layer_key: str | None = None,
    sae_local_dir: str | None = None,
    threshold_offset: float = _HF_THRESHOLD_OFFSET,
) -> dict:
    """Encode captured activations through SAE and map to feature descriptions.

    Three tiers of analysis:
      1. Raw activation statistics (always available)
      2. SAE feature decomposition (if checkpoint found)
      3. Feature label mapping (if descriptions found)
    """
    layer_key = layer_key or f"residual_{sae_layer}"
    results: dict = {
        "steps": [],
        "sae_available": False,
        "features_available": False,
        "sae_layer": sae_layer,
        "sae_layer_key": layer_key,
        "sae_source": sae_local_dir or sae_repo_id,
        "sae_threshold_offset": threshold_offset,
        "backend": "hf",
    }

    # Tier 1: Raw activation statistics
    for step_label, acts in activation_log:
        step_info: dict = {
            "step": step_label,
            "layers_captured": list(acts.keys()),
            "raw_stats": {},
        }
        for layer_name, vec in acts.items():
            step_info["raw_stats"][layer_name] = {
                "mean": float(np.mean(vec)),
                "std": float(np.std(vec)),
                "l2_norm": float(np.linalg.norm(vec)),
                "max_abs": float(np.max(np.abs(vec))),
                "sparsity": float(np.mean(np.abs(vec) < 0.01)),
            }
        results["steps"].append(step_info)

    # Tier 2: SAE feature decomposition (local or HuggingFace Hub)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sae_local_dir:
        sae, feature_descs = _load_sae_local(sae_local_dir, sae_layer, device=device)
    else:
        sae, feature_descs = _load_sae_from_hub(sae_repo_id, sae_layer, device=device)
    if sae is None:
        print("  SAE not available -- showing raw activation stats only.")
        return results

    # Load contrastive feature map (feature_index -> themes from training),
    # restricted to the five home-repair contrast types.
    contrastive_map: dict[int, list[dict]] = {}
    if sae_local_dir:
        contrastive_map = filter_contrastive_map(
            _load_contrastive_feature_map(sae_local_dir, sae_layer)
        )

    results["sae_available"] = True
    results["contrast_themes"] = (
        sorted({entry["theme"] for entries in contrastive_map.values() for entry in entries})
        if contrastive_map
        else []
    )
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype
    if threshold_offset:
        if not hasattr(sae, "threshold"):
            raise AttributeError("Loaded SAE has no threshold parameter to recalibrate.")
        with torch.no_grad():
            sae.threshold.add_(threshold_offset)
        print(
            f"  Applied in-memory HF threshold offset: {threshold_offset:+.7f} "
            "(checkpoint unchanged)"
        )

    for step_info, (_step_label, acts) in zip(results["steps"], activation_log, strict=True):
        if layer_key not in acts:
            continue
        vec = acts[layer_key]
        if vec.shape[-1] != sae.d_model:
            raise ValueError(
                f"Activation {layer_key} has width {vec.shape[-1]}, but the SAE "
                f"expects d_model={sae.d_model}. Check --model-name, --sae-layer, "
                "and the checkpoint source."
            )
        vec_tensor = torch.from_numpy(vec).unsqueeze(0).to(device=device, dtype=sae_dtype)
        with torch.no_grad():
            features = sae.encode(sae.normalize_input(vec_tensor))
        features_np = features.squeeze(0).cpu().float().numpy()

        nonzero_mask = features_np > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = features_np[nonzero_indices]

        sort_order = np.argsort(-nonzero_values)
        top_k = min(20, len(sort_order))
        top_indices = nonzero_indices[sort_order[:top_k]]
        top_values = nonzero_values[sort_order[:top_k]]

        top_features = []
        for idx, val in zip(top_indices, top_values, strict=True):
            feat_entry: dict = {"index": int(idx), "activation": float(val)}
            # Annotate with contrast themes from training
            if int(idx) in contrastive_map:
                feat_entry["themes"] = contrastive_map[int(idx)]
            top_features.append(feat_entry)

        # Every active feature (not just the top-k), with labels when known,
        # plus contrastive-map theme evidence over the full active set.
        active_pairs = [(int(idx), float(features_np[idx])) for idx in nonzero_indices[sort_order]]
        active_features = [
            {
                "index": idx,
                "activation": act,
                "label": _label_for(feature_descs, idx) if feature_descs else "unlabeled",
            }
            for idx, act in active_pairs
        ]

        step_info["sae_features"] = {
            "num_active": int(nonzero_mask.sum()),
            "total_features": int(features_np.shape[0]),
            "sparsity_pct": float((1.0 - nonzero_mask.mean()) * 100),
            "top_features": top_features,
            "active_features": active_features,
            "theme_evidence": (
                contrastive_theme_evidence(active_pairs, contrastive_map, feature_descs)
                if contrastive_map
                else None
            ),
        }

    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Tier 3: Feature label mapping (already loaded from Hub or local)
    if feature_descs is None:
        print("  Feature descriptions not found -- showing SAE features without labels.")
        return results

    results["features_available"] = True

    for step_info in results["steps"]:
        if "sae_features" not in step_info:
            continue
        for feat in step_info["sae_features"]["top_features"]:
            desc = feature_descs.get(str(feat["index"]))
            if desc:
                if isinstance(desc, str):
                    feat["label"] = desc
                    feat["description"] = ""
                    feat["confidence"] = "unknown"
                else:
                    feat["label"] = desc.get("label", "unknown")
                    feat["description"] = desc.get("description", "")
                    feat["confidence"] = desc.get("confidence", "low")

    return results


# ---------------------------------------------------------------------------
# Section E: UI data generation (for index.html)
# ---------------------------------------------------------------------------


_TOOL_DISPLAY_NAMES = {
    "ManualCheck": "Repair Manual Lookup",
    "PartsSearch": "Parts & Pricing Search",
    "TutorialSearch": "Video Tutorial Search",
    "ProQuote": "Professional Quote",
}

_PROBLEM_META = {
    "dishwasher_leak": {
        "icon": "\U0001f4a7",
        "urgency": {"label": "Fix Soon", "level": "yellow"},
        "difficulty": {"label": "Moderate", "level": "yellow"},
        "safety": {"label": "Moderate", "level": "yellow"},
        "costRange": "$13\u2013$90 DIY",
    },
    "disposal_stuck": {
        "icon": "\u2699\ufe0f",
        "urgency": {"label": "Can Wait", "level": "green"},
        "difficulty": {"label": "Easy", "level": "green"},
        "safety": {"label": "Low", "level": "green"},
        "costRange": "$0\u2013$8 DIY",
    },
    "water_heater_noise": {
        "icon": "\U0001f525",
        "urgency": {"label": "Act Now", "level": "red"},
        "difficulty": {"label": "Difficult", "level": "red"},
        "safety": {"label": "High", "level": "red"},
        "costRange": "$15\u2013$649 DIY",
    },
}

# ---------------------------------------------------------------------------
# Section E1: Feature-level helpers (pure functions, unit tested)
# ---------------------------------------------------------------------------

# Appliance keywords used to decide whether a feature label belongs to one of
# the three demo scenarios.  Shared with evaluate_sae_layers.py.
_SCENARIO_PATTERNS = {
    "dishwasher_leak": (
        "dishwasher",
        "door gasket",
        "spray arm",
        "dishwashing",
    ),
    "disposal_stuck": (
        "garbage disposal",
        "disposal",
        "flywheel",
        "impeller",
        "allen wrench",
        "hex socket",
    ),
    "water_heater_noise": (
        "water heater",
        "anode",
        "sediment",
        "thermocouple",
        "pilot light",
        "gas appliance",
        "gas leak",
        "tank corrosion",
    ),
}


def classify_scenario_label(label: str) -> set[str]:
    """Return the demo scenarios whose appliance keywords appear in ``label``."""
    normalized = label.casefold()
    return {
        problem
        for problem, patterns in _SCENARIO_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    }


# The five contrast types configured for the home-repair scenario.  The pair
# generator names the *first* side of each type the "anchor" request; the
# contrastive feature map records, per feature, whether it fires more on the
# anchor or the contrast side.  Empirically (output/pairs) the anchor side is
# strongly associated with one tool (e.g. warranty_covered -> pro_quote), so
# theme evidence is partly a proxy for the tool the model is about to pick.
_THEME_SIDES = {
    "diy_vs_professional": ("diy", "professional"),
    "urgent_vs_planned": ("urgent", "planned"),
    "cheap_fix_vs_replacement": ("cheap_fix", "replacement"),
    "safe_vs_hazardous": ("safe", "hazardous"),
    "warranty_covered_vs_out_of_pocket": ("warranty_covered", "out_of_pocket"),
}

_THEME_META = {
    "diy_vs_professional": {
        "title": "DIY vs. Professional",
        "anchorLabel": "Easy DIY",
        "contrastLabel": "Needs a Pro",
    },
    "urgent_vs_planned": {
        "title": "Urgent vs. Planned",
        "anchorLabel": "Act Now",
        "contrastLabel": "Can Wait",
    },
    "cheap_fix_vs_replacement": {
        "title": "Cheap Fix vs. Replacement",
        "anchorLabel": "Quick Part Swap",
        "contrastLabel": "Consider Replacing",
    },
    "safe_vs_hazardous": {
        "title": "Safe vs. Hazardous",
        "anchorLabel": "Low Risk",
        "contrastLabel": "High Hazard",
    },
    "warranty_covered_vs_out_of_pocket": {
        "title": "Warranty vs. Out of Pocket",
        "anchorLabel": "May Be Covered",
        "contrastLabel": "Out of Pocket",
    },
}


def _label_for(labels: dict | None, index: int) -> str:
    entry = (labels or {}).get(str(index), (labels or {}).get(index))
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("label") or f"Feature #{index}"
    return f"Feature #{index}"


def contrastive_theme_evidence(
    active: list[tuple[int, float]],
    contrastive_map: dict[int, list[dict]],
    labels: dict | None = None,
    themes: dict[str, tuple[str, str]] | None = None,
    shrink: float = 2.0,
    min_features: int = 3,
    min_coverage: float = 0.02,
    drivers_per_side: int = 2,
) -> dict[str, dict]:
    """Score each contrast theme from *all* active features of one prompt.

    For a theme, every active feature that the training contrastive map lists
    for that theme contributes ``activation * |cohens_d|`` to the mass of the
    side it fires more on (anchor or contrast).  ``position`` is the contrast
    side's share of that mass, shrunk toward 0.5 by ``shrink`` pseudo-mass so
    a theme supported by one or two weak features cannot pin the marker to an
    extreme.  ``coverage`` is the share of the prompt's total activation that
    lands on mapped features; themes with too few features or too little
    coverage are flagged ``insufficient`` and the UI hides their marker.
    """
    themes = themes or _THEME_SIDES
    total_activation = float(sum(activation for _, activation in active))
    evidence: dict[str, dict] = {}
    for theme, (anchor_side, contrast_side) in themes.items():
        sides: dict[str, list[dict]] = {"anchor": [], "contrast": []}
        for index, activation in active:
            for entry in contrastive_map.get(int(index), []):
                if entry.get("theme") != theme:
                    continue
                cohens_d = abs(float(entry.get("cohens_d", 0.0)))
                sides[entry.get("direction", "anchor")].append(
                    {
                        "index": int(index),
                        "label": _label_for(labels, int(index)),
                        "activation": round(float(activation), 4),
                        "cohensD": round(cohens_d, 4),
                        "weight": round(float(activation) * cohens_d, 4),
                    }
                )
        anchor_mass = sum(row["weight"] for row in sides["anchor"])
        contrast_mass = sum(row["weight"] for row in sides["contrast"])
        anchor_activation = sum(row["activation"] for row in sides["anchor"])
        contrast_activation = sum(row["activation"] for row in sides["contrast"])
        n_features = len(sides["anchor"]) + len(sides["contrast"])
        anchor_share = anchor_activation / total_activation if total_activation else 0.0
        contrast_share = contrast_activation / total_activation if total_activation else 0.0
        coverage = anchor_share + contrast_share
        denominator = anchor_mass + contrast_mass + shrink
        position = (contrast_mass + shrink / 2.0) / denominator if denominator > 0 else 0.5
        evidence[theme] = {
            "anchorSide": anchor_side,
            "contrastSide": contrast_side,
            "anchorMass": round(anchor_mass, 4),
            "contrastMass": round(contrast_mass, 4),
            "anchorShare": round(anchor_share, 4),
            "contrastShare": round(contrast_share, 4),
            "nFeatures": n_features,
            "coverage": round(coverage, 4),
            "position": round(position, 4),
            "insufficient": n_features < min_features or coverage < min_coverage,
            "drivers": {
                side: sorted(rows, key=lambda row: -row["weight"])[:drivers_per_side]
                for side, rows in sides.items()
            },
        }
    return evidence


def filter_contrastive_map(
    contrastive_map: dict[int, list[dict]], themes: dict | None = None
) -> dict[int, list[dict]]:
    """Keep only entries for the home-repair contrast themes."""
    themes = themes or _THEME_SIDES
    filtered: dict[int, list[dict]] = {}
    for index, entries in contrastive_map.items():
        kept = [entry for entry in entries if entry.get("theme") in themes]
        if kept:
            filtered[int(index)] = kept
    return filtered


def tool_first_token_ids(tokenizer, tools: list[dict]) -> dict[str, int]:
    """Map each tool id to the first token of ``" {name}"``.

    The decision prompt ends in ``I'll use the`` (no trailing space), so the
    model's next token is the space-prefixed tool name.  The four home-repair
    tools must start with distinct tokens for a single-position readout to be
    meaningful; otherwise a ``ValueError`` is raised.
    """
    ids: dict[str, int] = {}
    for tool in tools:
        name = tool["name"]
        encoded = tokenizer.encode(f" {name}", add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Tokenizer produced no tokens for tool {name!r}")
        ids[name] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"Tool names do not start with distinct tokens: {ids}")
    for name, token_id in ids.items():
        piece = tokenizer.decode([token_id]).strip()
        if piece and not name.startswith(piece):
            print(f"  WARNING: first token {piece!r} is not a prefix of tool {name!r}")
    return ids


def decision_from_logprobs(
    logprobs: dict[int, float],
    tool_to_token: dict[str, int],
    sampled_id: int | None = None,
    completion: str = "",
    truncated: bool = False,
    coverage_warn: float = 0.5,
) -> dict:
    """Turn next-token log-probabilities into a tool-choice readout.

    ``logprobs`` maps token id -> natural-log probability at the decision
    position.  Probabilities are renormalised over the tool tokens; ``coverage``
    is the raw probability mass they account for.  When ``truncated`` (top-k
    logprobs only), tools missing from ``logprobs`` are reported as 0.
    """
    import math

    raw = {
        tool: (math.exp(logprobs[token_id]) if token_id in logprobs else 0.0)
        for tool, token_id in tool_to_token.items()
    }
    coverage = float(sum(raw.values()))
    distribution = {
        tool: (value / coverage if coverage > 0 else 0.0) for tool, value in raw.items()
    }
    token_to_tool = {token_id: tool for tool, token_id in tool_to_token.items()}
    sampled_tool = token_to_tool.get(sampled_id) if sampled_id is not None else None
    best_tool = max(distribution, key=distribution.get) if coverage > 0 else None
    tool = sampled_tool or best_tool
    return {
        "toolId": tool,
        "display": tool_display(tool) if tool else None,
        "prob": round(distribution.get(tool, 0.0), 4) if tool else 0.0,
        "distribution": {tool_display(t): round(p, 4) for t, p in distribution.items()},
        "raw": {tool_display(t): round(p, 6) for t, p in raw.items()},
        "coverage": round(coverage, 4),
        "lowCoverage": coverage < coverage_warn,
        "truncated": truncated,
        "sampledToken": sampled_id,
        "sampledTool": sampled_tool,
        "completion": completion,
    }


_LABEL_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}


def _label_key(label: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z]+", label.lower())
    normalized = set()
    for token in tokens:
        if token in _LABEL_STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        normalized.add(token)
    return frozenset(normalized)


def dedupe_feature_rows(rows: list[dict], jaccard: float = 0.6) -> list[dict]:
    """Merge rows whose labels are near-duplicates (feature splitting).

    Rows are processed in the given order; a row joins the first group whose
    representative label shares at least ``jaccard`` of its normalised tokens.
    The representative keeps the highest activation and lists ``merged``
    feature indices so nothing is hidden.
    """
    groups: list[tuple[frozenset[str], dict]] = []
    for row in rows:
        key = _label_key(row.get("label", ""))
        for group_key, representative in groups:
            union = key | group_key
            if union and len(key & group_key) / len(union) >= jaccard:
                if row.get("activation", 0.0) > representative.get("activation", 0.0):
                    merged = representative.setdefault("merged", [])
                    merged.append(representative["index"])
                    for field in ("index", "label", "activation", "delta", "share"):
                        if field in row:
                            representative[field] = row[field]
                    representative["merged"] = merged
                else:
                    representative.setdefault("merged", []).append(row["index"])
                break
        else:
            groups.append((key, dict(row)))
    return [representative for _, representative in groups]


# Specific vocabulary a label may carry that a viewer would recognise as an
# inference if the request never states it.  Generic labels (e.g. "Home
# appliance repair guide request") never trigger the flag.
_SPECIFIC_TERMS = {
    "gas": ("gas",),
    "electric": ("electric", "electrical", "electricity"),
    "pilot": ("pilot",),
    "valve": ("valve", "valves"),
    "burner": ("burner", "burners"),
    "ignition": ("ignition", "ignite", "igniter"),
    "warranty": ("warranty", "warranties", "warrantied", "covered"),
    "urgent": ("urgent", "urgency", "urgently", "immediate", "immediately"),
    "emergency": ("emergency", "emergencies"),
    "replacement": ("replacement", "replace", "replacing", "replaced"),
    "old": ("old", "aging", "ageing", "aged"),
    "corrosion": ("corrosion", "corroded", "corroding"),
    "rust": ("rust", "rusty", "rusted"),
    "hazard": ("hazard", "hazardous", "hazards"),
    "safety": ("safety", "safe", "safely", "unsafe", "danger", "dangerous"),
    "reset": ("reset", "resetting"),
    "breaker": ("breaker", "breakers", "tripped"),
    "drain": ("drain", "drainage", "draining", "drains"),
    "leak": ("leak", "leaking", "leakage", "leaks"),
    "flood": ("flood", "flooding", "flooded"),
    "sediment": ("sediment", "scale", "buildup"),
    "water heater": ("water heater", "heater"),
    "dishwasher": ("dishwasher", "dishwashing"),
    "disposal": ("disposal", "disposer"),
    "washing machine": ("washing machine", "washer"),
    "dryer": ("dryer",),
    "refrigerator": ("refrigerator", "fridge", "freezer"),
    "professional": ("professional", "technician", "plumber", "electrician", "licensed"),
    "jam": ("jam", "jammed", "jamming", "unjamming", "unjam"),
    "humming": ("humming", "hums", "hum"),
    "motor": ("motor",),
    "impeller": ("impeller", "flywheel"),
    "gasket": ("gasket", "seal"),
    "spray arm": ("spray arm", "spray-arm"),
    "anode": ("anode",),
    "thermocouple": ("thermocouple",),
    "burning": ("burning", "burnt", "smoke", "smell"),
}
# Ask-type words (part, guide, video, quote) are deliberately absent: a label
# such as "part replacement" is stated by a request that names the part
# ("replacement gasket"), so flagging them produces false positives.  The badge
# is for situational content the request never mentions.


def _terms_in(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z\- ]", " ", text.lower())
    words = set(normalized.replace("-", " ").split())
    found = set()
    for term, variants in _SPECIFIC_TERMS.items():
        for variant in variants:
            if " " in variant:
                if variant in normalized:
                    found.add(term)
                    break
            elif variant in words:
                found.add(term)
                break
    return found


def not_stated_in_request(label: str, request: str) -> list[str]:
    """Specific terms the label carries that the request never mentions."""
    return sorted(_terms_in(label) - _terms_in(request))


def feature_rows(
    active: list[tuple[int, float]],
    baseline_mean: dict[int, float],
    labels: dict | None,
    request: str,
    top_n: int = 6,
) -> list[dict]:
    """Rank a prompt's active features by how much they exceed the baseline.

    ``baseline_mean`` is the mean activation over the base prompts (absent =
    0), so prompt-specific features rise to the top.  Rows keep the raw
    activation, the deviation and the share of the prompt's maximum activation
    so the UI can show all three without inventing a "strength".
    """
    max_activation = max((activation for _, activation in active), default=0.0) or 1.0
    rows = []
    for index, activation in active:
        rows.append(
            {
                "index": int(index),
                "label": _label_for(labels, int(index)),
                "activation": round(float(activation), 4),
                "delta": round(float(activation) - float(baseline_mean.get(int(index), 0.0)), 4),
                "share": round(float(activation) / max_activation, 4),
            }
        )
    rows.sort(key=lambda row: (-row["delta"], -row["activation"]))
    rows = dedupe_feature_rows(rows)
    rows.sort(key=lambda row: (-row["delta"], -row["activation"]))
    selected = rows[:top_n]
    for row in selected:
        row["notStated"] = not_stated_in_request(row["label"], request)
        row.setdefault("merged", [])
    return selected


def shared_feature_rows(
    active_by_prompt: list[list[tuple[int, float]]], labels: dict | None, top_n: int = 5
) -> list[dict]:
    """Features active on every prompt, ranked by their minimum activation."""
    if not active_by_prompt:
        return []
    per_prompt = [{int(i): float(a) for i, a in active} for active in active_by_prompt]
    common = set(per_prompt[0])
    for mapping in per_prompt[1:]:
        common &= set(mapping)
    rows = [
        {
            "index": index,
            "label": _label_for(labels, index),
            "minActivation": round(min(mapping[index] for mapping in per_prompt), 4),
            "meanActivation": round(
                sum(mapping[index] for mapping in per_prompt) / len(per_prompt), 4
            ),
        }
        for index in common
    ]
    rows.sort(key=lambda row: -row["minActivation"])
    return dedupe_feature_rows(rows)[:top_n]


def also_fired(rows: list[dict], problem_id: str) -> list[dict]:
    """Rows whose label names a *different* demo scenario (contamination)."""
    flagged = []
    for row in rows:
        matches = classify_scenario_label(row.get("label", ""))
        others = sorted(matches - {problem_id})
        if others:
            flagged.append({**row, "otherScenarios": others})
    return flagged


def contrast_diff(
    base_active: list[tuple[int, float]],
    variant_active: list[tuple[int, float]],
    labels: dict | None,
    request: str,
    top_n: int = 5,
) -> dict[str, list[dict]]:
    """Features gained, lost, or shifted between a base prompt and its variant."""
    base = {int(i): float(a) for i, a in base_active}
    variant = {int(i): float(a) for i, a in variant_active}
    gained, lost, shifted = [], [], []
    for index in set(base) | set(variant):
        before = base.get(index, 0.0)
        after = variant.get(index, 0.0)
        row = {
            "index": index,
            "label": _label_for(labels, index),
            "base": round(before, 4),
            "variant": round(after, 4),
            "delta": round(after - before, 4),
        }
        if before <= 0 < after:
            gained.append(row)
        elif after <= 0 < before:
            lost.append(row)
        elif before > 0 and after > 0:
            shifted.append(row)
    gained.sort(key=lambda row: -row["variant"])
    lost.sort(key=lambda row: -row["base"])
    shifted.sort(key=lambda row: -abs(row["delta"]))
    result = {}
    for name, rows in (("gained", gained), ("lost", lost), ("shifted", shifted)):
        rows = dedupe_feature_rows(rows)[:top_n]
        for row in rows:
            row["notStated"] = not_stated_in_request(row["label"], request)
            row.setdefault("merged", [])
        result[name] = rows
    return result


def _active_pairs(sae_features: dict) -> list[tuple[int, float]]:
    """(index, activation) pairs for a step, preferring the full active list."""
    rows = sae_features.get("active_features") or sae_features.get("top_features") or []
    return [(int(row["index"]), float(row["activation"])) for row in rows]


def _labels_from_rows(*row_lists: list[dict]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for rows in row_lists:
        for row in rows or []:
            label = row.get("label")
            if label:
                labels[str(row["index"])] = label
    return labels


# ---------------------------------------------------------------------------
# Section E2: "is this just keyword matching?" probes
# ---------------------------------------------------------------------------


def row_family(row: dict) -> list[int]:
    """The feature indices a snapshot row stands for (itself + merged twins)."""
    return [int(row["index"])] + [int(index) for index in row.get("merged", []) or []]


def family_activation(active: dict[int, float] | list[tuple[int, float]], family: list[int]):
    """Maximum activation of a feature family on a prompt (0 when silent)."""
    mapping = active if isinstance(active, dict) else {int(i): float(a) for i, a in active}
    return max((float(mapping.get(int(index), 0.0)) for index in family), default=0.0)


def active_overlap(
    left: list[tuple[int, float]], right: list[tuple[int, float]]
) -> dict[str, float]:
    """Jaccard of the active sets and cosine of the sparse activation vectors."""
    left_map = {int(i): float(a) for i, a in left if a > 0}
    right_map = {int(i): float(a) for i, a in right if a > 0}
    union = set(left_map) | set(right_map)
    inter = set(left_map) & set(right_map)
    jaccard = len(inter) / len(union) if union else 0.0
    dot = sum(left_map[i] * right_map[i] for i in inter)
    left_norm = math.sqrt(sum(v * v for v in left_map.values()))
    right_norm = math.sqrt(sum(v * v for v in right_map.values()))
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return {"jaccard": round(jaccard, 4), "cosine": round(cosine, 4)}


def label_mentions(label: str, words: list[str] | tuple[str, ...]) -> bool:
    """True when the label contains one of ``words`` as a whole word."""
    tokens = set(re.findall(r"[a-z]+", label.lower()))
    return any(word.lower() in tokens for word in words)


def snapshot_feature_rows(
    active_by_step: dict[str, list[tuple[int, float]]], labels: dict | None, top_n: int = 6
) -> tuple[dict[str, list[dict]], dict[int, float], list[dict]]:
    """Per-problem snapshot rows exactly as the page shows them.

    Returns ``(rows_by_problem, baseline_mean, shared_rows)``.  The baseline is
    the mean activation over the base prompts with absence counted as zero, so
    prompt-specific features rise to the top (see :func:`feature_rows`).  The
    steering script calls this too, so the causal check ablates the very rows
    the page displays.
    """
    base_steps = [f"{p['id']}_InitialDecision" for p in _PROBLEMS]
    base_active = [active_by_step[s] for s in base_steps if s in active_by_step]
    baseline_mean: dict[int, float] = {}
    if base_active:
        for active in base_active:
            for index, activation in active:
                baseline_mean[int(index)] = baseline_mean.get(int(index), 0.0) + float(activation)
        baseline_mean = {i: v / len(base_active) for i, v in baseline_mean.items()}
    shared_rows = shared_feature_rows(base_active, labels) if len(base_active) > 1 else []
    rows_by_problem: dict[str, list[dict]] = {}
    for p in _PROBLEMS:
        _, request = _initial_tool_decision(p)
        active = active_by_step.get(f"{p['id']}_InitialDecision", [])
        rows_by_problem[p["id"]] = (
            feature_rows(active, baseline_mean, labels, request, top_n=top_n) if active else []
        )
    return rows_by_problem, baseline_mean, shared_rows


_FIRES_FRACTION = 0.5


def _row_cells(rows: list[dict], base_map: dict[int, float], active_map: dict[int, float]):
    """Per snapshot row: family activation on a prompt relative to its base value.

    ``fires`` is a graded notion — the family must reach at least half of its
    base-prompt activation.  Snapshot rows stand for families of split
    features (up to two dozen), so "any member above zero" is nearly always
    true somewhere and would not separate a paraphrase from an unrelated
    request; the ratio does.
    """
    cells = []
    for row in rows:
        family = row_family(row)
        base_value = family_activation(base_map, family)
        value = family_activation(active_map, family)
        ratio = value / base_value if base_value > 0 else (1.0 if value > 0 else 0.0)
        cells.append(
            {
                "index": row["index"],
                "activation": round(value, 4),
                "base": round(base_value, 4),
                "ratio": round(ratio, 4),
                "active": value > 0,
                "fires": ratio >= _FIRES_FRACTION and value > 0,
            }
        )
    return cells


def paraphrase_evidence(
    rows: list[dict],
    base_active: list[tuple[int, float]],
    paraphrases: list[dict],
    other_prompts: list[dict],
    base_tool_id: str | None = None,
) -> dict:
    """Do the snapshot features survive a rewording?

    ``paraphrases`` and ``other_prompts`` are ``{step, request, active,
    modelChoice, label}`` dicts; ``other_prompts`` are the comparison points
    (the same problem's contrast and the other problems' base prompts) whose
    overlap with the base prompt calibrates what "similar" means here.  Both
    get the same per-row cells (see :func:`_row_cells`).  ``sameTool`` counts
    the paraphrases whose readout picks ``base_tool_id`` (the tool the base
    request's readout picked) — a consistency check, not evidence: the
    paraphrases restate the ask.
    """
    base_map = {int(i): float(a) for i, a in base_active}
    items = []
    for item in paraphrases:
        active_map = {int(i): float(a) for i, a in item["active"]}
        cells = _row_cells(rows, base_map, active_map)
        items.append(
            {
                "step": item["step"],
                "request": item["request"],
                "modelChoice": item.get("modelChoice"),
                "overlap": active_overlap(base_active, item["active"]),
                "rows": cells,
                "rowsFiring": sum(1 for cell in cells if cell["fires"]),
            }
        )
    comparisons = []
    for item in other_prompts:
        active_map = {int(i): float(a) for i, a in item["active"]}
        cells = _row_cells(rows, base_map, active_map)
        comparisons.append(
            {
                "step": item["step"],
                "label": item.get("label") or item["step"],
                "request": item["request"],
                "modelChoice": item.get("modelChoice"),
                "overlap": active_overlap(base_active, item["active"]),
                "rows": cells,
                "rowsFiring": sum(1 for cell in cells if cell["fires"]),
            }
        )
    row_summary = []
    for position, row in enumerate(rows):
        fires_in = sum(1 for item in items if item["rows"][position]["fires"])
        row_summary.append(
            {
                "index": row["index"],
                "label": row["label"],
                "baseActivation": round(family_activation(base_map, row_family(row)), 4),
                "firesIn": fires_in,
                "of": len(items),
                "firesInComparisons": sum(1 for c in comparisons if c["rows"][position]["fires"]),
                "ofComparisons": len(comparisons),
            }
        )
    same_tool = sum(
        1
        for item in items
        if base_tool_id is not None
        and (item.get("modelChoice") or {}).get("toolId") == base_tool_id
    )
    return {
        "paraphrases": items,
        "comparisons": comparisons,
        "rowSummary": row_summary,
        "meanJaccard": round(
            sum(i["overlap"]["jaccard"] for i in items) / len(items) if items else 0.0, 4
        ),
        "meanCosine": round(
            sum(i["overlap"]["cosine"] for i in items) / len(items) if items else 0.0, 4
        ),
        "rowsFiringInAll": sum(1 for r in row_summary if r["firesIn"] == len(items)),
        "firesFraction": _FIRES_FRACTION,
        "sameTool": same_tool,
    }


def keyword_control_evidence(
    control: dict,
    base_active: list[tuple[int, float]],
    control_active: list[tuple[int, float]],
    active_by_step: dict[str, list[tuple[int, float]]],
    labels: dict | None,
    rows: list[dict],
    max_targets: int = 4,
) -> dict:
    """Did adding (or removing) a word switch on (or off) the features named by it?

    Targets are the captured features whose label mentions one of the
    control's ``targetWords``; for an *added* word they are ranked by their
    strongest activation on any captured prompt (so the viewer sees features
    that demonstrably can fire), for a *removed* word by their activation on
    the base prompt.  Each target reports its base and control activation.
    """
    base_map = {int(i): float(a) for i, a in base_active}
    control_map = {int(i): float(a) for i, a in control_active}
    words = control.get("targetWords") or [control.get("keyword", "")]
    # Where does each candidate feature fire most among the captured prompts?
    strongest: dict[int, tuple[float, str]] = {}
    for step, active in active_by_step.items():
        for index, activation in active:
            index = int(index)
            if label_mentions(_label_for(labels, index), words):
                if float(activation) > strongest.get(index, (0.0, ""))[0]:
                    strongest[index] = (float(activation), step)
    if control.get("direction") == "removed":
        candidates = sorted(
            (index for index in strongest if base_map.get(index, 0.0) > 0),
            key=lambda index: -base_map.get(index, 0.0),
        )
    else:
        candidates = sorted(strongest, key=lambda index: -strongest[index][0])
    targets = []
    for index in candidates[:max_targets]:
        base_value = base_map.get(index, 0.0)
        control_value = control_map.get(index, 0.0)
        peak, peak_step = strongest[index]
        targets.append(
            {
                "index": index,
                "label": _label_for(labels, index),
                "base": round(base_value, 4),
                "control": round(control_value, 4),
                "peak": round(peak, 4),
                "peakStep": peak_step,
                "delta": round(control_value - base_value, 4),
                # A target "responds" to the word when the change is at least a
                # quarter of the feature's strongest captured activation.
                "responds": abs(control_value - base_value) >= 0.25 * peak and peak > 0,
            }
        )
    responding = [t for t in targets if t["responds"]]
    if control.get("direction") == "removed":
        verdict = "turned off" if responding else "still fire"
        if targets and all(t["base"] > 0 and t["control"] <= 0 for t in targets):
            verdict = "turned off"
    else:
        verdict = "fired" if responding else "stayed quiet"
    snapshot = [
        {
            "index": row["index"],
            "base": round(family_activation(base_map, row_family(row)), 4),
            "control": round(family_activation(control_map, row_family(row)), 4),
        }
        for row in rows
    ]
    return {
        "step": control["step"],
        "request": control["request"],
        "direction": control.get("direction", "added"),
        "keyword": control.get("keyword"),
        "note": control.get("note"),
        "modelChoice": control.get("modelChoice"),
        "overlap": active_overlap(base_active, control_active),
        "targets": targets,
        "responding": len(responding),
        "verdict": verdict,
        "snapshotRows": snapshot,
    }


def _summarize_manual(data: dict) -> str:
    causes = ", ".join(c.split(" -- ")[0] for c in data.get("possible_causes", []))
    safety = data.get("safety", "")
    difficulty = data.get("diy_difficulty", "")
    parts = [f"<strong>Possible causes:</strong> {causes}"]
    if difficulty:
        parts.append(f"<strong>DIY difficulty:</strong> {difficulty}")
    if safety:
        parts.append(f"<strong>Safety:</strong> {safety}")
    if data.get("quick_fix"):
        parts.append(f"<strong>Quick fix:</strong> {data['quick_fix']}")
    return "<br>".join(parts)


def _summarize_parts(data: dict) -> str:
    items = []
    for p in data.get("parts", [])[:3]:
        items.append(f"<strong>{p['name']}:</strong> ${p['price']:.2f}")
    line1 = " &middot; ".join(items)
    line2 = data.get("diy_cost_range", "")
    return f"{line1}<br>{line2}" if line2 else line1


def _summarize_tutorials(data: dict) -> str:
    lines = []
    for r in data.get("results", [])[:2]:
        meta = []
        if r.get("views"):
            meta.append(f"{r['views']} views")
        if r.get("duration"):
            meta.append(r["duration"])
        if r.get("difficulty"):
            meta.append(r["difficulty"])
        meta_str = f" ({', '.join(meta)})" if meta else ""
        channel = r.get("channel", "")
        lines.append(f'<strong>"{r["title"]}"</strong> by {channel}{meta_str}')
    return "<br>".join(lines)


def _summarize_pro_quote(data: dict) -> str:
    items = []
    for est in data.get("repair_estimates", []):
        items.append(f"<strong>{est['repair']}:</strong> ${est['total']} total")
    line1 = " &middot; ".join(items)
    extras = []
    if data.get("diagnosis_fee"):
        extras.append(f"<strong>Diagnosis fee:</strong> ${data['diagnosis_fee']}")
    if data.get("next_available"):
        extras.append(f"Available in {data['next_available']}")
    if data.get("warranty_on_repair"):
        extras.append(f"{data['warranty_on_repair']} warranty")
    line2 = ". ".join(extras)
    return f"{line1}<br>{line2}"


_TOOL_SUMMARIZERS = {
    "ManualCheck": _summarize_manual,
    "PartsSearch": _summarize_parts,
    "TutorialSearch": _summarize_tutorials,
    "ProQuote": _summarize_pro_quote,
}


def _generate_feature_sentence(tool_name: str, features: list[dict]) -> str:
    """Generate a plain-language sentence from top features."""
    if not features:
        return ""
    display = _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    top = features[:2]
    labels = [f"<strong>{f['label']}</strong>" for f in top]
    if len(labels) == 2:
        focus = f"{labels[0]} and {labels[1]}"
    else:
        focus = labels[0]
    verb = {
        "ManualCheck": "checking the repair manual",
        "PartsSearch": "reviewing parts and pricing",
        "TutorialSearch": "searching for tutorials",
        "ProQuote": "reviewing professional quotes",
    }.get(tool_name, f"running {display}")
    return f"While deciding to start by {verb}, the model's most distinctive features were {focus}."


def _decision_lookup(analysis: dict) -> dict[str, dict]:
    return {step_info["step"]: step_info for step_info in analysis.get("steps", [])}


def _model_choice_from_step(step_info: dict | None) -> dict | None:
    if not step_info:
        return None
    decision = step_info.get("decision")
    return dict(decision) if decision else None


def attribution_backends_agree(attribution: dict | None, model_choice: dict | None) -> bool:
    """True when the HF baseline picks the same tool as the vLLM readout.

    The per-row ablations run on the HF backend; when its baseline decision
    differs from the vLLM decision shown in the snapshot, the ablation measures
    a different decision than the one on the page, so the causal column is
    withheld rather than shown with a caveat.  Without a vLLM readout there is
    nothing to disagree with.
    """
    if not attribution:
        return False
    hf_choice = attribution.get("hfChoice")
    vllm_choice = (model_choice or {}).get("display")
    return not (hf_choice and vllm_choice and hf_choice != vllm_choice)


_MIN_CAUSAL_EFFECT = 0.02  # below 2 pp a family is "descriptive" whatever the random band


def attach_attribution(rows: list[dict], attribution: dict | None) -> None:
    """Attach the HF ablation result of each row family to the row (in place).

    ``attribution`` is one problem's entry of ``steering_results.json``'s
    ``attribution`` block: ``rows`` (per snapshot row: ``deltaTarget``,
    ``intervened``, ``hfActivation`` ...), ``controlThreshold`` (the largest
    |delta| produced by ablating random same-sized sets of active features).
    Rows whose effect does not exceed both the control threshold and
    ``_MIN_CAUSAL_EFFECT`` (2 pp — a tiny random band must not promote a
    0.3 pp effect) are marked ``descriptive``; the UI sorts by causal effect
    when this block exists. Callers should first check
    :func:`attribution_backends_agree`.
    """
    if not attribution:
        return
    by_index = {int(entry["index"]): entry for entry in attribution.get("rows", [])}
    threshold = max(float(attribution.get("controlThreshold") or 0.0), _MIN_CAUSAL_EFFECT)
    for row in rows:
        entry = by_index.get(int(row["index"]))
        if not entry:
            continue
        delta = entry.get("deltaTarget")
        row["causal"] = {
            "deltaTarget": delta,
            "targetTool": attribution.get("targetTool"),
            "hfActivation": entry.get("hfActivation"),
            "inactiveUnderHf": bool(entry.get("inactiveUnderHf")),
            "intervened": entry.get("intervened"),
            "argmaxChanged": entry.get("argmaxChanged"),
            "descriptive": delta is None or abs(float(delta)) <= threshold,
        }


def probe_evidence(
    problem: dict,
    rows: list[dict],
    active_by_step: dict[str, list[tuple[int, float]]],
    step_lookup: dict[str, dict],
    labels: dict | None,
) -> dict | None:
    """Assemble the paraphrase + keyword-control block for one problem."""
    pid = problem["id"]
    base_step = f"{pid}_InitialDecision"
    base_active = active_by_step.get(base_step)
    if not base_active:
        return None
    probe_prompts = [
        item
        for item in decision_prompts(include_contrasts=True, include_probes=True)
        if item["problem"] == pid and item["kind"] in ("paraphrase", "control")
    ]
    base_choice = _model_choice_from_step(step_lookup.get(base_step)) or {}

    def _entry(item: dict, label: str | None = None) -> dict | None:
        active = active_by_step.get(item["step"])
        if active is None:
            return None
        return {
            **item,
            "label": label,
            "active": active,
            "modelChoice": _model_choice_from_step(step_lookup.get(item["step"])),
        }

    paraphrases = [
        entry
        for entry in (_entry(item) for item in probe_prompts if item["kind"] == "paraphrase")
        if entry
    ]
    controls = [
        entry
        for entry in (_entry(item) for item in probe_prompts if item["kind"] == "control")
        if entry
    ]
    if not paraphrases and not controls:
        return None

    others = []
    contrast = problem.get("contrast")
    if contrast and active_by_step.get(f"{pid}_Contrast") is not None:
        others.append(
            {
                "step": f"{pid}_Contrast",
                "label": "same situation, different ask",
                "request": contrast["request"],
                "active": active_by_step[f"{pid}_Contrast"],
                "modelChoice": _model_choice_from_step(step_lookup.get(f"{pid}_Contrast")),
            }
        )
    for other in _PROBLEMS:
        if other["id"] == pid:
            continue
        step = f"{other['id']}_InitialDecision"
        if active_by_step.get(step) is None:
            continue
        others.append(
            {
                "step": step,
                "label": f"different problem ({other['summary'].lower()})",
                "request": _initial_tool_decision(other)[1],
                "active": active_by_step[step],
                "modelChoice": _model_choice_from_step(step_lookup.get(step)),
            }
        )
    block: dict = {"baseTool": base_choice.get("display")}
    if paraphrases:
        block["paraphrase"] = paraphrase_evidence(
            rows, base_active, paraphrases, others, base_tool_id=base_choice.get("toolId")
        )
    if controls:
        block["controls"] = [
            keyword_control_evidence(
                control, base_active, control["active"], active_by_step, labels, rows
            )
            for control in controls
        ]
    return block


def open_request_evidence(
    problem: dict,
    rows: list[dict],
    active_by_step: dict[str, list[tuple[int, float]]],
    step_lookup: dict[str, dict],
    labels: dict | None,
) -> dict | None:
    """What the model does on the situation alone (no ask), vs the base request.

    The base request's ask names the tool, so its readout is a restatement;
    the open request is where the tool choice is a genuine decision.  Returns
    the open readout, the active-set overlap with the base request, the
    snapshot rows followed into the open prompt (same cells as the paraphrase
    table) and the features gained / lost / shifted without the ask.
    """
    pid = problem["id"]
    open_meta = problem.get("open")
    base_active = active_by_step.get(f"{pid}_InitialDecision")
    open_active = active_by_step.get(f"{pid}_Open")
    if not open_meta or base_active is None or open_active is None:
        return None
    base_map = {int(i): float(a) for i, a in base_active}
    open_map = {int(i): float(a) for i, a in open_active}
    cells = _row_cells(rows, base_map, open_map)
    for cell, row in zip(cells, rows, strict=True):
        cell["label"] = row["label"]
    return {
        "request": open_meta["request"],
        "modelChoice": _model_choice_from_step(step_lookup.get(f"{pid}_Open")),
        "baseChoice": _model_choice_from_step(step_lookup.get(f"{pid}_InitialDecision")),
        "overlap": active_overlap(base_active, open_active),
        "numActive": len(open_map),
        "rows": cells,
        "rowsFiring": sum(1 for cell in cells if cell["fires"]),
        "firesFraction": _FIRES_FRACTION,
        **contrast_diff(base_active, open_active, labels, open_meta["request"]),
    }


def injection_summary(injection: dict | None, open_choice: dict | None) -> dict | None:
    """Condense one problem's ``steering_results.json`` ``injection`` block.

    The injection clamps the explicit-ask request's snapshot families into the
    open (no-ask) prompt on the HF backend; only the headline numbers reach the
    page.  Withheld when the HF open baseline picks a different tool than the
    vLLM readout shown next to it.
    """
    if not injection:
        return None
    hf_choice = injection.get("hfChoice")
    vllm_choice = (open_choice or {}).get("display")
    if hf_choice and vllm_choice and hf_choice != vllm_choice:
        return {"withheld": True, "hfChoice": hf_choice, "vllmChoice": vllm_choice}
    rows = injection.get("rows") or []
    best = max(rows, key=lambda r: r.get("deltaTarget") or 0.0, default=None)
    return {
        "withheld": False,
        "targetTool": injection.get("targetTool"),
        "hfChoice": hf_choice,
        "bestRow": (
            {"index": best["index"], "label": best.get("label"), "deltaTarget": best["deltaTarget"]}
            if best
            else None
        ),
        "allRows": {
            key: (injection.get("allRows") or {}).get(key)
            for key in ("size", "deltaTarget", "argmaxChanged", "choice")
        },
        "allBase": {
            key: (injection.get("allBase") or {}).get(key)
            for key in ("size", "deltaTarget", "argmaxChanged", "choice")
        },
        "controlThreshold": injection.get("controlThreshold"),
    }


def build_ui_data(
    analysis: dict,
    per_problem: dict[str, str],
    final_recommendation: str,
    youtube_api_key: str | None = None,
    model_name: str = _MODEL_NAME,
    sae_layer: int = _SAE_LAYER,
    threshold_offset: float = _HF_THRESHOLD_OFFSET,
    steering: dict | None = None,
) -> dict:
    """Transform demo outputs into the DATA shape expected by index.html.

    Sections that depend on data the run did not produce (tool-choice readout,
    contrast prompts, probes, open requests) are emitted as ``null``/absent so the page can
    hide them instead of inventing values.  ``steering`` is the
    ``steering_results.json`` payload; only its per-row ``attribution`` block
    reaches the page (as ``features[*].causal``) — the ablate/clamp experiments
    it also records are kept on disk but not displayed.
    """

    # --- problems ---
    problems = []
    for p in _PROBLEMS:
        meta = _PROBLEM_META.get(p["id"], {})
        problems.append(
            {
                "id": p["id"],
                "icon": meta.get("icon", ""),
                "title": p["summary"],
                "appliance": p["appliance"],
                "age": p["age"],
                "details": p["details"],
                "urgency": meta.get("urgency", {"label": "Unknown", "level": "yellow"}),
                "difficulty": meta.get("difficulty", {"label": "Unknown", "level": "yellow"}),
                "safety": meta.get("safety", {"label": "Unknown", "level": "yellow"}),
                "costRange": meta.get("costRange", ""),
            }
        )

    # --- toolResults: HTML summaries from mock data ---
    tool_results: dict[str, dict[str, str]] = {}
    for p in _PROBLEMS:
        pid = p["id"]
        tool_results[pid] = {}
        for tool_name, (source, _) in _TOOLS.items():
            data = source.get(pid, {})
            summarizer = _TOOL_SUMMARIZERS.get(tool_name)
            tool_results[pid][tool_name] = summarizer(data) if summarizer else json.dumps(data)

    # --- per-step feature data ---
    step_lookup = _decision_lookup(analysis)
    labels: dict[str, str] = {}
    active_by_step: dict[str, list[tuple[int, float]]] = {}
    for step_name, step_info in step_lookup.items():
        sae_features = step_info.get("sae_features")
        if not sae_features:
            continue
        active_by_step[step_name] = _active_pairs(sae_features)
        labels.update(
            _labels_from_rows(
                sae_features.get("active_features", []), sae_features.get("top_features", [])
            )
        )

    # Baseline: mean activation over the *base* prompts, treating absence as
    # zero (see snapshot_feature_rows).  The steering script derives the rows
    # it ablates from the same function, so the causal numbers refer to the
    # rows on the page.
    rows_by_problem, _, shared_rows = snapshot_feature_rows(active_by_step, labels)
    attribution = (steering or {}).get("attribution") or {}

    # --- decisionFeatures: one honest initial tool-choice snapshot per problem ---
    decision_features: dict[str, dict] = {}
    contrasts: dict[str, dict] = {}
    probes: dict[str, dict] = {}
    open_requests: dict[str, dict] = {}
    for p in _PROBLEMS:
        pid = p["id"]
        tool_name, request = _initial_tool_decision(p)
        step_name = f"{pid}_InitialDecision"
        step_info = step_lookup.get(step_name)
        active = active_by_step.get(step_name, [])
        rows = rows_by_problem.get(pid, [])
        sae_features = (step_info or {}).get("sae_features") or {}
        theme_evidence = sae_features.get("theme_evidence") or None
        model_choice = _model_choice_from_step(step_info)
        # Causal column only where the HF pass and the vLLM readout agree on
        # the baseline tool; otherwise record why it is withheld.
        problem_attribution = attribution.get(pid)
        causal_withheld = None
        if problem_attribution and attribution_backends_agree(problem_attribution, model_choice):
            attach_attribution(rows, problem_attribution)
        elif problem_attribution:
            causal_withheld = {
                "reason": "hf_baseline_disagrees",
                "hfChoice": problem_attribution.get("hfChoice"),
                "vllmChoice": (model_choice or {}).get("display"),
            }
        # Phrase the summary around the tool the model actually chose when the
        # readout exists; the tool the ask names is only a fallback.
        sentence_tool = (model_choice or {}).get("display") or tool_name
        decision_features[pid] = {
            # The tool the ask describes — a design input, not a prediction.
            "askTool": tool_name,
            "request": request,
            "features": rows,
            "sentence": _generate_feature_sentence(sentence_tool, rows),
            "modelChoice": model_choice,
            "sharedAcrossProblems": shared_rows,
            "alsoFired": also_fired(rows, pid),
            "themeEvidence": theme_evidence,
            "numActive": sae_features.get("num_active"),
            "totalFeatures": sae_features.get("total_features"),
        }
        if problem_attribution and causal_withheld is None:
            decision_features[pid]["attribution"] = {
                key: value for key, value in problem_attribution.items() if key != "rows"
            }
        elif causal_withheld:
            decision_features[pid]["causalWithheld"] = causal_withheld

        probe_block = probe_evidence(p, rows, active_by_step, step_lookup, labels)
        if probe_block:
            probes[pid] = probe_block
        open_block = open_request_evidence(p, rows, active_by_step, step_lookup, labels)
        if open_block:
            summary = injection_summary(
                ((steering or {}).get("injection") or {}).get(pid), open_block.get("modelChoice")
            )
            if summary:
                open_block["injection"] = summary
            open_requests[pid] = open_block

        contrast_step = step_lookup.get(f"{pid}_Contrast")
        contrast_meta = p.get("contrast")
        if contrast_step and contrast_meta and active:
            variant_active = active_by_step.get(f"{pid}_Contrast", [])
            contrast_features = contrast_step.get("sae_features") or {}
            variant_by_index = {int(i): float(a) for i, a in variant_active}
            # Follow the base decision's "not stated in request" features into
            # the variant: does the model update them when the wording changes?
            tracked = [
                {
                    "index": row["index"],
                    "label": row["label"],
                    "notStated": row["notStated"],
                    "base": row["activation"],
                    "variant": round(variant_by_index.get(row["index"], 0.0), 4),
                    "delta": round(variant_by_index.get(row["index"], 0.0) - row["activation"], 4),
                }
                for row in rows
                if row.get("notStated")
            ]
            contrasts[pid] = {
                "request": contrast_meta["request"],
                "changed": contrast_meta["changed"],
                "modelChoice": _model_choice_from_step(contrast_step),
                "themeEvidence": contrast_features.get("theme_evidence") or None,
                "tracked": tracked,
                **contrast_diff(active, variant_active, labels, contrast_meta["request"]),
            }

    # --- recommendations: authoritative tool-grounded conclusions ---
    # The per-tool generations are useful for SAE inspection, but they are not
    # an authoritative source for facts displayed to the user.  Keep the UI in
    # sync with the grounded final report assembled above.
    pro_quote_data = _TOOLS["ProQuote"][0]
    parts_data = _TOOLS["PartsSearch"][0]

    recommendations: dict[str, dict] = {}
    for p in _PROBLEMS:
        pid = p["id"]
        meta = _PROBLEM_META.get(pid, {})
        pq = pro_quote_data.get(pid, {})
        pd_parts = parts_data.get(pid, {})

        # Determine verdict from difficulty/urgency
        difficulty_level = meta.get("difficulty", {}).get("level", "yellow")
        if difficulty_level == "red":
            verdict, verdict_label = "pro", "Call a Professional"
        elif difficulty_level == "green":
            verdict, verdict_label = "diy", "Easy DIY Fix"
        else:
            verdict, verdict_label = "diy", "DIY Repair"

        # Cost ranges from tool data
        diy_cost = pd_parts.get("diy_cost_range", "")
        estimates = pq.get("repair_estimates", [])
        if estimates:
            lo = min(e["total"] for e in estimates)
            hi = max(e["total"] for e in estimates)
            pro_cost = f"${lo}–${hi}" if lo != hi else f"${lo}"
        else:
            pro_cost = ""

        recommendations[pid] = {
            "verdict": verdict,
            "verdictLabel": verdict_label,
            "diyCost": diy_cost,
            "proCost": pro_cost,
            "rationale": _GROUNDED_RATIONALES[pid],
        }

    # --- themes: contrast-type definitions + contrastive-map evidence ---
    themes = []
    for theme_id, tmeta in _THEME_META.items():
        evidence = {}
        for p in _PROBLEMS:
            theme_evidence = decision_features[p["id"]].get("themeEvidence") or {}
            if theme_id in theme_evidence:
                evidence[p["id"]] = theme_evidence[theme_id]
        themes.append(
            {
                "id": theme_id,
                "title": tmeta["title"],
                "description": _SCENARIO.get("contrast_types", {}).get(theme_id, ""),
                "leftLabel": tmeta["anchorLabel"],
                "rightLabel": tmeta["contrastLabel"],
                "evidence": evidence,
            }
        )

    run_metadata = {
        "model": Path(model_name).name,
        "saeLayer": sae_layer,
        "thresholdOffset": threshold_offset,
    }
    if analysis.get("backend"):
        run_metadata["backend"] = analysis["backend"]
    if analysis.get("logprobs_mode"):
        run_metadata["logprobsMode"] = analysis["logprobs_mode"]
    hf_summary = hf_parity_summary(steering)
    if hf_summary:
        run_metadata["hf"] = hf_summary

    ui_data = {
        "runMetadata": run_metadata,
        "problems": problems,
        "toolResults": tool_results,
        "decisionFeatures": decision_features,
        "recommendations": recommendations,
        "themes": themes,
    }
    if contrasts:
        ui_data["contrasts"] = contrasts
    if probes:
        ui_data["probes"] = probes
    if open_requests:
        ui_data["openRequests"] = open_requests
    return ui_data


# ---------------------------------------------------------------------------
# Section F: Explanation generation
# ---------------------------------------------------------------------------


def _build_feature_summary(analysis_results: dict) -> str:
    lines = []
    sae_layer_key = analysis_results.get("sae_layer_key")
    for step_info in analysis_results["steps"]:
        lines.append(f"## {step_info['step']}")

        for layer, stats in step_info.get("raw_stats", {}).items():
            if layer == sae_layer_key:
                lines.append(
                    f"  L2={stats['l2_norm']:.2f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}"
                )

        sae = step_info.get("sae_features")
        if sae:
            lines.append(
                f"  Active SAE features: {sae['num_active']}/{sae['total_features']} "
                f"({100 - sae['sparsity_pct']:.1f}% active)"
            )
            for feat in sae["top_features"][:5]:
                label = feat.get("label", f"Feature #{feat['index']}")
                desc = feat.get("description", "")
                act = feat["activation"]
                line = f"    - {label} (activation={act:.4f})"
                if desc:
                    line += f": {desc}"
                lines.append(line)
        lines.append("")
    return "\n".join(lines)


def generate_decision_explanations(
    engine: HFEngine, analysis_results: dict, agent_output: str
) -> tuple[str, str]:
    """Generate a technical and a plain-language explanation."""

    feature_summary = _build_feature_summary(analysis_results)

    # Technical explanation
    technical_prompt = engine._build_prompt(
        (
            "You are an AI interpretability researcher. You have access to "
            "Sparse Autoencoder (SAE) analysis of a home repair AI advisor's "
            "internal activations captured at each decision step. Explain what "
            "the activation patterns reveal about HOW the AI made its decisions."
        ),
        (
            "A home repair AI advisor analyzed three household problems "
            "(dishwasher leak, stuck garbage disposal, noisy water heater). "
            "Here is its final recommendation:\n\n"
            f"{agent_output[:2000]}\n\n"
            "Here is the SAE feature analysis at each step "
            "(steps are labeled as PROBLEM_ToolName):\n\n"
            f"{feature_summary}\n\n"
            "Please explain:\n"
            "1. What patterns in the agent's internal representations drove "
            "its decisions?\n"
            "2. How did the active features change across problem types and "
            "tool types?\n"
            "3. What does this reveal about how the agent distinguishes "
            "DIY-safe repairs from those requiring professionals?"
        ),
    )
    technical = engine.generate(technical_prompt, "explain_technical", max_tokens=1024)

    # Layman explanation
    feature_evidence_lines = []
    for step_info in analysis_results["steps"]:
        sae = step_info.get("sae_features")
        if not sae:
            continue
        top3 = sae["top_features"][:3]
        labeled = [f for f in top3 if f.get("label")]
        if not labeled:
            continue
        parts = ", ".join(f'"{f["label"]}" (strength {f["activation"]:.1f})' for f in labeled)
        step = step_info["step"]
        step_readable = step.replace("_", " ").lower()
        if "final" in step_readable:
            step_readable = "making the final recommendation"
        else:
            step_readable = f"analyzing {step_readable}"
        feature_evidence_lines.append(
            f"- While {step_readable}: strongest brain signals were {parts}"
        )
    feature_evidence = "\n".join(feature_evidence_lines) or (
        "- No labeled features were available."
    )

    layman_prompt = engine._build_prompt(
        (
            "You explain AI decisions in plain, everyday language. No jargon, "
            "no technical terms. Write as if explaining to someone with no "
            "technical background."
        ),
        (
            "An AI home repair advisor was asked to help with three problems: "
            "a leaking dishwasher, a stuck garbage disposal, and a noisy water "
            "heater. Here is what it recommended:\n\n"
            f"{agent_output[:1500]}\n\n"
            "We looked inside the AI's brain at each step. "
            "Here is what we found:\n\n"
            f"{feature_evidence}\n\n"
            "Using the brain signals above, explain in 4-6 plain sentences "
            "what was going through the AI's mind when making these "
            "recommendations. Reference which signals were strongest and "
            "how they differed between the three repair problems. "
            "Start with 'The AI advisor decided'."
        ),
    )
    layman = engine.generate(layman_prompt, "explain_layman", max_tokens=512)

    return technical, layman


# ---------------------------------------------------------------------------
# Section F: Summary printing + file output
# ---------------------------------------------------------------------------


def render_markdown(text: str, width: int = 100) -> str:
    """Render markdown text for terminal display with table formatting."""
    lines = text.split("\n")
    output = []
    table_buffer: list[str] = []
    in_table = False

    def flush_table():
        if not table_buffer:
            return
        rows = []
        for line in table_buffer:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        if len(rows) < 2:
            output.extend(table_buffer)
            return
        header = rows[0]
        data_rows = [r for r in rows[1:] if not all(set(c.strip()) <= {"-", ":"} for c in r)]
        n_cols = len(header)
        col_widths = [len(h) for h in header]
        for row in data_rows:
            for i, cell in enumerate(row[:n_cols]):
                col_widths[i] = max(col_widths[i], len(cell))
        col_widths = [min(w, 50) for w in col_widths]

        def make_row(cells, widths):
            parts = []
            for cell, w in zip(cells, widths, strict=False):
                if len(cell) > w:
                    cell = cell[: w - 3] + "..."
                parts.append(cell.ljust(w))
            return "| " + " | ".join(parts) + " |"

        def make_separator(widths):
            return "+-" + "-+-".join("-" * w for w in widths) + "-+"

        output.append(make_separator(col_widths))
        output.append(make_row(header, col_widths))
        output.append(make_separator(col_widths))
        for row in data_rows:
            row = row + [""] * (n_cols - len(row))
            output.append(make_row(row, col_widths))
        output.append(make_separator(col_widths))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                table_buffer = []
            table_buffer.append(line)
        else:
            if in_table:
                flush_table()
                table_buffer = []
                in_table = False
            if stripped.startswith("**") and stripped.endswith("**"):
                output.extend(["", stripped, ""])
            elif stripped.startswith("* ") or stripped.startswith("- "):
                output.append(textwrap.fill(stripped, width=width, subsequent_indent="  "))
            elif stripped:
                output.append(textwrap.fill(stripped, width=width))
            else:
                output.append("")

    if in_table:
        flush_table()

    return "\n".join(output)


def print_analysis_summary(analysis: dict):
    print(f"\n{'=' * 60}")
    print("  ACTIVATION ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Decision steps captured: {len(analysis['steps'])}")
    print(f"  SAE available: {analysis['sae_available']}")
    print(f"  Feature labels available: {analysis['features_available']}")
    print(f"  SAE layer: {analysis['sae_layer']}")
    sae_layer_key = analysis["sae_layer_key"]

    for step_info in analysis["steps"]:
        print(f"\n  --- {step_info['step']} ---")
        for layer, stats in step_info.get("raw_stats", {}).items():
            if layer == sae_layer_key:
                print(f"    L2 norm: {stats['l2_norm']:.2f}, std: {stats['std']:.4f}")
        sae = step_info.get("sae_features")
        if sae:
            print(
                f"    Active features: {sae['num_active']}/{sae['total_features']} "
                f"({100 - sae['sparsity_pct']:.1f}%)"
            )
            for feat in sae["top_features"][:5]:
                label = feat.get("label", f"Feature #{feat['index']}")
                themes = feat.get("themes", [])
                theme_str = ""
                if themes:
                    theme_names = [t["theme"] for t in themes[:2]]
                    theme_str = f"  [{', '.join(theme_names)}]"
                print(f"      {label}: {feat['activation']:.4f}{theme_str}")
            # Show contrastive theme evidence
            theme_evidence = sae.get("theme_evidence") or {}
            if theme_evidence:
                print("    Theme evidence (position toward contrast side, 0-1):")
                for theme, info in theme_evidence.items():
                    flag = " (insufficient)" if info.get("insufficient") else ""
                    print(
                        f"      {theme}: position={info['position']:.2f} "
                        f"({info['nFeatures']} features, coverage {info['coverage']:.1%}){flag}"
                    )
        decision = step_info.get("decision")
        if decision:
            print(
                f"    Model tool choice: {decision.get('display')} "
                f"(p={decision.get('prob', 0):.2f}, coverage {decision.get('coverage', 0):.2f})"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analysis_from_evaluation_report(report: dict, sae_layer: int) -> dict:
    """Adapt a modified-vLLM layer evaluation into the demo analysis shape.

    This lets the HTML use activations from the same backend that trained and
    validated the SAEs, without loading the subject model a second time through
    HuggingFace merely to construct ``ui_data.json``.

    Accepts both the legacy report (three base prompts, ``top_features`` only)
    and the current one (base + contrast prompts, ``active_features``,
    ``theme_evidence`` and a tool-choice ``decisions`` list).
    """

    def _key(prompt: dict) -> tuple[str, str, str]:
        return (prompt["step"], prompt.get("kind", "base"), prompt["request"])

    prompts = report.get("prompts", [])
    report_keys = [_key(prompt) for prompt in prompts]
    base_keys = [_key(prompt) for prompt in decision_prompts(include_contrasts=False)]
    full_keys = [_key(prompt) for prompt in decision_prompts(include_contrasts=True)]
    probe_keys = [
        _key(prompt) for prompt in decision_prompts(include_contrasts=True, include_probes=True)
    ]
    if report_keys not in (base_keys, full_keys, probe_keys):
        raise ValueError(
            "The evaluation report prompts do not match the current home-repair "
            "demo decisions. Rerun compare_sae_backends.py before building the UI."
        )

    layer_result = next(
        (result for result in report.get("layers", []) if result.get("layer") == sae_layer),
        None,
    )
    if layer_result is None:
        raise ValueError(f"Evaluation report has no result for SAE layer {sae_layer}.")

    top_features = layer_result.get("top_features", [])
    per_prompt_l0 = layer_result.get("l0", {}).get("per_prompt", [])
    if len(top_features) != len(prompts) or len(per_prompt_l0) != len(prompts):
        raise ValueError("Evaluation report has inconsistent prompt, feature, or L0 row counts.")
    active_features = layer_result.get("active_features") or [None] * len(prompts)
    theme_evidence = layer_result.get("theme_evidence") or [None] * len(prompts)
    if len(active_features) != len(prompts) or len(theme_evidence) != len(prompts):
        raise ValueError("Evaluation report active_features/theme_evidence rows do not align.")

    decisions_by_step: dict[str, dict] = {}
    for decision in report.get("decisions") or []:
        if decision and decision.get("step"):
            decisions_by_step[decision["step"]] = decision

    d_sae = int(layer_result["d_sae"])
    steps = []
    for prompt, features, active, evidence, active_count in zip(
        prompts, top_features, active_features, theme_evidence, per_prompt_l0, strict=True
    ):
        steps.append(
            {
                "step": prompt["step"],
                "kind": prompt.get("kind", "base"),
                "problem": prompt.get("problem"),
                "request": prompt["request"],
                "sae_features": {
                    "num_active": int(active_count),
                    "total_features": d_sae,
                    "sparsity_pct": (1.0 - int(active_count) / d_sae) * 100.0,
                    "top_features": features,
                    "active_features": active or features,
                    "theme_evidence": evidence,
                },
                "decision": decisions_by_step.get(prompt["step"]),
            }
        )

    return {
        "steps": steps,
        "sae_available": True,
        "features_available": any(top_features),
        "sae_layer": sae_layer,
        "sae_layer_key": f"residual_{sae_layer}",
        "sae_source": report.get("model", ""),
        "sae_threshold_offset": report.get("threshold_offset", 0.0),
        "backend": report.get("backend", "vllm"),
        "logprobs_mode": report.get("logprobs_mode"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Home Repair Agent Demo with SAE Activation Analysis"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=_MODEL_NAME,
        help=f"HuggingFace model ID or local model directory (default: {_MODEL_NAME})",
    )
    parser.add_argument(
        "--sae-repo-id",
        type=str,
        default=_SAE_REPO_ID,
        help=f"HuggingFace repo ID for the SAE (default: {_SAE_REPO_ID})",
    )
    parser.add_argument(
        "--sae-local-dir",
        type=str,
        default=_DEFAULT_SAE_LOCAL_DIR,
        help="Load SAE from local pipeline output directory instead of HF Hub "
        f"(default: {_DEFAULT_SAE_LOCAL_DIR or 'Hub fallback'})",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=_SAE_LAYER,
        help=f"SAE layer to load from the HF repo (default: {_SAE_LAYER})",
    )
    parser.add_argument(
        "--sae-threshold-offset",
        type=float,
        default=_HF_THRESHOLD_OFFSET,
        help="Additive threshold calibration applied only to the in-memory SAE "
        f"(default: +{_HF_THRESHOLD_OFFSET:.7f}; use 0 to disable)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=_TRAINED_LAYERS,
        help=f"Transformer layers to hook (default: {_TRAINED_LAYERS})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum tokens to generate per LLM call",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cuda:0', 'mps', 'cpu'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (use float16 for MPS, float32 for CPU)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEMO_DIR / "output"),
        help="Directory for JSON and text results",
    )
    parser.add_argument(
        "--youtube-api-key",
        type=str,
        default=None,
        help="YouTube Data API v3 key for real tutorial search (optional)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Generate LLM-based explanations in Phase 4 (needs extra GPU memory)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Allow reasoning-mode generation. Disabled by default so internal "
        "thinking is not emitted in demo output.",
    )
    parser.add_argument(
        "--ui-from-evaluation",
        type=str,
        default=None,
        metavar="REPORT_JSON",
        help="Build ui_data.json from a modified-vLLM SAE evaluation report "
        "without loading the subject model",
    )
    parser.add_argument(
        "--steering",
        type=str,
        default=None,
        metavar="STEERING_JSON",
        help="Optional steering_results.json (from steer_tool_choice.py) to merge "
        "into ui_data.json",
    )
    args = parser.parse_args()

    steering = None
    if args.steering:
        steering = json.loads(Path(args.steering).read_text())

    if args.ui_from_evaluation:
        report_path = Path(args.ui_from_evaluation)
        report = json.loads(report_path.read_text())
        analysis = analysis_from_evaluation_report(report, args.sae_layer)
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ui_data = build_ui_data(
            analysis=analysis,
            per_problem={},
            final_recommendation="",
            youtube_api_key=args.youtube_api_key,
            model_name=report.get("model", args.model_name),
            sae_layer=args.sae_layer,
            threshold_offset=report.get("threshold_offset", 0.0),
            steering=steering,
        )
        ui_data = attach_spec_sheet(ui_data, "home_repair", args.sae_layer)
        ui_path = output_path / "ui_data.json"
        ui_path.write_text(json.dumps(ui_data, indent=2, ensure_ascii=False) + "\n")
        print(f"Built {ui_path} from {report_path} (SAE layer {args.sae_layer}).")
        return

    print("=" * 60)
    print("  Home Repair Agent Demo")
    print("  Nemotron-3.5-Nano-30B (HuggingFace) + SAE Activation Analysis")
    print(f"  Subject model: {args.model_name}")
    sae_source = args.sae_local_dir or args.sae_repo_id
    print(f"  SAE: {sae_source} (layer {args.sae_layer})")
    print(f"  HF SAE threshold offset: {args.sae_threshold_offset:+.7f}")
    print("=" * 60)

    # Phase 1: Run scripted multi-step analysis
    print("\n[Phase 1] Running home repair analysis...")
    engine = HFEngine(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        allow_thinking=args.thinking,
    )
    final_recommendation, per_problem = run_home_repair_analysis(engine, args.youtube_api_key)

    # Contrast variants: one edited-wording prompt per problem, recorded after
    # the scripted analysis so the decision log keeps its base-first order.
    for prompt in decision_prompts(include_contrasts=True):
        if prompt["kind"] == "contrast":
            engine.record_tool_decision(prompt["step"], prompt["request"])

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete. {len(engine.prompt_log)} decision points recorded.")
    print(f"{'=' * 60}")
    print(f"\n  FINAL RECOMMENDATION:\n{final_recommendation}")

    # Phase 2: Extract activations (reuses same model, forward pass with hooks)
    print("\n[Phase 2] Extracting activations...")
    hook_layers = list(dict.fromkeys([*args.layers, args.sae_layer]))
    if args.sae_layer not in args.layers:
        print(f"  Adding SAE layer {args.sae_layer} to the extraction layers.")
    activation_log = engine.extract_all_prompts(hook_layers)

    # Free generation KV cache before SAE analysis
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 3: Analyze activations through SAE
    print("\n[Phase 3] Analyzing activations through SAE...")
    analysis = analyze_activations(
        activation_log=activation_log,
        sae_repo_id=args.sae_repo_id,
        sae_layer=args.sae_layer,
        sae_local_dir=args.sae_local_dir,
        threshold_offset=args.sae_threshold_offset,
    )
    print_analysis_summary(analysis)

    # Phase 4: Generate explanations (reuses same model)
    # Skip by default on single-GPU setups — the explanation prompts are
    # large and the naive Mamba path (no causal-conv1d) is memory-hungry.
    # Use --explain to opt in.
    technical, layman = "", ""
    if not args.explain:
        print("\n[Phase 4] Skipped explanation generation (pass --explain to enable).")
    elif analysis["sae_available"]:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n[Phase 4] Generating decision explanations...")
        technical, layman = generate_decision_explanations(engine, analysis, final_recommendation)
        print(f"\n{'=' * 60}")
        print("  TECHNICAL EXPLANATION")
        print(f"{'=' * 60}")
        print(render_markdown(technical))
        print(f"\n{'=' * 60}")
        print("  PLAIN LANGUAGE SUMMARY")
        print(f"{'=' * 60}")
        print(render_markdown(layman))
    else:
        print("\n[Phase 4] Skipped explanation generation (no SAE available).")

    # Cleanup model
    engine.cleanup()

    # Phase 5: Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "analysis_results.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    with open(output_path / "agent_output.txt", "w") as f:
        f.write(final_recommendation)
    with open(output_path / "per_problem_analyses.json", "w") as f:
        json.dump(per_problem, f, indent=2)
    if technical:
        with open(output_path / "technical_explanation.txt", "w") as f:
            f.write(technical)
    if layman:
        with open(output_path / "layman_explanation.txt", "w") as f:
            f.write(layman)

    # Phase 6: Generate UI data for index.html
    print("\n[Phase 6] Generating UI data...")
    ui_data = build_ui_data(
        analysis=analysis,
        per_problem=per_problem,
        final_recommendation=final_recommendation,
        youtube_api_key=args.youtube_api_key,
        model_name=args.model_name,
        sae_layer=args.sae_layer,
        threshold_offset=args.sae_threshold_offset,
        steering=steering,
    )
    ui_data = attach_spec_sheet(ui_data, "home_repair", args.sae_layer)
    with open(output_path / "ui_data.json", "w") as f:
        json.dump(ui_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {output_path}/")
    print("  Open demo/home_repair/index.html to view the interactive explanation.")
    print("\n  Done.")


if __name__ == "__main__":
    main()
