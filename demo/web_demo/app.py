#!/usr/bin/env python3
"""
Web Demo: Interactive Home Repair AI Advisor with SAE Explanations.

Integrated web application that:
  1. Accepts home repair requests via a browser form
  2. Runs a CrewAI multi-agent workflow backed by Nemotron-3-Nano-30B
  3. Extracts activations and runs SAE analysis to explain the AI's reasoning
  4. Streams progress and renders an interactive visualization

Prerequisites:
    pip install fastapi uvicorn crewai litellm
    # Plus kiji-inspector core dependencies

Usage:
    uv run python demo/web_demo/app.py
    uv run python demo/web_demo/app.py --device cuda --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import queue
import re
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from kiji_inspector.core.sae import SAE

# Try importing CrewAI + litellm (will fail gracefully with a clear message)
try:
    import litellm
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool as crewai_tool

    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_AVAILABLE = False
    _IMPORT_ERR = str(e)


# ---------------------------------------------------------------------------
# A. Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_SAE_REPO_ID = (
    "davidnet/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-Home-scenarios"
)
_SAE_LAYER = 20

# Regex to extract the content after "Final Answer:" from CrewAI task outputs.
_FINAL_ANSWER_RE = re.compile(
    r"Final Answer\s*:\s*(.+)",
    re.DOTALL,
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def _clean_model_output(text: str) -> str:
    """Strip <think> blocks and trailing junk from model output."""
    # Remove closed <think>...</think> blocks
    text = _THINK_RE.sub("", text)
    # Remove unclosed <think> (model hit max tokens mid-thought)
    text = _THINK_OPEN_RE.sub("", text)
    # Remove orphaned </think> (pre-fill skips <think> but model emits closing tag)
    text = text.replace("</think>", "")
    return text.strip()


def _strip_scaffolding(text: str) -> str:
    """Extract only the Final Answer content from CrewAI task output.

    CrewAI task output strings embed descriptions, tool instructions, and
    ReAct scaffolding.  The model confuses these with real instructions at
    synthesis time.  This extracts only the actual findings (the text after
    "Final Answer:").  If no Final Answer is found, falls back to removing
    known scaffolding lines.
    """
    # Try to extract just the Final Answer portion
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: strip known scaffolding lines
    scaffolding_re = re.compile(
        r"^(Thought|Action|Action Input|Observation|Final Answer)\s*:.*$",
        re.MULTILINE,
    )
    cleaned = scaffolding_re.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_SYSTEM_PROMPT = (
    "You are an experienced home repair advisor. You help homeowners diagnose "
    "appliance problems and decide whether to attempt a DIY repair or hire a "
    "professional. Always consider safety first, then cost, difficulty, and "
    "warranty implications. Reference specific details from the data provided."
)


# ---------------------------------------------------------------------------
# B. HFEngine — single HuggingFace model for generation + activation capture
# ---------------------------------------------------------------------------


class HFEngine:
    """HuggingFace model for text generation AND activation extraction."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 400,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.prompt_log: list[tuple[str, str]] = []

        torch_dtype = getattr(torch, dtype)

        print(f"  Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"torch_dtype": torch_dtype}
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        self._input_device = self._find_input_device()
        self.hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(
            self.model.config, "text_config", self.model.config
        ).hidden_size
        print(f"  Model ready on {self._input_device} ({torch_dtype})")

    def _find_input_device(self) -> torch.device:
        for attr_path in (
            "language_model.model.embed_tokens",
            "language_model.embed_tokens",
            "model.embed_tokens",
            "transformer.wte",
        ):
            obj = self.model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return next(obj.parameters()).device
            except (AttributeError, StopIteration):
                continue
        return next(self.model.parameters()).device

    def _get_model_layers(self):
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
            if hasattr(lm, "layers"):
                return lm.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError(
            f"Cannot locate transformer layers for {type(self.model).__name__}"
        )

    def _get_inner_model(self):
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            return lm.model if hasattr(lm, "model") else lm
        return self.model.model if hasattr(self.model, "model") else self.model

    def _build_prompt(self, system: str, user: str) -> str:
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate(
        self, prompt: str, step_label: str, max_tokens: int | None = None
    ) -> str:
        self.prompt_log.append((step_label, prompt))
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
        print(f"  [{step_label}] Generated {len(new_tokens)} tokens")
        return response

    def extract_all_prompts(
        self, layers: list[int]
    ) -> list[tuple[str, dict[str, np.ndarray]]]:
        model_layers = self._get_model_layers()
        activations: dict[str, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def _make_hook(name: str):
            def hook(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                activations[name] = act.detach().cpu().to(torch.float32)

            return hook

        for idx in layers:
            if idx < len(model_layers):
                h = model_layers[idx].register_forward_hook(
                    _make_hook(f"residual_{idx}")
                )
                hooks.append(h)

        inner = self._get_inner_model()
        results: list[tuple[str, dict[str, np.ndarray]]] = []

        for step_label, prompt in self.prompt_log:
            activations.clear()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
            with torch.no_grad():
                inner(**inputs)
            result = {
                name: act[:, -1, :].squeeze(0).numpy()
                for name, act in activations.items()
            }
            results.append((step_label, result))
            print(f"    {step_label}: {len(result)} layers extracted")

        for h in hooks:
            h.remove()
        return results

    def clear_log(self):
        self.prompt_log.clear()

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


# ---------------------------------------------------------------------------
# C. Mock tool data (same as home_repair demo)
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
        "tools_needed": [
            "Phillips screwdriver",
            "Torx T20 driver",
            "flashlight",
            "towels",
        ],
    },
    "disposal_stuck": {
        "model": "InSinkErator Badger 5",
        "possible_causes": [
            "Jammed flywheel -- foreign object wedged between impellers",
            "Thermal overload tripped (red reset button on bottom of unit)",
            "Capacitor failure -- motor hums but cannot start rotation (less common)",
        ],
        "diy_difficulty": "Easy",
        "safety": "NEVER put hand inside disposal. Ensure power is OFF at breaker.",
        "tools_needed": ["1/4-inch Allen wrench", "flashlight", "tongs or pliers"],
        "quick_fix": (
            "Insert 1/4-inch Allen wrench into hex socket on bottom center. "
            "Rotate back and forth to free jam. Remove debris with tongs. "
            "Press red reset button. Restore power and test."
        ),
    },
    "water_heater_noise": {
        "model": "Rheem Performance 50-Gal Gas XR50T06EC36U1",
        "possible_causes": [
            "Sediment buildup on tank bottom -- mineral deposits heat and pop",
            "Anode rod depleted -- sacrificial rod no longer protecting tank",
            "Scale buildup on burner assembly reducing heat transfer",
            "Possible tank corrosion if popping is severe and rust is present",
        ],
        "diy_difficulty": "Moderate to Difficult",
        "safety": (
            "GAS APPLIANCE: Risk of scalding and gas leaks. Turn off gas "
            "supply valve before any work. If you smell gas, leave immediately."
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
            {"name": "Door Gasket Seal (OEM)", "price": 42.99, "in_stock": True},
            {"name": "Drain Pump Assembly", "price": 89.50, "in_stock": True},
            {"name": "Water Inlet Valve", "price": 55.75, "in_stock": True},
            {"name": "Spray Arm Seal Kit", "price": 12.99, "in_stock": True},
        ],
        "diy_cost_range": "$13 - $90 depending on which part has failed",
    },
    "disposal_stuck": {
        "parts": [
            {"name": "Self-Service Wrench Kit", "price": 7.99, "in_stock": True},
            {"name": "Badger 5 Replacement Unit", "price": 99.00, "in_stock": True},
        ],
        "diy_cost_range": "$0 - $8 if jam clears; $99 + install if motor is dead",
    },
    "water_heater_noise": {
        "parts": [
            {"name": "Aluminum Anode Rod", "price": 29.99, "in_stock": True},
            {"name": "Tank Flush Kit", "price": 14.99, "in_stock": True},
            {"name": "Drain Valve Replacement", "price": 11.49, "in_stock": True},
            {"name": "Rheem 50-Gal Replacement Unit", "price": 649.00, "in_stock": True},
        ],
        "diy_cost_range": "$15-$45 maintenance; $649+ if tank replacement needed",
    },
}

_TUTORIAL_DATA = {
    "dishwasher_leak": {
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
        ],
    },
    "disposal_stuck": {
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
        ],
    },
    "water_heater_noise": {
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
        ],
    },
}

_PRO_QUOTE_DATA = {
    "dishwasher_leak": {
        "diagnosis_fee": 89,
        "repair_estimates": [
            {"repair": "Door gasket replacement", "total": 163, "time": "1 hour"},
            {"repair": "Pump seal replacement", "total": 270, "time": "1.5 hours"},
        ],
        "warranty_on_repair": "90-day parts and labor",
        "urgency": "Moderate -- continued use risks water damage to flooring",
        "next_available": "2-3 business days",
    },
    "disposal_stuck": {
        "diagnosis_fee": 75,
        "repair_estimates": [
            {"repair": "Clear jam + reset", "total": 75, "time": "30 min"},
            {"repair": "Full unit replacement", "total": 249, "time": "1.5 hours"},
        ],
        "warranty_on_repair": "90-day labor",
        "urgency": "Low -- disposal is non-essential; sink still drains",
        "next_available": "3-5 business days",
    },
    "water_heater_noise": {
        "diagnosis_fee": 95,
        "repair_estimates": [
            {"repair": "Tank flush + anode rod replacement", "total": 245, "time": "2h"},
            {"repair": "Full unit replacement (50-gal gas)", "total": 1099, "time": "4-6h"},
        ],
        "warranty_on_repair": "1-year labor, 6-year tank on new unit",
        "urgency": "Moderate-High -- sediment reduces efficiency; corrosion risk",
        "next_available": "1-2 business days (prioritized for gas appliances)",
    },
}

_TOOL_SOURCES = {
    "ManualCheck": _MANUAL_DATA,
    "PartsSearch": _PARTS_DATA,
    "TutorialSearch": _TUTORIAL_DATA,
    "ProQuote": _PRO_QUOTE_DATA,
}


def _match_problem(query: str) -> str:
    """Match a free-text query to the closest predefined problem id."""
    query_lower = query.lower()
    keywords = {
        "dishwasher_leak": [
            "dishwasher",
            "leak",
            "water pool",
            "bosch",
            "wash cycle",
        ],
        "disposal_stuck": [
            "disposal",
            "garbage",
            "hum",
            "stuck",
            "insinkerator",
            "won't spin",
        ],
        "water_heater_noise": [
            "water heater",
            "heater",
            "popping",
            "rumbling",
            "rheem",
            "gas",
            "noise",
            "hot water",
        ],
    }
    scores = {
        pid: sum(1 for kw in kws if kw in query_lower)
        for pid, kws in keywords.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "dishwasher_leak"  # default fallback
    return best


# ---------------------------------------------------------------------------
# D. Custom LiteLLM provider wrapping HFEngine
# ---------------------------------------------------------------------------

# Global engine reference (set during startup)
_engine: HFEngine | None = None
_step_counter: int = 0
_progress_queue: queue.Queue | None = None


def _reset_step_counter():
    global _step_counter
    _step_counter = 0


def _nemotron_completion(model: str, messages: list[dict], **kwargs) -> dict:
    """Generate a completion using the local HFEngine."""
    global _step_counter
    assert _engine is not None, "Model not loaded"

    # Extract messages into system + user
    system_msg = _SYSTEM_PROMPT
    user_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            system_msg = content
        elif role in ("user", "human"):
            user_parts.append(content)
        elif role == "assistant":
            # Include prior assistant context for multi-turn
            user_parts.append(f"[Previous response]: {content[:500]}")

    user_msg = "\n\n".join(user_parts) if user_parts else "Please analyze this."
    prompt = _engine._build_prompt(system_msg, user_msg)

    # Determine step label by searching ALL message content for tool names.
    # CrewAI may place tool info in system, user, or combined messages.
    _step_counter += 1
    step_label = f"step_{_step_counter}"
    all_text = system_msg + "\n" + user_msg
    # Match tool names case-insensitively (CrewAI may lowercase them)
    _tool_names_ordered = [
        ("Manual Check", ["manualcheck", "manual_check"]),
        ("Parts Search", ["partssearch", "parts_search"]),
        ("Tutorial Search", ["tutorialsearch", "tutorial_search"]),
        ("Pro Quote", ["proquote", "pro_quote"]),
    ]
    all_lower = all_text.lower()
    for tool_label, variants in _tool_names_ordered:
        if any(v in all_lower for v in variants):
            step_label = tool_label
            break
    else:
        if "synthesize" in all_lower or "recommendation" in all_lower:
            step_label = "Synthesis"

    if _progress_queue:
        _progress_queue.put(
            {"type": "step", "label": step_label, "status": "generating"}
        )

    response_text = _engine.generate(prompt, step_label)

    if _progress_queue:
        _progress_queue.put(
            {"type": "step", "label": step_label, "status": "complete"}
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": response_text, "role": "assistant"},
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": len(response_text.split()),
            "total_tokens": 100 + len(response_text.split()),
        },
    }


def _setup_litellm():
    """Register the Nemotron local provider with litellm."""
    try:
        # litellm >= 1.40: use CustomLLM
        from litellm import CustomLLM, ModelResponse

        class NemotronLLM(CustomLLM):
            def completion(self, model, messages, **kwargs):
                raw = _nemotron_completion(model, messages, **kwargs)
                return ModelResponse(**raw)

        handler = NemotronLLM()
        litellm.custom_provider_map = [
            {"provider": "nemotron-local", "custom_handler": handler}
        ]
        print("  Registered nemotron-local provider via CustomLLM")
    except (ImportError, AttributeError):
        # Fallback: monkey-patch litellm.completion
        from litellm import ModelResponse

        _original = litellm.completion

        def _patched(*args, **kwargs):
            model = kwargs.get("model", args[0] if args else "")
            if "nemotron-local" in str(model):
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                raw = _nemotron_completion(model, messages, **kwargs)
                return ModelResponse(**raw)
            return _original(*args, **kwargs)

        litellm.completion = _patched
        print("  Registered nemotron-local provider via monkey-patch")


# ---------------------------------------------------------------------------
# E. CrewAI tools, agent, and crew factory
# ---------------------------------------------------------------------------


def _make_tools(problem_id: str):
    """Create CrewAI tool functions bound to a specific problem."""

    @crewai_tool("ManualCheck")
    def manual_check(query: str) -> str:
        """Look up appliance troubleshooting guides, error codes, manufacturer
        diagnostic steps, and safety warnings for a home appliance problem."""
        pid = problem_id or _match_problem(query)
        data = _MANUAL_DATA.get(pid, list(_MANUAL_DATA.values())[0])
        return json.dumps(data, indent=2)

    @crewai_tool("PartsSearch")
    def parts_search(query: str) -> str:
        """Search for replacement parts with pricing, availability, and
        compatibility information for a home appliance."""
        pid = problem_id or _match_problem(query)
        data = _PARTS_DATA.get(pid, list(_PARTS_DATA.values())[0])
        return json.dumps(data, indent=2)

    @crewai_tool("TutorialSearch")
    def tutorial_search(query: str) -> str:
        """Find video tutorials and step-by-step repair guides with difficulty
        ratings, required tools, and estimated completion times."""
        pid = problem_id or _match_problem(query)
        data = _TUTORIAL_DATA.get(pid, list(_TUTORIAL_DATA.values())[0])
        return json.dumps(data, indent=2)

    @crewai_tool("ProQuote")
    def pro_quote(query: str) -> str:
        """Get professional repair service quotes including labor costs,
        turnaround times, service warranties, and urgency assessment."""
        pid = problem_id or _match_problem(query)
        data = _PRO_QUOTE_DATA.get(pid, list(_PRO_QUOTE_DATA.values())[0])
        return json.dumps(data, indent=2)

    return [manual_check, parts_search, tutorial_search, pro_quote]


def create_crew(problem_info: dict) -> Crew:
    """Create a CrewAI crew for analyzing a home repair problem.

    Uses separate tasks per tool so the local Nemotron model doesn't need to
    drive a multi-step ReAct loop (which it struggles with).  Each task gets
    exactly one tool so a single Action/Observation cycle completes it.
    A final synthesis task (no tools) writes the recommendation.
    """
    from crewai import LLM

    llm = LLM(model="nemotron-local/nemotron-3-nano-30b", temperature=0.7)

    problem_id = _match_problem(
        f"{problem_info.get('appliance', '')} {problem_info.get('details', '')}"
    )
    tools = _make_tools(problem_id)
    # tools order: ManualCheck, PartsSearch, TutorialSearch, ProQuote
    manual_tool, parts_tool, tutorial_tool, quote_tool = tools

    user_desc = (
        f"Appliance: {problem_info.get('appliance', 'Unknown')}\n"
        f"Age: {problem_info.get('age', 'Unknown')}\n"
        f"Problem: {problem_info.get('details', 'No details provided')}"
    )

    # -- Custom prompt templates for local Nemotron model --
    # CrewAI's default ReAct template confuses Nemotron.  We supply simple,
    # explicit templates so the model knows exactly what to do.

    # Template for tool-using agents: call the ONE tool, then summarize.
    _tool_system = (
        "You are {role}. {backstory}\n"
        "Your personal goal is: {goal}\n\n"
        "You have access to these tools:\n{tools}\n\n"
        "INSTRUCTIONS: You MUST call the tool ONCE, then summarize the "
        "result. Use EXACTLY this format:\n\n"
        "Thought: I will call the tool to get the data.\n"
        "Action: {tool_names}\n"
        'Action Input: {{"query": "<your query>"}}\n'
        "Observation: <the tool result will appear here>\n\n"
        "After you see the Observation, write:\n"
        "Thought: I now know the final answer\n"
        "Final Answer: <your summary of the data>\n"
    )
    _tool_prompt = "{{ .Prompt }}"

    # -- Agent per research step (each has exactly ONE tool) --
    def _make_agent(role, goal, backstory, tool_list):
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tool_list,
            llm=llm,
            verbose=True,
            max_iter=3,
            system_template=_tool_system,
            prompt_template=_tool_prompt,
        )

    diagnostics_agent = _make_agent(
        role="Appliance Diagnostics Specialist",
        goal="Look up troubleshooting guides and safety warnings for the appliance problem.",
        backstory="You are an appliance diagnostics expert. Use the ManualCheck tool to retrieve diagnostic data, then summarize the key findings.",
        tool_list=[manual_tool],
    )

    parts_agent = _make_agent(
        role="Parts & Pricing Researcher",
        goal="Find replacement parts, pricing, and availability for the repair.",
        backstory="You are a parts sourcing specialist. Use the PartsSearch tool to find parts and pricing, then summarize the options.",
        tool_list=[parts_tool],
    )

    tutorial_agent = _make_agent(
        role="Tutorial Researcher",
        goal="Find video tutorials and repair guides with difficulty ratings.",
        backstory="You research repair tutorials. Use the TutorialSearch tool to find guides, then summarize difficulty and quality.",
        tool_list=[tutorial_tool],
    )

    quote_agent = _make_agent(
        role="Professional Service Researcher",
        goal="Get professional repair quotes with labor costs and timelines.",
        backstory="You gather professional service quotes. Use the ProQuote tool to get pricing, then summarize the options.",
        tool_list=[quote_tool],
    )

    # NOTE: Synthesis is handled outside CrewAI (direct HFEngine call)
    # to avoid Nemotron getting confused by CrewAI's ReAct prompt scaffolding.

    # -- One task per tool + final synthesis --
    common_ctx = f"The homeowner's problem:\n{user_desc}"

    task_manual = Task(
        description=(
            f"{common_ctx}\n\n"
            "Use the ManualCheck tool to look up troubleshooting guides for this "
            "appliance problem. Pass a descriptive query about the appliance and "
            "symptoms. Then summarize: possible causes, DIY difficulty, safety "
            "warnings, and tools needed."
        ),
        expected_output="Summary of diagnostic findings, safety warnings, and DIY difficulty.",
        agent=diagnostics_agent,
    )

    task_parts = Task(
        description=(
            f"{common_ctx}\n\n"
            "Use the PartsSearch tool to find replacement parts and pricing. "
            "Pass a query about the appliance. Then summarize: available parts, "
            "prices, and the DIY cost range."
        ),
        expected_output="List of parts with prices and DIY cost range.",
        agent=parts_agent,
    )

    task_tutorials = Task(
        description=(
            f"{common_ctx}\n\n"
            "Use the TutorialSearch tool to find repair tutorials. "
            "Pass a query about this repair. Then summarize: best tutorials, "
            "difficulty ratings, and whether a beginner could follow them."
        ),
        expected_output="Tutorial recommendations with difficulty assessment.",
        agent=tutorial_agent,
    )

    task_quotes = Task(
        description=(
            f"{common_ctx}\n\n"
            "Use the ProQuote tool to get professional service quotes. "
            "Pass a query about this repair. Then summarize: repair options, "
            "costs, turnaround time, and urgency assessment."
        ),
        expected_output="Professional repair options with costs and timelines.",
        agent=quote_agent,
    )

    return Crew(
        agents=[diagnostics_agent, parts_agent, tutorial_agent, quote_agent],
        tasks=[task_manual, task_parts, task_tutorials, task_quotes],
        process=Process.sequential,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# F. SAE activation analysis (adapted from home_repair demo)
# ---------------------------------------------------------------------------


def analyze_activations(
    activation_log: list[tuple[str, dict[str, np.ndarray]]],
    sae_repo_id: str,
    sae_layer: int,
    layer_key: str = "residual_20",
    sae_local_dir: str | None = None,
) -> dict:
    """Encode captured activations through SAE and map to feature descriptions."""
    results: dict = {
        "steps": [],
        "sae_available": False,
        "features_available": False,
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

    # Tier 2: SAE feature decomposition
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae = None
    feature_descs = None

    if sae_local_dir:
        from kiji_inspector.core.sae_core import JumpReLUSAE

        layer_dir = Path(sae_local_dir) / f"layer_{sae_layer}"
        checkpoint = layer_dir / "sae_checkpoints" / "sae_final.pt"
        if checkpoint.exists():
            sae = JumpReLUSAE.from_pretrained(str(checkpoint), device=device)
            desc_path = layer_dir / "activations" / "feature_descriptions.json"
            if desc_path.exists():
                with open(desc_path) as f:
                    feature_descs = json.load(f)
    else:
        try:
            sae, feature_descs = SAE.from_pretrained(
                repo_id=sae_repo_id, layer=sae_layer, device=device
            )
        except Exception as e:
            print(f"  Could not load SAE: {e}")

    if sae is None:
        print("  SAE not available -- raw activation stats only.")
        return results

    # Load contrastive feature map if local
    contrastive_map: dict[int, list[dict]] = {}
    if sae_local_dir:
        report_path = Path(sae_local_dir) / f"layer_{sae_layer}" / "contrastive_features.json"
        # Also check under activations/ subdirectory
        if not report_path.exists():
            report_path = Path(sae_local_dir) / f"layer_{sae_layer}" / "activations" / "contrastive_features.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            for theme, info in report.items():
                if theme.startswith("_"):
                    continue
                for feat in info.get("top_features", []):
                    idx = feat["feature_index"]
                    anchor_act = feat.get("anchor_mean_activation", 0)
                    contrast_act = feat.get("contrast_mean_activation", 0)
                    direction = "anchor" if anchor_act > contrast_act else "contrast"
                    contrastive_map.setdefault(idx, []).append({
                        "theme": theme,
                        "rank": feat["rank"],
                        "cohens_d": feat["cohens_d"],
                        "direction": direction,
                    })

    results["sae_available"] = True
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype

    for step_info, (_step_label, acts) in zip(
        results["steps"], activation_log, strict=True
    ):
        if layer_key not in acts:
            continue
        vec = acts[layer_key]
        vec_tensor = torch.from_numpy(vec).unsqueeze(0).to(
            device=device, dtype=sae_dtype
        )
        with torch.no_grad():
            features = sae.encode(vec_tensor)
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
            if int(idx) in contrastive_map:
                feat_entry["themes"] = contrastive_map[int(idx)]
            top_features.append(feat_entry)

        # Theme score aggregation
        theme_scores: dict[str, list[float]] = {}
        for idx in nonzero_indices:
            act_val = float(features_np[idx])
            for entry in contrastive_map.get(int(idx), []):
                theme = entry["theme"]
                score = act_val * abs(entry["cohens_d"])
                theme_scores.setdefault(theme, []).append(score)

        step_info["sae_features"] = {
            "num_active": int(nonzero_mask.sum()),
            "total_features": int(features_np.shape[0]),
            "sparsity_pct": float((1.0 - nonzero_mask.mean()) * 100),
            "top_features": top_features,
            "theme_activations": {
                theme: {
                    "total_score": round(sum(scores), 4),
                    "num_features": len(scores),
                    "mean_score": round(sum(scores) / len(scores), 4),
                }
                for theme, scores in sorted(
                    theme_scores.items(), key=lambda x: -sum(x[1])
                )
            },
        }

    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Tier 3: Feature labels
    if feature_descs:
        results["features_available"] = True
        for step_info in results["steps"]:
            if "sae_features" not in step_info:
                continue
            for feat in step_info["sae_features"]["top_features"]:
                desc = feature_descs.get(str(feat["index"]))
                if desc:
                    feat["label"] = desc.get("label", "unknown")
                    feat["description"] = desc.get("description", "")
                    feat["confidence"] = desc.get("confidence", "low")

    return results


# ---------------------------------------------------------------------------
# G. Build UI-friendly results from analysis
# ---------------------------------------------------------------------------


def build_ui_results(
    analysis: dict,
    crew_output: str,
    problem_info: dict,
) -> dict:
    """Transform SAE analysis + crew output into a visualization-ready structure."""

    steps_ui = []

    # Build per-feature activation across all steps for deviation scoring
    feature_activations: dict[int, list[float]] = {}
    for step_info in analysis.get("steps", []):
        if "sae_features" not in step_info:
            continue
        for f in step_info["sae_features"].get("top_features", []):
            feature_activations.setdefault(f.get("index", -1), []).append(
                f.get("activation", 0)
            )
    feature_mean: dict[int, float] = {
        idx: sum(vals) / len(vals) for idx, vals in feature_activations.items()
    }

    for step_info in analysis.get("steps", []):
        step_label = step_info["step"]

        # Determine display name from step label
        tool_display = {
            "Manual Check": {"name": "Repair Manual Lookup", "icon": "book"},
            "Parts Search": {"name": "Parts & Pricing Search", "icon": "wrench"},
            "Tutorial Search": {"name": "Video Tutorial Search", "icon": "video"},
            "Pro Quote": {"name": "Professional Quote", "icon": "clipboard"},
            "Synthesis": {"name": "Final Recommendation", "icon": "clipboard"},
        }
        display_name = step_label
        display_icon = "cog"
        for tool_name, meta in tool_display.items():
            if tool_name.lower() in step_label.lower():
                display_name = meta["name"]
                display_icon = meta["icon"]
                break

        # SAE features with deviation scoring
        features_list = []
        sentence = ""
        if "sae_features" in step_info:
            top_feats = step_info["sae_features"].get("top_features", [])
            scored = []
            for f in top_feats:
                idx = f.get("index", -1)
                act = f.get("activation", 0)
                mean = feature_mean.get(idx, act)
                deviation = act - mean
                scored.append((deviation, f))
            scored.sort(key=lambda x: x[0], reverse=True)

            distinctive = scored[:5]
            max_dev = max((abs(d) for d, _ in distinctive), default=1.0) or 1.0
            for dev, f in distinctive:
                features_list.append(
                    {
                        "label": f.get("label", f"Feature #{f.get('index', '?')}"),
                        "strength": round(abs(dev) / max_dev, 2),
                        "description": f.get("description", ""),
                    }
                )

            # Generate explanation tied to tool context and detected themes
            if features_list:
                top2_labels = [f["label"] for f in features_list[:2]]
                focus = " and ".join(top2_labels)

                # Get top active themes for this step
                theme_acts = step_info.get("sae_features", {}).get(
                    "theme_activations", {}
                )
                top_themes = sorted(
                    theme_acts.items(),
                    key=lambda x: x[1].get("total_score", 0),
                    reverse=True,
                )[:2]
                theme_names = [
                    t[0].replace("_", " ").replace(" vs ", " vs. ")
                    for t in top_themes
                ]

                # Tool-specific framing
                tool_context = {
                    "Manual Check": (
                        "When consulting the repair manual, the model's "
                        "representations were dominated by {focus}."
                    ),
                    "Parts Search": (
                        "While evaluating replacement parts and pricing, "
                        "the model activated most strongly on {focus}."
                    ),
                    "Tutorial Search": (
                        "When assessing available repair tutorials, the "
                        "model's internal state highlighted {focus}."
                    ),
                    "Pro Quote": (
                        "While reviewing professional service options, "
                        "the model weighted {focus} most heavily."
                    ),
                    "Synthesis": (
                        "In forming the final recommendation, the model "
                        "drew most on {focus}."
                    ),
                }

                # Find which tool this step belongs to
                tool_key = "step"
                for tn in tool_context:
                    if tn.lower() in step_label.lower():
                        tool_key = tn
                        break

                base = tool_context.get(tool_key, (
                    "At this step, the model focused most on {focus}."
                )).format(focus=focus)

                # Tie to themes
                if theme_names:
                    theme_str = " and ".join(theme_names)
                    sentence = (
                        f"{base} This aligns with the {theme_str} "
                        f"dimensions of the problem."
                    )
                else:
                    sentence = base

        # Raw stats
        raw = {}
        for layer, stats in step_info.get("raw_stats", {}).items():
            if "residual_20" in layer:
                raw = stats

        steps_ui.append(
            {
                "label": step_label,
                "displayName": display_name,
                "icon": display_icon,
                "features": features_list,
                "sentence": sentence,
                "stats": {
                    "active_features": step_info.get("sae_features", {}).get(
                        "num_active", 0
                    ),
                    "total_features": step_info.get("sae_features", {}).get(
                        "total_features", 0
                    ),
                    "sparsity": step_info.get("sae_features", {}).get(
                        "sparsity_pct", 0
                    ),
                    "l2_norm": raw.get("l2_norm", 0),
                },
                "theme_activations": step_info.get("sae_features", {}).get(
                    "theme_activations", {}
                ),
            }
        )

    return {
        "problem": problem_info,
        "recommendation": crew_output,
        "sae_available": analysis.get("sae_available", False),
        "features_available": analysis.get("features_available", False),
        "steps": steps_ui,
        "num_steps": len(steps_ui),
    }


# ---------------------------------------------------------------------------
# H. FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Kiji Inspector - Home Repair Demo")

# Job storage: job_id -> {"status": ..., "result": ..., "queue": ...}
_jobs: dict[str, dict] = {}
_model_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.post("/api/analyze")
async def start_analysis(request: Request):
    if not CREWAI_AVAILABLE:
        return JSONResponse(
            {"error": f"CrewAI not installed: {_IMPORT_ERR}"},
            status_code=500,
        )
    if _engine is None:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    body = await request.json()
    job_id = uuid.uuid4().hex[:12]
    q: queue.Queue = queue.Queue()
    _jobs[job_id] = {"status": "running", "result": None, "queue": q}

    def run_job():
        global _progress_queue
        _progress_queue = q
        try:
            q.put({"type": "status", "message": "Starting CrewAI agent..."})

            with _model_lock:
                _engine.clear_log()
                _reset_step_counter()

                # Run CrewAI crew (4 tool-research tasks)
                q.put({"type": "status", "message": "Agent is analyzing your problem..."})
                crew = create_crew(body)
                crew_result = crew.kickoff()

                # Build research context from raw tool data (not CrewAI
                # output, which is polluted with ReAct scaffolding).
                pid = _match_problem(body.get("details", ""))
                research_sections = []
                for tool_name, data_store in _TOOL_SOURCES.items():
                    data = data_store.get(pid, list(data_store.values())[0])
                    research_sections.append(
                        f"{tool_name}:\n{json.dumps(data, indent=2)}"
                    )
                research_context = "\n\n".join(research_sections)
                # Truncate if too long for the model
                if len(research_context) > 4000:
                    research_context = research_context[-4000:]

                # Synthesis: call HFEngine directly with a clean prompt.
                # We pre-fill the assistant response with "The agent recommends "
                # so the model continues with the actual recommendation instead
                # of starting a <think> block or meta-reasoning.
                q.put({"type": "step", "label": "Synthesis", "status": "generating"})
                synth_prompt = _engine._build_prompt(
                    "You are a home repair advisor.",
                    f"A {body.get('age', 'Unknown')}-old "
                    f"{body.get('appliance', 'appliance')} has this problem: "
                    f"{body.get('details', 'unknown issue')}.\n\n"
                    f"Key findings from research:\n{research_context}\n\n"
                    "Based on the research data above, write a 2-3 sentence "
                    "recommendation.",
                ) + "The agent recommends "
                print("\n" + "=" * 60)
                print("  [DEBUG] Synthesis prompt:")
                print("=" * 60)
                print(synth_prompt)
                print("=" * 60)

                raw_output = _engine.generate(synth_prompt, "Synthesis", max_tokens=256)

                print("\n" + "-" * 60)
                print("  [DEBUG] Synthesis raw output:")
                print("-" * 60)
                print(repr(raw_output))
                print("-" * 60)

                cleaned = _clean_model_output(raw_output)
                # Take the first paragraph as the recommendation
                crew_output = "The agent recommends " + cleaned.split("\n\n")[0]

                print("  [DEBUG] Final recommendation:")
                print(crew_output)
                print("=" * 60 + "\n")

                q.put({"type": "step", "label": "Synthesis", "status": "complete"})

                q.put({"type": "status", "message": "Extracting activations..."})
                layers = [20]
                activation_log = _engine.extract_all_prompts(layers)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            q.put({"type": "status", "message": "Running SAE analysis..."})
            analysis = analyze_activations(
                activation_log=activation_log,
                sae_repo_id=_SAE_REPO_ID,
                sae_layer=_SAE_LAYER,
                sae_local_dir=_cli_args.sae_local_dir if _cli_args else None,
            )

            q.put({"type": "status", "message": "Building visualization..."})
            ui_results = build_ui_results(analysis, crew_output, body)

            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["result"] = ui_results
            q.put({"type": "complete", "data": ui_results})

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"Job {job_id} failed: {tb}")
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {"error": str(e)}
            q.put({"type": "error", "message": str(e)})
        finally:
            _progress_queue = None

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    q = _jobs[job_id]["queue"]

    async def event_generator():
        while True:
            try:
                msg = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.3)
                # Check if job is done (might have missed the event)
                if _jobs[job_id]["status"] in ("complete", "error"):
                    if _jobs[job_id]["result"]:
                        if _jobs[job_id]["status"] == "error":
                            yield f"data: {json.dumps({'type': 'error', 'message': str(_jobs[job_id]['result'])})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'complete', 'data': _jobs[job_id]['result']})}\n\n"
                    break
                continue

            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") in ("complete", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {
        "model_loaded": _engine is not None,
        "crewai_available": CREWAI_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
    }


# ---------------------------------------------------------------------------
# I. Fallback: scripted mode (no CrewAI, same as original demo)
# ---------------------------------------------------------------------------


def run_scripted_analysis(
    engine: HFEngine, problem_info: dict, progress_queue: queue.Queue
) -> tuple[str, dict]:
    """Run scripted analysis without CrewAI (fallback mode)."""
    problem_id = _match_problem(
        f"{problem_info.get('appliance', '')} {problem_info.get('details', '')}"
    )

    tools = {
        "ManualCheck": ("appliance troubleshooting guide", _MANUAL_DATA),
        "PartsSearch": ("replacement parts and pricing", _PARTS_DATA),
        "TutorialSearch": ("repair video tutorials", _TUTORIAL_DATA),
        "ProQuote": ("professional repair quotes", _PRO_QUOTE_DATA),
    }

    all_context = ""
    for tool_name, (tool_desc, source) in tools.items():
        tool_result = json.dumps(source.get(problem_id, {}), indent=2)
        step_label = f"{problem_id}_{tool_name}"

        progress_queue.put(
            {"type": "step", "label": step_label, "status": "generating"}
        )

        user_msg = (
            f"A homeowner needs help with:\n"
            f"Appliance: {problem_info.get('appliance', 'Unknown')}\n"
            f"Age: {problem_info.get('age', 'Unknown')}\n"
            f"Problem: {problem_info.get('details', '')}\n\n"
        )
        if all_context:
            user_msg += f"Your analysis so far:\n{all_context}\n\n"
        user_msg += (
            f"Here is the {tool_desc} data:\n"
            f"```json\n{tool_result}\n```\n\n"
            f"Analyze this {tool_desc} data. Highlight key takeaways, "
            f"safety concerns, and whether this points toward DIY or "
            f"professional repair."
        )

        prompt = engine._build_prompt(_SYSTEM_PROMPT, user_msg)
        analysis = engine.generate(prompt, step_label, max_tokens=300)
        all_context += f"\n[{tool_name}] {analysis.strip()}\n"

        progress_queue.put(
            {"type": "step", "label": step_label, "status": "complete"}
        )

    # Final recommendation — use raw tool data (not model analysis output,
    # which may contain scaffolding the model echoes at synthesis time).
    progress_queue.put(
        {"type": "step", "label": "final_recommendation", "status": "generating"}
    )
    research_sections = []
    for tool_name, (_, source) in tools.items():
        data = source.get(problem_id, list(source.values())[0])
        research_sections.append(
            f"{tool_name}:\n{json.dumps(data, indent=2)}"
        )
    raw_research = "\n\n".join(research_sections)
    truncated = raw_research[-4000:] if len(raw_research) > 4000 else raw_research
    final_msg = (
        f"A {problem_info.get('age', 'Unknown')}-old "
        f"{problem_info.get('appliance', 'appliance')} has this problem: "
        f"{problem_info.get('details', 'unknown issue')}.\n\n"
        f"Key findings from research:\n{truncated}\n\n"
        "The agent recommends "
    )
    final_prompt = engine._build_prompt(
        "You are a home repair advisor.",
        final_msg.replace(
            "The agent recommends ",
            "Based on the research data above, write a 2-3 sentence "
            "recommendation.",
        ),
    ) + "The agent recommends "
    print("\n" + "=" * 60)
    print("  [DEBUG] Scripted synthesis prompt:")
    print("=" * 60)
    print(final_prompt)
    print("=" * 60)

    raw_rec = engine.generate(final_prompt, "final_recommendation", max_tokens=256)

    print("\n" + "-" * 60)
    print("  [DEBUG] Scripted synthesis raw output:")
    print("-" * 60)
    print(repr(raw_rec))
    print("-" * 60)

    cleaned = _clean_model_output(raw_rec)
    final_rec = "The agent recommends " + cleaned.split("\n\n")[0]

    print("  [DEBUG] Final recommendation:")
    print(final_rec)
    print("=" * 60 + "\n")
    progress_queue.put(
        {"type": "step", "label": "final_recommendation", "status": "complete"}
    )

    return final_rec, {"context": all_context}


@app.post("/api/analyze-scripted")
async def start_scripted_analysis(request: Request):
    """Fallback endpoint that runs without CrewAI."""
    if _engine is None:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    body = await request.json()
    job_id = uuid.uuid4().hex[:12]
    q: queue.Queue = queue.Queue()
    _jobs[job_id] = {"status": "running", "result": None, "queue": q}

    def run_job():
        try:
            q.put({"type": "status", "message": "Starting analysis..."})

            with _model_lock:
                _engine.clear_log()

                q.put({"type": "status", "message": "Agent is analyzing your problem..."})
                final_rec, _extra = run_scripted_analysis(_engine, body, q)

                q.put({"type": "status", "message": "Extracting activations..."})
                layers = [20]
                activation_log = _engine.extract_all_prompts(layers)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            q.put({"type": "status", "message": "Running SAE analysis..."})
            analysis = analyze_activations(
                activation_log=activation_log,
                sae_repo_id=_SAE_REPO_ID,
                sae_layer=_SAE_LAYER,
                sae_local_dir=_cli_args.sae_local_dir if _cli_args else None,
            )

            q.put({"type": "status", "message": "Building visualization..."})
            ui_results = build_ui_results(analysis, final_rec, body)

            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["result"] = ui_results
            q.put({"type": "complete", "data": ui_results})

        except Exception as e:
            import traceback

            print(f"Job {job_id} failed: {traceback.format_exc()}")
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {"error": str(e)}
            q.put({"type": "error", "message": str(e)})

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id})


# ---------------------------------------------------------------------------
# J. Main entry point
# ---------------------------------------------------------------------------

_cli_args = None


def main():
    global _engine, _cli_args

    parser = argparse.ArgumentParser(
        description="Web Demo: Home Repair AI Advisor with SAE Explanations"
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--sae-local-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "output_merged"),
        help="Local SAE directory (default: output_merged/)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Start server without loading model (for UI development)",
    )
    _cli_args = parser.parse_args()

    print("=" * 60)
    print("  Kiji Inspector - Web Demo")
    print("  Home Repair AI Advisor + SAE Explanations")
    print("=" * 60)

    if not _cli_args.no_model:
        print("\n[1/3] Loading Nemotron model...")
        _engine = HFEngine(
            device=_cli_args.device,
            dtype=_cli_args.dtype,
            max_new_tokens=_cli_args.max_new_tokens,
        )

        if CREWAI_AVAILABLE:
            print("\n[2/3] Setting up CrewAI + LiteLLM...")
            _setup_litellm()
        else:
            print(f"\n[2/3] CrewAI not available ({_IMPORT_ERR}). Using scripted mode.")

        print("\n[3/3] Starting web server...")
    else:
        print("\n  --no-model: Skipping model load (UI dev mode)")
        print("  Starting web server...")

    print(f"\n  Open http://localhost:{_cli_args.port} in your browser")
    print("=" * 60)

    uvicorn.run(app, host=_cli_args.host, port=_cli_args.port, log_level="info")


if __name__ == "__main__":
    main()
