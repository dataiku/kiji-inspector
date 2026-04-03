#!/usr/bin/env python3
"""
Home Repair Agent Demo with SAE Activation Analysis.

Orchestrates a home repair advisor using NVIDIA Nemotron-3-Nano-30B:
  1. For each repair problem (dishwasher, disposal, water heater), the model
     is presented with tool results and asked to analyze them -- one LLM call
     per (problem, tool) pair
  2. After all data is gathered, the model produces a final recommendation
  3. Activations are extracted at each decision point via HF forward hooks
  4. Activations are mapped through a trained SAE to explain the reasoning

Uses HuggingFace transformers for both generation and activation capture
(no vLLM required). Single model instance serves both purposes.

The SAE is loaded from HuggingFace Hub (davidnet/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).

Prerequisites:
    pip install 'kiji-inspector[huggingface]'
    huggingface-cli login

Usage:
    uv run python demo/home_repair/home_repair_demo.py
    uv run python demo/home_repair/home_repair_demo.py --device cuda
    uv run python demo/home_repair/home_repair_demo.py --youtube-api-key YOUR_KEY
    uv run python demo/home_repair/home_repair_demo.py --sae-layer 20
"""

from __future__ import annotations

import argparse
import gc
import json
import textwrap
from pathlib import Path

import numpy as np
import torch

from kiji_inspector.core.sae import SAE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_SAE_REPO_ID = "davidnet/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_SAE_LAYER = 20

_SYSTEM_PROMPT = (
    "You are an experienced home repair advisor. You help homeowners diagnose "
    "appliance problems and decide whether to attempt a DIY repair or hire a "
    "professional. Always consider safety first, then cost, difficulty, and "
    "warranty implications. Reference specific details from the data provided."
)

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
    },
]


# ---------------------------------------------------------------------------
# Section A: HuggingFace generation + extraction engine
# ---------------------------------------------------------------------------


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

        # Resolve hidden_size (some models nest under text_config)
        self.hidden_size = (
            getattr(self.model.config, "hidden_size", None)
            or getattr(self.model.config, "text_config", self.model.config).hidden_size
        )

        print(f"  Model ready on {self._input_device} ({torch_dtype})")
        print(f"  hidden_size: {self.hidden_size}")

    # --- Device helpers ---

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
        raise AttributeError(f"Cannot locate transformer layers for {type(self.model).__name__}")

    def _get_inner_model(self):
        """Get transformer body, skipping lm_head to avoid logit allocation."""
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model"):
                return lm.model
            return lm
        if hasattr(self.model, "model"):
            return self.model.model
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
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Some models may not support system role -- prepend to user message
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # --- Generation ---

    def generate(self, prompt: str, step_label: str, max_tokens: int | None = None) -> str:
        """Generate text and log the prompt for later activation extraction."""
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

    # --- Activation extraction ---

    def extract_all_prompts(self, layers: list[int]) -> list[tuple[str, dict[str, np.ndarray]]]:
        """Extract last-token activations for every logged prompt.

        Registers forward hooks on the target layers, runs a single forward
        pass per prompt through the transformer body, then removes all hooks.
        """
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
                h = model_layers[idx].register_forward_hook(_make_hook(f"residual_{idx}"))
                hooks.append(h)

        inner = self._get_inner_model()
        results: list[tuple[str, dict[str, np.ndarray]]] = []

        for step_label, prompt in self.prompt_log:
            activations.clear()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
            with torch.no_grad():
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
    Then asks for a final recommendation across all three problems.

    Returns:
        (final_recommendation, per_problem_analyses)
    """
    per_problem_analyses: dict[str, str] = {}
    all_context = ""

    for problem in _PROBLEMS:
        pid = problem["id"]
        print(f"\n  --- Analyzing: {problem['summary']} ---")
        problem_context = ""

        for tool_name, (_, tool_desc) in _TOOLS.items():
            tool_result = _get_tool_result(tool_name, pid, youtube_api_key)
            step_label = f"{pid}_{tool_name}"

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
        all_context += f"\n=== {problem['summary']} ===\n{problem_context}\n"

    # Final recommendation
    print("\n  --- Final Recommendation ---")
    final_user_msg = (
        "You have analyzed three home repair problems. "
        "Here is your full analysis:\n\n"
        f"{all_context}\n\n"
        "Now provide your final recommendation for each problem:\n"
        "1. DIY or hire a professional? Why?\n"
        "2. Estimated cost (DIY vs professional)\n"
        "3. Safety considerations\n"
        "4. Urgency level (fix now / schedule soon / can wait)\n"
        "5. Priority order: which problem should be addressed first?"
    )

    final_prompt = engine._build_prompt(_SYSTEM_PROMPT, final_user_msg)
    final_rec = engine.generate(final_prompt, "final_recommendation", max_tokens=800)

    return final_rec, per_problem_analyses


# ---------------------------------------------------------------------------
# Section D: Post-run activation extraction + SAE analysis
# ---------------------------------------------------------------------------


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
    layer_key: str = "residual_20",
) -> dict:
    """Encode captured activations through SAE and map to feature descriptions.

    Three tiers of analysis:
      1. Raw activation statistics (always available)
      2. SAE feature decomposition (if checkpoint found)
      3. Feature label mapping (if descriptions found)
    """
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

    # Tier 2: SAE feature decomposition (from HuggingFace Hub)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, feature_descs = _load_sae_from_hub(sae_repo_id, sae_layer, device=device)
    if sae is None:
        print("  SAE not available -- showing raw activation stats only.")
        return results

    results["sae_available"] = True
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype

    for step_info, (_step_label, acts) in zip(results["steps"], activation_log, strict=True):
        if layer_key not in acts:
            continue
        vec = acts[layer_key]
        vec_tensor = torch.from_numpy(vec).unsqueeze(0).to(device=device, dtype=sae_dtype)
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

        step_info["sae_features"] = {
            "num_active": int(nonzero_mask.sum()),
            "total_features": int(features_np.shape[0]),
            "sparsity_pct": float((1.0 - nonzero_mask.mean()) * 100),
            "top_features": [
                {"index": int(idx), "activation": float(val)}
                for idx, val in zip(top_indices, top_values, strict=True)
            ],
        }

    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Tier 3: Feature label mapping (already loaded from Hub)
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
        "costRange": "$13\u2013$90 DIY",
    },
    "disposal_stuck": {
        "icon": "\u2699\ufe0f",
        "urgency": {"label": "Can Wait", "level": "green"},
        "difficulty": {"label": "Easy", "level": "green"},
        "costRange": "$0\u2013$8 DIY",
    },
    "water_heater_noise": {
        "icon": "\U0001f525",
        "urgency": {"label": "Act Now", "level": "red"},
        "difficulty": {"label": "Difficult", "level": "red"},
        "costRange": "$15\u2013$649 DIY",
    },
}

# Feature-to-theme mapping for comparison chart derivation.
# Each theme has keywords — if a feature label contains any, it counts.
_THEME_KEYWORDS = {
    "Safety Concern": ["safety", "hazard", "gas", "risk", "danger"],
    "Cost Sensitivity": ["cost", "budget", "price", "replacement cost", "expense"],
    "DIY Feasibility": ["diy", "beginner", "skill", "tool requirement"],
    "Urgency Level": ["urgent", "immediate", "damage", "active"],
    "Age / Warranty Factor": ["age", "lifespan", "warranty", "coverage"],
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
        lines.append(f"<strong>\"{r['title']}\"</strong> by {channel}{meta_str}")
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
    return f"While {verb}, the AI focused most on {focus}."


def _derive_comparison_scores(
    sae_features: dict[str, dict[str, dict]],
) -> dict[str, dict[str, int]]:
    """Derive comparison scores from SAE features by theme-keyword matching."""
    comparison: dict[str, dict[str, float]] = {
        theme: {} for theme in _THEME_KEYWORDS
    }
    for pid in sae_features:
        # Collect all features for this problem across tools
        all_features: list[dict] = []
        for tool_data in sae_features[pid].values():
            all_features.extend(tool_data.get("features", []))

        for theme, keywords in _THEME_KEYWORDS.items():
            score = 0.0
            count = 0
            for f in all_features:
                label_lower = f.get("label", "").lower()
                if any(kw in label_lower for kw in keywords):
                    score += f.get("strength", 0)
                    count += 1
            comparison[theme][pid] = round(score / max(count, 1) * 100) if count else 10

    return comparison


def build_ui_data(
    analysis: dict,
    per_problem: dict[str, str],
    final_recommendation: str,
    youtube_api_key: str | None = None,
) -> dict:
    """Transform demo outputs into the DATA shape expected by index.html."""

    # --- problems ---
    problems = []
    for p in _PROBLEMS:
        meta = _PROBLEM_META.get(p["id"], {})
        problems.append({
            "id": p["id"],
            "icon": meta.get("icon", ""),
            "title": p["summary"],
            "appliance": p["appliance"],
            "age": p["age"],
            "details": p["details"],
            "urgency": meta.get("urgency", {"label": "Unknown", "level": "yellow"}),
            "difficulty": meta.get("difficulty", {"label": "Unknown", "level": "yellow"}),
            "costRange": meta.get("costRange", ""),
        })

    # --- toolResults: HTML summaries from mock data ---
    tool_results: dict[str, dict[str, str]] = {}
    for p in _PROBLEMS:
        pid = p["id"]
        tool_results[pid] = {}
        for tool_name, (source, _) in _TOOLS.items():
            data = source.get(pid, {})
            summarizer = _TOOL_SUMMARIZERS.get(tool_name)
            tool_results[pid][tool_name] = summarizer(data) if summarizer else json.dumps(data)

    # --- saeFeatures: from analysis_results steps ---
    sae_features: dict[str, dict[str, dict]] = {}
    step_lookup: dict[str, dict] = {}
    for step_info in analysis.get("steps", []):
        step_lookup[step_info["step"]] = step_info

    for p in _PROBLEMS:
        pid = p["id"]
        sae_features[pid] = {}
        for tool_name in _TOOLS:
            step_key = f"{pid}_{tool_name}"
            step_info = step_lookup.get(step_key)

            features_list: list[dict] = []
            if step_info and "sae_features" in step_info:
                top_feats = step_info["sae_features"].get("top_features", [])
                # Normalize strengths: max feature in this step = 1.0
                max_act = max((f.get("activation", 0) for f in top_feats[:5]), default=1.0) or 1.0
                for f in top_feats[:5]:
                    act = f.get("activation", 0)
                    features_list.append({
                        "label": f.get("label", f"Feature #{f.get('index', '?')}"),
                        "strength": round(act / max_act, 2),
                        "description": f.get("description", ""),
                    })

            sentence = _generate_feature_sentence(tool_name, features_list)
            sae_features[pid][tool_name] = {
                "features": features_list,
                "sentence": sentence,
            }

    # --- recommendations: from per-problem analyses + final recommendation ---
    # Parse the LLM's final recommendation into per-problem sections.
    # Fallback: use per-problem analysis text directly.
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
            pro_cost = f"${lo}\u2013${hi}" if lo != hi else f"${lo}"
        else:
            pro_cost = ""

        # Rationale from per-problem analysis
        raw = per_problem.get(pid, "")
        # Use the last tool analysis (ProQuote) as it's the most synthesized
        sections = raw.split("[ProQuote]")
        rationale = sections[-1].strip() if len(sections) > 1 else raw.strip()
        # Truncate to reasonable length
        if len(rationale) > 500:
            rationale = rationale[:497] + "..."

        recommendations[pid] = {
            "verdict": verdict,
            "verdictLabel": verdict_label,
            "diyCost": diy_cost,
            "proCost": pro_cost,
            "rationale": rationale,
        }

    # --- comparison: derive from SAE feature strengths ---
    comparison = _derive_comparison_scores(sae_features)

    # --- themes: static definitions from home_repair.json contrast types ---
    themes_raw = {
        "diy_vs_professional": {
            "title": "DIY vs. Professional",
            "leftLabel": "Easy DIY",
            "rightLabel": "Needs a Pro",
        },
        "urgent_vs_planned": {
            "title": "Urgent vs. Planned",
            "leftLabel": "Can Wait",
            "rightLabel": "Act Now",
        },
        "cheap_fix_vs_replacement": {
            "title": "Cheap Fix vs. Replacement",
            "leftLabel": "Quick Part Swap",
            "rightLabel": "Consider Replacing",
        },
        "safe_vs_hazardous": {
            "title": "Safe vs. Hazardous",
            "leftLabel": "Low Risk",
            "rightLabel": "High Hazard",
        },
        "warranty_vs_out_of_pocket": {
            "title": "Warranty vs. Out of Pocket",
            "leftLabel": "May Be Covered",
            "rightLabel": "Out of Pocket",
        },
    }

    # Load descriptions from home_repair.json if available
    json_path = Path(__file__).parent / "home_repair.json"
    contrast_descriptions = {}
    if json_path.exists():
        with open(json_path) as f:
            hr_config = json.load(f)
        contrast_descriptions = hr_config.get("contrast_types", {})

    # Map comparison themes to theme IDs for marker derivation
    theme_to_comparison = {
        "diy_vs_professional": "DIY Feasibility",
        "urgent_vs_planned": "Urgency Level",
        "cheap_fix_vs_replacement": "Cost Sensitivity",
        "safe_vs_hazardous": "Safety Concern",
        "warranty_vs_out_of_pocket": "Age / Warranty Factor",
    }

    themes = []
    for theme_id, tmeta in themes_raw.items():
        desc = contrast_descriptions.get(theme_id, "")
        comp_key = theme_to_comparison.get(theme_id, "")
        comp_scores = comparison.get(comp_key, {})
        markers = {}
        for pid in ["dishwasher_leak", "disposal_stuck", "water_heater_noise"]:
            markers[pid] = comp_scores.get(pid, 50)

        # Identify top 2 features driving this theme
        comp_theme_kws = _THEME_KEYWORDS.get(comp_key, [])
        driving_features: list[str] = []
        for pid_data in sae_features.values():
            for tool_data in pid_data.values():
                for f in tool_data.get("features", []):
                    lbl = f.get("label", "").lower()
                    if any(kw in lbl for kw in comp_theme_kws) and f["label"] not in driving_features:
                        driving_features.append(f["label"])
        driving_features = driving_features[:2]
        if driving_features:
            features_html = "Driven by " + " and ".join(
                f"<strong>{lbl}</strong>" for lbl in driving_features
            )
        else:
            features_html = ""

        themes.append({
            "id": theme_id,
            "title": tmeta["title"],
            "description": desc,
            "leftLabel": tmeta["leftLabel"],
            "rightLabel": tmeta["rightLabel"],
            "markers": markers,
            "features": features_html,
        })

    return {
        "problems": problems,
        "toolResults": tool_results,
        "saeFeatures": sae_features,
        "recommendations": recommendations,
        "comparison": comparison,
        "themes": themes,
    }


# ---------------------------------------------------------------------------
# Section F: Explanation generation
# ---------------------------------------------------------------------------


def _build_feature_summary(analysis_results: dict) -> str:
    lines = []
    for step_info in analysis_results["steps"]:
        lines.append(f"## {step_info['step']}")

        for layer, stats in step_info.get("raw_stats", {}).items():
            if "residual_20" in layer:
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

    for step_info in analysis["steps"]:
        print(f"\n  --- {step_info['step']} ---")
        for layer, stats in step_info.get("raw_stats", {}).items():
            if "residual_20" in layer:
                print(f"    L2 norm: {stats['l2_norm']:.2f}, std: {stats['std']:.4f}")
        sae = step_info.get("sae_features")
        if sae:
            print(
                f"    Active features: {sae['num_active']}/{sae['total_features']} "
                f"({100 - sae['sparsity_pct']:.1f}%)"
            )
            for feat in sae["top_features"][:3]:
                label = feat.get("label", f"Feature #{feat['index']}")
                print(f"      {label}: {feat['activation']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Home Repair Agent Demo with SAE Activation Analysis"
    )
    parser.add_argument(
        "--sae-repo-id",
        type=str,
        default=_SAE_REPO_ID,
        help=f"HuggingFace repo ID for the SAE (default: {_SAE_REPO_ID})",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=_SAE_LAYER,
        help=f"SAE layer to load from the HF repo (default: {_SAE_LAYER})",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
        help="Transformer layers to hook for activation extraction",
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
        "--youtube-api-key",
        type=str,
        default=None,
        help="YouTube Data API v3 key for real tutorial search (optional)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Home Repair Agent Demo")
    print("  Nemotron-3-Nano-30B (HuggingFace) + SAE Activation Analysis")
    print(f"  SAE: {args.sae_repo_id} (layer {args.sae_layer})")
    print("=" * 60)

    # Phase 1: Run scripted multi-step analysis
    print("\n[Phase 1] Running home repair analysis...")
    engine = HFEngine(
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    final_recommendation, per_problem = run_home_repair_analysis(engine, args.youtube_api_key)

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete. {len(engine.prompt_log)} decision points recorded.")
    print(f"{'=' * 60}")
    print(f"\n  FINAL RECOMMENDATION:\n{final_recommendation}")

    # Phase 2: Extract activations (reuses same model, forward pass with hooks)
    print("\n[Phase 2] Extracting activations...")
    activation_log = engine.extract_all_prompts(args.layers)

    # Phase 3: Analyze activations through SAE
    print("\n[Phase 3] Analyzing activations through SAE...")
    analysis = analyze_activations(
        activation_log=activation_log,
        sae_repo_id=args.sae_repo_id,
        sae_layer=args.sae_layer,
    )
    print_analysis_summary(analysis)

    # Phase 4: Generate explanations (reuses same model)
    if analysis["sae_available"]:
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
        technical, layman = "", ""
        print("\n[Phase 4] Skipped explanation generation (no SAE available).")

    # Cleanup model
    engine.cleanup()

    # Phase 5: Save results
    output_path = Path("demo/home_repair/output")
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
    )
    with open(output_path / "ui_data.json", "w") as f:
        json.dump(ui_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {output_path}/")
    print(f"  Open demo/home_repair/index.html to view the interactive explanation.")
    print("\n  Done.")


if __name__ == "__main__":
    main()
