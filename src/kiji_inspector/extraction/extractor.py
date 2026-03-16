"""
Activation extraction: build agent prompts and extract raw activations
at the decision token for SAE training.

The standard approach is to train the SAE on raw activations and use
contrastive pairs *post-hoc* to identify which SAE features correspond
to tool-selection decisions.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from kiji_inspector.data.contrastive_dataset import ContrastivePair


def build_agent_prompt_from_tokenizer(
    tokenizer,
    system_prompt: str,
    tools: list[dict],
    user_request: str,
    assistant_prefill: str = "I'll use the ",
) -> str:
    """Build a prompt using the tokenizer's native chat template.

    Works with any HuggingFace model that ships a chat template
    (Qwen, Llama, Mistral, Gemma, Nemotron, etc.).

    Args:
        tokenizer: A HuggingFace tokenizer with a chat template.
        system_prompt: The system message.
        tools: Tool definitions for the prompt.
        user_request: The user's message.
        assistant_prefill: Text to prepend to the assistant turn
            (creates a "prefill" so we extract at the decision point).

    Returns:
        The formatted prompt string ending at the decision token.
    """
    tool_descriptions = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)

    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                f"When you decide to use a tool, respond with the tool name."
            ),
        },
        {"role": "user", "content": user_request},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    formatted += assistant_prefill
    return formatted


def build_agent_prompt(
    system_prompt: str,
    tools: list[dict],
    user_request: str,
    model_type: str = "auto",
    tokenizer=None,
    assistant_prefill: str = "I'll use the ",
) -> str:
    """
    Build a full prompt in the target model's chat template.

    When a *tokenizer* is provided and has a chat template, the prompt is
    built via ``tokenizer.apply_chat_template`` (preferred — works with any
    HuggingFace model).  Otherwise falls back to manual format strings
    selected by *model_type*.

    Supported model_type values (legacy fallback):
        - "nemotron"  : ChatML-style used by NVIDIA Nemotron models
        - "llama"     : Llama 3 Instruct format
        - "mistral"   : Mistral instruct format
        - "generic"   : Plain text fallback
        - "auto"      : Use tokenizer if available, else "generic"

    Returns:
        The formatted prompt string ending just before the model's
        tool-selection decision token.
    """
    # Prefer tokenizer-based approach
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return build_agent_prompt_from_tokenizer(
            tokenizer, system_prompt, tools, user_request, assistant_prefill
        )

    # Legacy manual format strings
    if model_type == "auto":
        model_type = "generic"
        warnings.warn(
            "No tokenizer provided and model_type='auto'. "
            "Falling back to 'generic' prompt format. "
            "Pass a tokenizer for best results.",
            stacklevel=2,
        )

    tool_descriptions = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)

    if model_type == "nemotron":
        # ChatML format used by Nemotron-3-Nano
        prompt = (
            f"<|im_start|>system\n"
            f"{system_prompt}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"When you decide to use a tool, respond with the tool name.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_request}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{assistant_prefill}"
        )

    elif model_type == "llama":
        # Llama 3 Instruct format
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"When you decide to use a tool, respond with the tool name."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_request}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_prefill}"
        )

    elif model_type == "mistral":
        prompt = (
            f"[INST] {system_prompt}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"User request: {user_request} [/INST]\n\n"
            f"{assistant_prefill}"
        )

    else:
        # Generic / fallback
        prompt = (
            f"System: {system_prompt}\n\n"
            f"Tools: {tool_descriptions}\n\n"
            f"User: {user_request}\n\n"
            f"Assistant: {assistant_prefill}"
        )

    return prompt


class RawActivationExtractor:
    """Extract raw activations at the decision token and save as numpy shards.

    Output format is compatible with CachedActivationBuffer:
        activations_dir/
            shard_000000.npy   # float32, shape (tokens_in_shard, d_model)
            shard_000001.npy
            ...
            metadata.json      # {model, layer, d_model, total_tokens, ...}
    """

    def __init__(
        self,
        base_extractor: Any,
        model_type: str = "auto",
        layer_key: str = "residual_20",
        tokenizer=None,
    ):
        self.extractor = base_extractor
        self.model_type = model_type
        self.layer_key = layer_key
        self.tokenizer = tokenizer or base_extractor.tokenizer

    def extract_to_shards(
        self,
        pairs: list[ContrastivePair],
        output_dir: str | Path,
        layer_keys: list[str] | None = None,
        batch_size: int = 512,
        shard_size: int = 500_000,
        show_progress: bool = True,
        scenarios_meta: dict | None = None,
        dp_size: int = 1,
        # Legacy single-scenario args (backward compat)
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Path]:
        """Extract raw activations for every prompt and save as numpy shards.

        Each contrastive pair contributes TWO activation vectors (one per
        prompt).  Activations are saved as float32 numpy shards compatible
        with ``CachedActivationBuffer``.

        When multiple ``layer_keys`` are provided, the base extractor returns
        all layers in a single forward pass.  Activations for each layer are
        written to separate subdirectories::

            output_dir/layer_10/activations/shard_*.npy
            output_dir/layer_20/activations/shard_*.npy

        Args:
            pairs: Contrastive pairs (text only).
            output_dir: Base directory for output.
            layer_keys: Layer keys to save (e.g. ``["residual_10", "residual_20"]``).
                If ``None``, falls back to ``[self.layer_key]``.
            batch_size: Number of prompts per GPU forward-pass batch.
            shard_size: Number of activation vectors per shard file.
            show_progress: Show a tqdm progress bar.
            scenarios_meta: {scenario_name: ScenarioConfig} for per-pair lookup.
            system_prompt: (Legacy) Single system prompt for all pairs.
            tools: (Legacy) Single tool list for all pairs.

        Returns:
            Dict mapping each layer_key to its output directory path.
        """
        from kiji_inspector.data.scenario import default_scenario

        output_dir = Path(output_dir)

        if layer_keys is None:
            layer_keys = [self.layer_key]

        # Build scenario lookup
        if scenarios_meta is not None:
            _scenarios = scenarios_meta
        elif system_prompt is not None and tools is not None:
            from kiji_inspector.data.scenario import ScenarioConfig

            legacy = ScenarioConfig(
                name="tool_selection",
                system_prompt=system_prompt,
                tools=tools,
                contrast_types={},
            )
            _scenarios = {legacy.name: legacy}
        else:
            ds = default_scenario()
            _scenarios = {ds.name: ds}

        _default = default_scenario()

        # Collect all prompts from both sides of each pair
        all_prompts: list[str] = []
        user_requests: list[str] = []
        for pair in pairs:
            scenario = _scenarios.get(pair.scenario_name, _default)
            all_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.anchor_prompt,
                    model_type=self.model_type,
                    tokenizer=self.tokenizer,
                )
            )
            user_requests.append(pair.anchor_prompt)
            all_prompts.append(
                build_agent_prompt(
                    system_prompt=scenario.system_prompt,
                    tools=scenario.tools,
                    user_request=pair.contrast_prompt,
                    model_type=self.model_type,
                    tokenizer=self.tokenizer,
                )
            )
            user_requests.append(pair.contrast_prompt)

        total_prompts = len(all_prompts)
        d_model = self.extractor.hidden_size

        # Per-layer output directories and shard state
        layer_dirs: dict[str, Path] = {}
        shard_buffers: dict[str, list[np.ndarray]] = {}
        shard_counts: dict[str, int] = {}
        shard_indices: dict[str, int] = {}
        total_written: dict[str, int] = {}

        for lk in layer_keys:
            # Parse layer number from "residual_N"
            layer_num = lk.split("_", 1)[1]
            ldir = output_dir / f"layer_{layer_num}" / "activations"
            ldir.mkdir(parents=True, exist_ok=True)
            layer_dirs[lk] = ldir
            shard_buffers[lk] = []
            shard_counts[lk] = 0
            shard_indices[lk] = 0
            total_written[lk] = 0

        pbar = tqdm(
            total=total_prompts,
            desc=f"Extracting activations ({len(layer_keys)} layer(s))",
            unit="prompt",
            disable=not show_progress,
        )

        if dp_size > 1 and hasattr(self.extractor, "config"):
            # Data-parallel: workers write shards directly — no temp files
            from kiji_inspector.extraction.vllm_activation_extractor import (
                run_dp_extraction_to_shards,
            )

            config_kwargs = {
                "model_name": self.extractor.config.model_name,
                "layers": self.extractor.config.layers,
                "token_positions": self.extractor.config.token_positions,
                "gpu_memory_utilization": self.extractor.config.gpu_memory_utilization,
                "tensor_parallel_size": getattr(self.extractor.config, "tensor_parallel_size", 1),
                "max_model_len": self.extractor.config.max_model_len,
                "trust_remote_code": self.extractor.config.trust_remote_code,
            }
            totals = run_dp_extraction_to_shards(
                prompts=all_prompts,
                dp_size=dp_size,
                config_kwargs=config_kwargs,
                batch_size=batch_size,
                layer_keys=layer_keys,
                layer_dirs={lk: str(layer_dirs[lk]) for lk in layer_keys},
                shard_size=shard_size,
            )
            for lk in layer_keys:
                total_written[lk] = totals[lk]
                shard_indices[lk] = len(list(layer_dirs[lk].glob("shard_*.npy")))

            pbar.update(total_prompts)
        else:
            # Single-GPU: use the existing extractor directly
            for i in range(0, total_prompts, batch_size):
                batch_prompts = all_prompts[i : i + batch_size]

                # Batched forward pass — returns all layers per prompt
                all_acts = self.extractor.extract_batch(
                    batch_prompts, batch_size=len(batch_prompts)
                )

                for act_dict in all_acts:
                    for lk in layer_keys:
                        vec = act_dict[lk]  # shape (d_model,), float32
                        shard_buffers[lk].append(vec.astype(np.float32))
                        shard_counts[lk] += 1

                        if shard_counts[lk] >= shard_size:
                            self._flush_shard(layer_dirs[lk], shard_indices[lk], shard_buffers[lk])
                            total_written[lk] += shard_counts[lk]
                            shard_indices[lk] += 1
                            shard_buffers[lk] = []
                            shard_counts[lk] = 0

                pbar.update(len(batch_prompts))

        # Flush remaining buffers
        for lk in layer_keys:
            if shard_buffers[lk]:
                self._flush_shard(layer_dirs[lk], shard_indices[lk], shard_buffers[lk])
                total_written[lk] += shard_counts[lk]
                shard_indices[lk] += 1

        pbar.close()

        # Write per-layer metadata.json and prompts.json
        for lk in layer_keys:
            ldir = layer_dirs[lk]

            metadata = {
                "model": self.extractor.config.model_name,
                "layer": lk,
                "d_model": d_model,
                "total_tokens": total_written[lk],
                "num_shards": shard_indices[lk],
                "shard_size": shard_size,
                "dtype": "float32",
                "source": "agentbench-sae-dataset",
                "prompts_per_pair": 2,
                "total_pairs": len(pairs),
            }
            with open(ldir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            with open(ldir / "prompts.json", "w") as f:
                json.dump(user_requests, f)

            print(
                f"  {lk}: {total_written[lk]} vectors across {shard_indices[lk]} shard(s) → {ldir}"
            )

        print(f"  Shape per vector: ({d_model},), dtype=float32")

        return layer_dirs

    @staticmethod
    def _flush_shard(
        output_dir: Path,
        shard_idx: int,
        buffer: list[np.ndarray],
    ) -> Path:
        """Stack buffered vectors and save as a numpy shard."""
        shard_data = np.stack(buffer, axis=0)  # (N, d_model), float32
        shard_path = output_dir / f"shard_{shard_idx:06d}.npy"
        np.save(shard_path, shard_data)
        return shard_path
