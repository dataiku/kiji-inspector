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

import numpy as np
from tqdm import tqdm

from data.contrastive_dataset import ContrastivePair
from extraction.activation_extractor import ActivationExtractor


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
            shard_000000.npy   # float16, shape (tokens_in_shard, d_model)
            shard_000001.npy
            ...
            metadata.json      # {model, layer, d_model, total_tokens, ...}
    """

    def __init__(
        self,
        base_extractor: ActivationExtractor,
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
        batch_size: int = 512,
        shard_size: int = 500_000,
        show_progress: bool = True,
        scenarios_meta: dict | None = None,
        # Legacy single-scenario args (backward compat)
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> Path:
        """Extract raw activations for every prompt and save as numpy shards.

        Each contrastive pair contributes TWO activation vectors (one per
        prompt).  Activations are saved as float16 numpy shards compatible
        with ``CachedActivationBuffer``.

        Per-pair tools and system prompts are looked up from
        ``scenarios_meta`` (a dict mapping scenario name to ScenarioConfig).
        Falls back to the legacy ``system_prompt``/``tools`` args or
        the built-in default scenario.

        Args:
            pairs: Contrastive pairs (text only).
            output_dir: Directory for numpy shards + metadata.json.
            batch_size: Number of prompts per GPU forward-pass batch.
            shard_size: Number of activation vectors per shard file.
            show_progress: Show a tqdm progress bar.
            scenarios_meta: {scenario_name: ScenarioConfig} for per-pair lookup.
            system_prompt: (Legacy) Single system prompt for all pairs.
            tools: (Legacy) Single tool list for all pairs.

        Returns:
            Path to the output directory.
        """
        from data.scenario import default_scenario

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build scenario lookup
        if scenarios_meta is not None:
            _scenarios = scenarios_meta
        elif system_prompt is not None and tools is not None:
            # Legacy single-scenario mode
            from data.scenario import ScenarioConfig

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

        # Streaming extraction: accumulate into shard buffer, flush when full
        shard_buffer: list[np.ndarray] = []
        shard_buffer_count = 0
        shard_idx = 0
        total_written = 0
        shard_paths: list[Path] = []

        pbar = tqdm(
            total=total_prompts,
            desc="Extracting activations",
            unit="prompt",
            disable=not show_progress,
        )

        for i in range(0, total_prompts, batch_size):
            batch_prompts = all_prompts[i : i + batch_size]

            # Batched forward pass
            all_acts = self.extractor.extract_batch(batch_prompts, batch_size=len(batch_prompts))

            for act_dict in all_acts:
                vec = act_dict[self.layer_key]  # shape (d_model,), float32
                shard_buffer.append(vec.astype(np.float16))
                shard_buffer_count += 1

                if shard_buffer_count >= shard_size:
                    shard_path = self._flush_shard(output_dir, shard_idx, shard_buffer)
                    shard_paths.append(shard_path)
                    total_written += shard_buffer_count
                    shard_idx += 1
                    shard_buffer = []
                    shard_buffer_count = 0

            pbar.update(len(batch_prompts))

        # Flush remaining
        if shard_buffer:
            shard_path = self._flush_shard(output_dir, shard_idx, shard_buffer)
            shard_paths.append(shard_path)
            total_written += shard_buffer_count
            shard_idx += 1

        pbar.close()

        # Write metadata.json
        metadata = {
            "model": self.extractor.config.model_name,
            "layer": self.layer_key,
            "d_model": d_model,
            "total_tokens": total_written,
            "num_shards": shard_idx,
            "shard_size": shard_size,
            "dtype": "float16",
            "source": "agentbench-sae-dataset",
            "prompts_per_pair": 2,
            "total_pairs": len(pairs),
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Write prompts.json -- user request text for each activation vector,
        # in the same order as the numpy shards.  Used by Step 5 to map
        # activations back to prompt text without re-running inference.
        prompts_path = output_dir / "prompts.json"
        with open(prompts_path, "w") as f:
            json.dump(user_requests, f)

        print(
            f"Wrote {total_written} activation vectors across {shard_idx} shard(s) to {output_dir}"
        )
        print(f"  Shape per vector: ({d_model},), dtype=float16")
        print(f"  Metadata: {metadata_path}")
        print(f"  Prompts: {prompts_path} ({len(user_requests)} entries)")

        return output_dir

    @staticmethod
    def _flush_shard(
        output_dir: Path,
        shard_idx: int,
        buffer: list[np.ndarray],
    ) -> Path:
        """Stack buffered vectors and save as a numpy shard."""
        shard_data = np.stack(buffer, axis=0)  # (N, d_model), float16
        shard_path = output_dir / f"shard_{shard_idx:06d}.npy"
        np.save(shard_path, shard_data)
        return shard_path
