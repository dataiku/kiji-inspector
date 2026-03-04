"""
Activation extractor using vLLM's built-in activation capture.

Drop-in replacement for ActivationExtractor that uses vLLM's
``extract_activation_layers`` API instead of HuggingFace Transformers
with manual forward hooks.  Significantly faster for bulk extraction
thanks to vLLM's continuous batching and optimized kernels.

Requires a patched vLLM build that supports ``extract_activation_layers``
in ``SamplingParams``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class VLLMActivationConfig:
    """Configuration for vLLM-based activation extraction."""

    model_name: str = ""
    layers: list[int] = field(default_factory=lambda: [20])
    token_positions: str = "decision"  # "all", "last", "decision"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    trust_remote_code: bool = True


class VLLMActivationExtractor:
    """Extract activations using vLLM's activation extraction API.

    Presents the same interface as ``ActivationExtractor`` (``extract()``,
    ``extract_batch()``, ``cleanup()``, ``tokenizer``, ``hidden_size``)
    so it can be used as a drop-in replacement wherever the HuggingFace
    Transformers-based extractor is used.
    """

    def __init__(self, config: VLLMActivationConfig):
        if not config.model_name:
            raise ValueError("VLLMActivationConfig.model_name is required.")
        self.config = config

        from vllm import LLM

        print(f"Loading model via vLLM: {config.model_name}")
        print(f"  layers: {config.layers}")
        print(f"  tensor_parallel_size: {config.tensor_parallel_size}")

        self.llm = LLM(
            model=config.model_name,
            enforce_eager=True,
            trust_remote_code=config.trust_remote_code,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            disable_log_stats=True,
        )

        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.llm.model_config.hf_text_config.hidden_size
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  token_positions: {config.token_positions}")

    def extract(
        self,
        prompt: str,
        decision_token_offset: int = -1,
    ) -> dict[str, np.ndarray]:
        """Extract activations for a single prompt.

        Args:
            prompt: The full formatted prompt text.
            decision_token_offset: Token position to extract from.
                -1 means last token (the decision point).

        Returns:
            Dictionary mapping layer names to activation vectors.
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extract_activation_layers=self.config.layers,
        )
        outputs = self.llm.generate([prompt], sp)
        activations = outputs[0].outputs[0].activations

        if not activations:
            raise RuntimeError("vLLM did not return activations. Check extract_activation_layers.")

        result = {}
        for layer_idx, tensor in activations.items():
            # tensor shape: (prompt_tokens + generated_tokens, hidden_size)
            # Drop the generated token to get prompt-only activations
            prompt_acts = tensor[:-1]
            key = f"residual_{layer_idx}"

            if self.config.token_positions in ("last", "decision"):
                result[key] = prompt_acts[decision_token_offset].float().numpy()
            elif self.config.token_positions == "all":
                result[key] = prompt_acts.float().numpy()
            else:
                raise ValueError(f"Unknown token_positions: {self.config.token_positions}")

        return result

    def extract_batch(
        self,
        prompts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, np.ndarray]]:
        """Extract activations for multiple prompts.

        vLLM handles batching internally via its scheduler, so all prompts
        are submitted at once.  The ``batch_size`` parameter is accepted
        for API compatibility but chunking is only used as a memory safety
        measure for very large prompt lists.
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extract_activation_layers=self.config.layers,
        )
        results: list[dict[str, np.ndarray]] = []

        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            outputs = self.llm.generate(chunk, sp)

            for output in outputs:
                activations = output.outputs[0].activations
                if not activations:
                    raise RuntimeError("vLLM did not return activations.")

                item = {}
                for layer_idx, tensor in activations.items():
                    prompt_acts = tensor[:-1]
                    key = f"residual_{layer_idx}"

                    if self.config.token_positions in ("last", "decision"):
                        position = prompt_acts.shape[0] - 1
                        item[key] = prompt_acts[position].float().numpy()
                    elif self.config.token_positions == "all":
                        item[key] = prompt_acts.float().numpy()
                    else:
                        raise ValueError(
                            f"Unknown token_positions: {self.config.token_positions}"
                        )

                results.append(item)

        return results

    def cleanup(self):
        """Free vLLM resources and GPU memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()
