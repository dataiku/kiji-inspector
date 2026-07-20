from kiji_inspector.extraction.activation_extractor import ActivationConfig, ActivationExtractor
from kiji_inspector.extraction.extractor import RawActivationExtractor, build_agent_prompt
from kiji_inspector.extraction.vllm_activation_extractor import (
    VLLMActivationConfig,
    VLLMActivationExtractor,
    recommended_vllm_kwargs,
    run_dp_extraction_to_shards,
)

__all__ = [
    "ActivationConfig",
    "ActivationExtractor",
    "VLLMActivationConfig",
    "VLLMActivationExtractor",
    "RawActivationExtractor",
    "build_agent_prompt",
    "create_extractor",
    "recommended_vllm_kwargs",
    "run_dp_extraction_to_shards",
]


def create_extractor(
    backend: str,
    model_name: str,
    layers: list[int],
    token_positions: str = "decision",
    **kwargs,
):
    """Create an activation extractor using the specified backend.

    Args:
        backend: ``"vllm"`` for vLLM-based extraction (fast, model-agnostic)
            or ``"hf"`` for HuggingFace Transformers (required for ablation).
        model_name: HuggingFace model ID.
        layers: Transformer layer indices to extract from.
        token_positions: ``"decision"``/``"last"`` for single-token or
            ``"all"`` for per-token extraction.
        **kwargs: Additional config options forwarded to the backend.

    Returns:
        An extractor with ``extract()``, ``extract_batch()``, ``cleanup()``,
        ``tokenizer``, and ``hidden_size``.
    """
    if backend == "vllm":
        # Apply model-family defaults (e.g. language_model_only for hybrid
        # multimodal Qwen), letting explicit kwargs win.
        merged = {**recommended_vllm_kwargs(model_name), **kwargs}
        config = VLLMActivationConfig(
            model_name=model_name,
            layers=layers,
            token_positions=token_positions,
            **merged,
        )
        return VLLMActivationExtractor(config)
    elif backend == "hf":
        import torch

        config = ActivationConfig(
            model_name=model_name,
            layers=layers,
            token_positions=token_positions,
            dtype=kwargs.get("dtype", torch.bfloat16),
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        return ActivationExtractor(config)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Expected 'vllm' or 'hf'.")
