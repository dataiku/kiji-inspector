"""Registry mapping base model IDs to HuggingFace SAE repos."""

from __future__ import annotations

# base_model → HuggingFace repo containing the trained SAEs
MODEL_REGISTRY: dict[str, str] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "hanneshapke/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
}


def resolve_repo_id(base_model: str) -> str:
    """Resolve a base model name to its HuggingFace SAE repo ID.

    Args:
        base_model: Model ID as it appears on HuggingFace,
            e.g. ``"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"``.

    Raises:
        KeyError: If the model is not in the registry.
    """
    if base_model in MODEL_REGISTRY:
        return MODEL_REGISTRY[base_model]

    available = "\n  ".join(sorted(MODEL_REGISTRY))
    raise KeyError(
        f"No SAE repo registered for {base_model!r}.\n"
        f"Available models:\n  {available}\n"
        f"Or pass repo_id directly to SAE.from_pretrained()."
    )
