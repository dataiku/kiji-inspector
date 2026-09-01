"""Registry mapping base model IDs to HuggingFace SAE repos."""

from __future__ import annotations

# base_model → HuggingFace repo containing the trained SAEs
MODEL_REGISTRY: dict[str, str] = {
    # NVIDIA
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": "575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16": "575-lab/kiji-inspector-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    # Google
    "google/gemma-4-E4B-it": "575-lab/kiji-inspector-google-gemma-4-E4B-it",
    # EXPERIMENTAL
    "google/gemma-3-27b-it": "575-lab/kiji-inspector-google-gemma-3-27b-it",
    # Qwen
    "Qwen/Qwen3.6-35B-A3B": "575-lab/kiji-inspector-Qwen-Qwen3.6-35B-A3B",
}

# repo → the commit the published results were produced against.
#
# ``hf_hub_download`` defaults to ``main``, which is a moving target: a repo
# that gains a file or a corrected checkpoint silently changes what a rerun
# loads, and nothing in the output records which version it saw.  The base
# model demonstrates the hazard rather than merely risking it --- the runs used
# ``d468880b``, and upstream ``main`` has since moved on.
#
# Pass ``revision`` explicitly to override, or ``revision="main"`` to opt back
# into whatever is current.
MODEL_REVISIONS: dict[str, str] = {
    "575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "1888e30db91c6d16194bea993fdf6a8701a74986",
    "575-lab/kiji-inspector-NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": "e3dc6cca61675050b94ccb79b44b2c0fe1013fef",
    "575-lab/kiji-inspector-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "2380c95cbeddeb8a7aca17a122bc18df5ae62aaa",
    "575-lab/kiji-inspector-google-gemma-4-E4B-it": "41ab88ba969ec5672c813d48dbd65931e74fc5fc",
    "575-lab/kiji-inspector-google-gemma-3-27b-it": "efdb6a6d3ae96a27ca3d0e5a0aab13bfa7a2cc1f",
    "575-lab/kiji-inspector-Qwen-Qwen3.6-35B-A3B": "d93fe627c223334a2c9b4dddfdabc4270aaa1851",
}

# The base checkpoints the SAEs were trained on, pinned for the same reason.
# Not used by the loader --- these are weights the user supplies --- but the
# provenance record and the paper both cite them, so they live beside the rest.
BASE_MODEL_REVISIONS: dict[str, str] = {
    "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16": "d468880b6ad3c6e0d21377ce7242adaea4cc884d",
}


def resolve_revision(repo_id: str) -> str | None:
    """The pinned commit for an SAE repo, or ``None`` to track ``main``."""
    return MODEL_REVISIONS.get(repo_id)


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
