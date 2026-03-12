"""SAE wrapper with from_pretrained that resolves base models to HF repos."""

from __future__ import annotations

import json
from typing import Optional

from huggingface_hub import hf_hub_download
from kiji_inspector.sae_core import JumpReLUSAE

from kiji_inspector.registry import resolve_repo_id


class SAE(JumpReLUSAE):
    """JumpReLU Sparse Autoencoder with HuggingFace Hub integration.

    Usage::

        from kiji_inspector import SAE

        sae, feature_descriptions = SAE.from_pretrained(
            base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            layer=20,
        )

        features = sae.encode(activations)
        reconstruction = sae.decode(features)
    """

    @classmethod
    def from_pretrained(
        cls,
        base_model: Optional[str] = None,
        layer: int = 0,
        device: str = "cpu",
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ) -> tuple["SAE", Optional[dict]]:
        """Download and load a single-layer SAE from HuggingFace.

        Resolves the base model name to a HuggingFace repo via the
        built-in registry, then downloads only the checkpoint and
        feature descriptions for the requested layer.

        Args:
            base_model: Model ID on HuggingFace, e.g.
                ``"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"``.
                Looked up in the registry to find the SAE repo.
            layer: Which layer to load, e.g. ``20``.
            device: ``"cpu"`` | ``"cuda"`` | ``"cuda:0"`` etc.
            repo_id: HuggingFace repo ID directly, bypassing the
                registry. Use this for repos not in the registry.
            cache_dir: Override the default HF cache directory.
            token: HuggingFace token for private repos.

        Returns:
            sae: Loaded SAE model (eval mode, on device).
            feature_descriptions: Dict of feature descriptions, or
                ``None`` if not available in the repo.

        Raises:
            KeyError: If ``base_model`` is not in the registry and
                ``repo_id`` is not provided.
            FileNotFoundError: If the checkpoint files are not found
                in the repo.
        """
        if repo_id is None:
            if base_model is None:
                raise ValueError("Provide either base_model or repo_id.")
            repo_id = resolve_repo_id(base_model)

        subfolder = f"layer_{layer}"

        def _download(filename: str, required: bool = True) -> Optional[str]:
            remote_path = f"{subfolder}/{filename}"
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    cache_dir=cache_dir,
                    token=token,
                )
            except Exception as e:
                if required:
                    raise FileNotFoundError(
                        f"Could not download '{remote_path}' from '{repo_id}'.\n"
                        f"  Make sure the repo contains {subfolder}/sae_checkpoints/ "
                        f"with sae_final.pt.\n"
                        f"  Original error: {e}"
                    )
                return None

        # --- Checkpoint ---
        checkpoint_path = _download("sae_checkpoints/sae_final.pt", required=True)
        sae = super().from_pretrained(checkpoint_path, device=device)
        sae.eval()

        # --- Feature descriptions (optional) ---
        feature_descriptions = None
        desc_path = _download("activations/feature_descriptions.json", required=False)
        if desc_path is not None:
            with open(desc_path) as f:
                feature_descriptions = json.load(f)

        return sae, feature_descriptions
