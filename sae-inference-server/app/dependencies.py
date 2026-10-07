from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

from app.config import (
    SAE_BASE_MODEL,
    SAE_CHECKPOINT_PATH,
    SAE_DEVICE,
    SAE_LAYER,
    SAE_REPO_ID,
)
from kiji_inspector import SAE
from kiji_inspector.core.sae_core import JumpReLUSAE


class SAEEngine:
    """Wraps a loaded SAE and its feature descriptions for the API layer."""

    def __init__(self, sae: JumpReLUSAE, feature_descriptions: Optional[dict]) -> None:
        self.sae = sae
        self.feature_descriptions = feature_descriptions or {}

    @staticmethod
    def _normalize_description(desc: object) -> dict | None:
        """Convert a feature description to a FeatureDescription-compatible dict."""
        if isinstance(desc, dict):
            return desc
        if isinstance(desc, str) and desc != "unknown":
            return {"label": desc, "description": desc}
        return None

    def describe(self, activation: list[float], top_k: int) -> dict:
        parameter = next(self.sae.parameters())
        x = torch.tensor(activation, dtype=torch.float32, device=parameter.device)
        if x.ndim != 1 or x.numel() != self.sae.d_model:
            raise ValueError(f"Expected one activation vector of size {self.sae.d_model}")
        if not torch.isfinite(x).all():
            raise ValueError("Activation values must be finite")
        with torch.inference_mode():
            x = self.sae.normalize_input(x).to(dtype=parameter.dtype)
            encoded = self.sae.encode(x.unsqueeze(0)).squeeze(0)
            values, indices = torch.topk(encoded, min(top_k, encoded.numel()))
            results = [
                (
                    index.item(),
                    self.sae._lookup_feature_description(self.feature_descriptions, index.item()),
                    value.item(),
                )
                for value, index in zip(values, indices, strict=True)
                if value.item() > 0
            ]
        top_features = [
            {
                "feature_id": str(feature_id),
                "activation": activation_value,
                "description": self._normalize_description(description),
            }
            for feature_id, description, activation_value in results
        ]
        return {
            "top_features": top_features,
            "num_active_features": int((encoded > 0).sum().item()),
        }


@lru_cache
def get_engine() -> SAEEngine:
    if SAE_CHECKPOINT_PATH:
        sae = JumpReLUSAE.from_pretrained(SAE_CHECKPOINT_PATH, device=SAE_DEVICE)
        sae.eval()
        feature_descriptions = None
    else:
        sae, feature_descriptions = SAE.from_pretrained(
            base_model=SAE_BASE_MODEL,
            layer=SAE_LAYER,
            device=SAE_DEVICE,
            repo_id=SAE_REPO_ID,
        )
    return SAEEngine(sae, feature_descriptions)
