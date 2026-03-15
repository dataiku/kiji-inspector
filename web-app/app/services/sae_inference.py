from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class SAEInference:
    def __init__(self, model_path: Path, descriptions_path: Path) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError("Unexpected checkpoint format: expected dict with model_state_dict")

        state_dict = checkpoint["model_state_dict"]
        self.w_enc = state_dict["W_enc"].float()
        self.b_enc = state_dict["b_enc"].float()
        self.threshold = state_dict["threshold"].float()
        self.b_dec = state_dict["b_dec"].float()

        self.d_model = int(self.w_enc.shape[0])
        self.d_sae = int(self.w_enc.shape[1])

        with descriptions_path.open("r", encoding="utf-8") as f:
            self.descriptions: dict[str, dict[str, Any]] = json.load(f)

    def encode(self, activation: list[float]) -> torch.Tensor:
        if len(activation) != self.d_model:
            raise ValueError(
                f"Invalid activation length: expected {self.d_model}, got {len(activation)}"
            )

        x = torch.tensor(activation, dtype=torch.float32)
        pre = (x - self.b_dec) @ self.w_enc + self.b_enc
        # Match training-time JumpReLU: z * H(z - theta), not ReLU(z - theta).
        return pre * (pre > self.threshold).to(pre.dtype)

    def describe(self, activation: list[float], top_k: int) -> dict[str, Any]:
        features = self.encode(activation)
        values, indices = torch.topk(features, k=min(top_k, self.d_sae))

        top_features: list[dict[str, Any]] = []
        for score_tensor, idx_tensor in zip(values, indices):
            score = float(score_tensor.item())
            if score <= 0:
                continue

            feature_id = str(int(idx_tensor.item()))
            top_features.append(
                {
                    "feature_id": feature_id,
                    "activation": score,
                    "description": self.descriptions.get(feature_id),
                }
            )

        return {
            "top_features": top_features,
            "num_active_features": int((features > 0).sum().item()),
        }
