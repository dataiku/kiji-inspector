"""
JumpReLU Sparse Autoencoder — inference-only base implementation.

Architecture:
    input x (d_model)
        |
        +-> W_enc @ (x - b_dec) + b_enc -> JumpReLU(-, theta) -> features (d_sae)
        |
        +-> W_dec @ features + b_dec -> reconstruction (d_model)

    where JumpReLU(z, theta) = z * H(z - theta)  (H is Heaviside step function)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JumpReLUFunction(torch.autograd.Function):
    """JumpReLU activation with STE gradients for the threshold."""

    @staticmethod
    def forward(ctx, z, threshold, bandwidth):
        ctx.save_for_backward(z, threshold)
        ctx.bandwidth = bandwidth
        mask = (z > threshold).to(z.dtype)
        return z * mask

    @staticmethod
    def backward(ctx, grad_output):
        z, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Straight-through for z where active
        mask = (z > threshold).to(z.dtype)
        grad_z = grad_output * mask

        # Rectangular kernel approximation for threshold gradient
        diff = z - threshold
        in_window = (diff.abs() < bandwidth).to(z.dtype)
        grad_threshold = -(grad_output * z * in_window / (2 * bandwidth)).sum(dim=0)

        return grad_z, grad_threshold, None


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder with learnable per-feature thresholds.

    This is the inference-only base class. For training, use the extended
    version in ``kiji_inspector.training.model`` which adds loss computation,
    weight initialisation, and serialisation helpers.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        dtype: torch.dtype = torch.bfloat16,
        bandwidth: float = 0.001,
        threshold_init: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.dtype = dtype
        self.bandwidth = bandwidth
        self.rms_scale: float | None = None  # Set from training; used to normalize inputs

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        self.threshold = nn.Parameter(torch.full((d_sae,), threshold_init, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=dtype))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        pre_activation = F.linear(x - self.b_dec, self.W_enc.t(), self.b_enc)
        return JumpReLUFunction.apply(pre_activation, self.threshold, self.bandwidth)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.W_dec.t(), self.b_dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, features)."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _lookup_feature_description(features: dict[int | str, str], feature_id: int) -> str:
        description = features.get(feature_id)
        if description is None:
            description = features.get(str(feature_id))
        return description if description is not None else "unknown"

    def describe(
        self, x: torch.Tensor, features: dict[int | str, str], top_k: int = 5
    ) -> list[tuple[int, str, float]]:
        """Return descriptions for the top active features.

        Args:
            x: Input activations, shape ``(d_model,)`` or ``(1, d_model)``.
            features: Mapping from feature dimension index to its description string.
                Accepts either integer keys or stringified integer keys from JSON.
            top_k: Maximum number of active features to return.

        Returns:
            List of ``(feature_id, description, activation_value)`` tuples,
            sorted by activation value descending. Inactive features
            (activation ``<= 0``) are excluded.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        encoded = self.encode(x).squeeze(0)  # (d_sae,)
        k = min(top_k, encoded.shape[0])
        top_values, top_indices = torch.topk(encoded, k)
        return [
            (
                idx.item(),
                self._lookup_feature_description(features, idx.item()),
                val.item(),
            )
            for idx, val in zip(top_indices, top_values, strict=True)
            if val.item() > 0
        ]

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> JumpReLUSAE:
        """Load from a .pt checkpoint saved by the training code."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint.get("config", {})
        state = checkpoint.get("model_state_dict", checkpoint)
        w_enc = state.get("W_enc", state.get("model_state_dict", {}).get("W_enc"))

        d_model = config.get("d_model", w_enc.shape[0])
        d_sae = config.get("d_sae", w_enc.shape[1])
        dtype_str = config.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
        if torch.device(device).type == "cpu" and dtype == torch.bfloat16:
            dtype = torch.float32
        bandwidth = config.get("bandwidth", 0.001)
        threshold_init = config.get("threshold_init", 0.01)

        model = cls(
            d_model=d_model,
            d_sae=d_sae,
            dtype=dtype,
            bandwidth=bandwidth,
            threshold_init=threshold_init,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.W_enc.data = checkpoint["W_enc"]
            model.b_enc.data = checkpoint["b_enc"]
            model.threshold.data = checkpoint["threshold"]
            model.W_dec.data = checkpoint["W_dec"]
            model.b_dec.data = checkpoint["b_dec"]

        model.rms_scale = config.get("rms_scale", None)

        return model.to(device)
