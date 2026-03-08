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
    version in ``sae.model`` which adds loss computation, weight initialisation,
    and serialisation helpers.
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

        return model.to(device)
