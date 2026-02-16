"""
JumpReLU Sparse Autoencoder.

Architecture:
    input x (d_model)
        |
        +-> W_enc @ (x - b_dec) + b_enc -> JumpReLU(-, theta) -> features (d_sae)
        |
        +-> W_dec @ features + b_dec -> reconstruction (d_model)

    where JumpReLU(z, theta) = z * H(z - theta)  (H is Heaviside step function)

Adapted from yaak-inspector-demo/sae_train/model.py.
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
    """JumpReLU Sparse Autoencoder with learnable per-feature thresholds."""

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

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        pre_activation = F.linear(x - self.b_dec, self.W_enc.t(), self.b_enc)
        return JumpReLUFunction.apply(pre_activation, self.threshold, self.bandwidth)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.W_dec.t(), self.b_dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, features, pre_activation)."""
        pre_activation = (x - self.b_dec) @ self.W_enc + self.b_enc
        features = JumpReLUFunction.apply(pre_activation, self.threshold, self.bandwidth)
        reconstruction = features @ self.W_dec + self.b_dec
        return reconstruction, features, pre_activation

    def compute_loss(
        self,
        x: torch.Tensor,
        l1_coefficient: float = 5e-3,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE reconstruction + tanh-smoothed L0 sparsity loss."""
        reconstruction, features, pre_activation = self.forward(x)

        recon_loss = F.mse_loss(reconstruction, x)
        l0 = (features.abs() > 0).float().sum(dim=-1).mean()

        # Tanh sparsity: smooth approximation providing gradients to encoder + thresholds
        tanh_vals = torch.tanh((pre_activation - self.threshold) / self.bandwidth)
        sparsity_loss = torch.relu(tanh_vals).sum(dim=-1).mean()

        total_loss = recon_loss + l1_coefficient * sparsity_loss

        with torch.no_grad():
            feature_activity = (features.abs() > 0).any(dim=0).float().mean()

        metrics = {
            "loss/total": total_loss.item(),
            "loss/reconstruction": recon_loss.item(),
            "loss/sparsity": sparsity_loss.item(),
            "sparsity/l0": l0.item(),
            "sparsity/feature_activity": feature_activity.item(),
            "threshold/mean": self.threshold.mean().item(),
            "threshold/std": self.threshold.std().item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.div_(norms.clamp(min=1e-8))

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> JumpReLUSAE:
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

    def save_pretrained(self, path: str, config: dict | None = None):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "d_model": self.d_model,
                "d_sae": self.d_sae,
                "dtype": str(self.dtype).split(".")[-1],
                "bandwidth": self.bandwidth,
                **(config or {}),
            },
        }
        torch.save(checkpoint, path)
