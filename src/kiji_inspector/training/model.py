"""
JumpReLU Sparse Autoencoder — training extensions.

Extends the inference-only base from ``kiji_inspector.sae_core`` with weight
initialisation, loss computation, decoder normalisation, and checkpoint
serialisation used during training.

Adapted from yaak-inspector-demo/sae_train/model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiji_inspector.sae_core import JumpReLUFunction
from kiji_inspector.sae_core import JumpReLUSAE as _BaseJumpReLUSAE


class JumpReLUSAE(_BaseJumpReLUSAE):
    """JumpReLU Sparse Autoencoder with training support.

    Adds weight initialisation, training loss, decoder normalisation,
    and checkpoint saving on top of the inference-only base class.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        dtype: torch.dtype = torch.bfloat16,
        bandwidth: float = 0.001,
        threshold_init: float = 0.01,
    ):
        super().__init__(
            d_model=d_model,
            d_sae=d_sae,
            dtype=dtype,
            bandwidth=bandwidth,
            threshold_init=threshold_init,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)

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
