"""Local metadata->latent direction model G_omega (PLAN.md 5.5, 12).

    G_omega(z_i, m_i, delta_m) -> delta_z_hat

Deliberately *local*: the model sees the anchor `(z_i, m_i)` as well as the
perturbation, so it never has to represent a single global invertible map from
metadata to latent - only assumption A2, that small controllable metadata
changes produce locally predictable latent changes.

`residual_jacobian=True` makes the network predict a local linear map applied
to `delta_m`, which keeps the output exactly linear in `delta_m` for small
steps and lets us read off a Jacobian to invert in the acquisition planner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ldva.models.common import mlp


@dataclass
class MetadataDirectionConfig:
    latent_dim: int = 32
    meta_dim: int = 3
    hidden: Sequence[int] = field(default_factory=lambda: (128, 128))
    activation: str = "relu"
    #: predict a local Jacobian rather than delta_z directly
    residual_jacobian: bool = True


class MetadataDirectionModel(nn.Module):
    def __init__(self, cfg: MetadataDirectionConfig):
        super().__init__()
        self.cfg = cfg
        anchor_dim = cfg.latent_dim + cfg.meta_dim
        if cfg.residual_jacobian:
            self.net = mlp(
                anchor_dim, cfg.hidden, cfg.latent_dim * cfg.meta_dim,
                activation=cfg.activation,
            )
        else:
            self.net = mlp(
                anchor_dim + cfg.meta_dim, cfg.hidden, cfg.latent_dim,
                activation=cfg.activation,
            )

    def jacobian(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Local d z / d m at the anchor -> (..., latent_dim, meta_dim)."""
        if not self.cfg.residual_jacobian:
            raise RuntimeError("jacobian() needs residual_jacobian=True")
        flat = self.net(torch.cat([z, m], dim=-1))
        return flat.reshape(*flat.shape[:-1], self.cfg.latent_dim, self.cfg.meta_dim)

    def forward(
        self, z: torch.Tensor, m: torch.Tensor, delta_m: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg.residual_jacobian:
            J = self.jacobian(z, m)
            return (J @ delta_m.unsqueeze(-1)).squeeze(-1)
        return self.net(torch.cat([z, m, delta_m], dim=-1))
