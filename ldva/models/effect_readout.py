"""Contextual effect readout R(z_i, h_{B\\i}, theta) -> s_hat_i (PLAN.md 4.3).

The readout exists mainly to *shape the latent geometry*: it is the only loss
term that forces `z_i` to carry information about how the sample behaves under
varying company. Conditioning on `h_{B\\i}` rather than `h_B` matters - with
`h_B` the readout could read `z_i`'s own contribution back out of the pooled
context and satisfy the loss without the latent meaning anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ldva.models.common import mlp


@dataclass
class EffectReadoutConfig:
    latent_dim: int = 32
    context_dim: int = 64
    policy_dim: int = 0
    hidden: Sequence[int] = field(default_factory=lambda: (128, 128))
    activation: str = "relu"
    #: drop the context input entirely - the "scalar score" ablation (PLAN.md 18.1)
    use_context: bool = True
    dropout: float = 0.0


class EffectReadout(nn.Module):
    def __init__(self, cfg: EffectReadoutConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.latent_dim + cfg.policy_dim
        if cfg.use_context:
            in_dim += cfg.context_dim
        self.net = mlp(
            in_dim, cfg.hidden, 1, activation=cfg.activation, dropout=cfg.dropout
        )

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
        policy_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z` is (B, N, d_z), `context` is (B, N, d_h) -> (B, N)."""
        parts = [z]
        if self.cfg.use_context:
            if context is None:
                raise ValueError("use_context=True but no context tensor was passed")
            parts.append(context)
        if self.cfg.policy_dim > 0:
            if policy_context is None:
                raise ValueError("policy_dim > 0 but no policy_context was passed")
            parts.append(policy_context.unsqueeze(1).expand(-1, z.shape[1], -1))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class ScalarScoreReadout(nn.Module):
    """Ablation 1 of PLAN.md 18: one fixed scalar value per sample.

    Implemented as a readout that ignores both the context and the policy, so
    the comparison isolates *contextuality* rather than model capacity - the
    parameter count and depth stay the same.
    """

    def __init__(self, cfg: EffectReadoutConfig):
        super().__init__()
        self.cfg = cfg
        self.net = mlp(cfg.latent_dim, cfg.hidden, 1, activation=cfg.activation)

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
        policy_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def build_effect_readout(cfg: EffectReadoutConfig, kind: str = "contextual") -> nn.Module:
    if kind == "contextual":
        return EffectReadout(cfg)
    if kind == "scalar":
        return ScalarScoreReadout(cfg)
    raise ValueError(f"unknown readout {kind!r}")
