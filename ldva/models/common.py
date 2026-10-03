"""Shared building blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    layer_norm: bool = False,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if final_activation:
        layers.append(act())
    return nn.Sequential(*layers)


class FiLM(nn.Module):
    """Feature-wise affine conditioning: h <- gamma(c) * h + beta(c).

    Used to condition a sample representation on the policy context without
    letting the context dominate the representation by sheer width, which a
    plain concatenation tends to do when `cond_dim` is comparable to `dim`.
    """

    def __init__(self, dim: int, cond_dim: int, hidden: int = 64):
        super().__init__()
        self.net = mlp(cond_dim, [hidden], 2 * dim)
        self.dim = dim

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean over `dim`, ignoring masked-out entries."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim)
