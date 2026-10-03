"""Behaviour-cloning policies (SETUP.md 9: start simple, MLP BC first).

`hidden=()` gives a plain linear policy, which is what Stage 0 uses: its
one-step update has a closed form, so batch gains are exact and the redundancy /
complementarity structure of a batch is directly readable from the spectrum of
its design matrix. The same class scales to an MLP for DMC / MetaWorld without
changing anything downstream.

Observations are standardized by default. Real simulators do not hand out
well-scaled states - DMC `reacher` observations have per-dimension standard
deviations spanning 0.10 to 3.66 and means up to 1.8 - and an unnormalized
MLP[128,128] on them diverges to NaN under the SGD settings that work fine for
the synthetic world. The statistics live in buffers, not parameters, so they
stay fixed across checkpoints and never enter `flat_params()`; the effect
estimators therefore still see a clean parameter vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class MLPPolicy(nn.Module):
    """Chunk-wise BC policy: obs_t -> act_t, applied independently per step."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (),
        activation: str = "relu",
        bias: bool = True,
        normalize_obs: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.normalize_obs = bool(normalize_obs)
        # buffers, not parameters: fixed across checkpoints and excluded from
        # flat_params(), so gradients and leave-one-out updates are unaffected
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_scale", torch.ones(self.obs_dim))
        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=bias), act_cls()]
            d = h
        layers.append(nn.Linear(d, act_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_obs_normalizer(self, obs: torch.Tensor | np.ndarray) -> None:
        """Set the observation statistics from data. Call once, before training."""
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32).reshape(-1, self.obs_dim)
        self.obs_mean.copy_(x.mean(0).to(self.obs_mean.device))
        self.obs_scale.copy_(x.std(0).clamp(min=1e-6).to(self.obs_scale.device))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """`obs` of shape (..., obs_dim) -> (..., act_dim)."""
        if self.normalize_obs:
            obs = (obs - self.obs_mean) / self.obs_scale
        return self.net(obs)

    def bc_loss(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.mean((self(obs) - act) ** 2)

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def load_flat_params(self, flat: torch.Tensor | np.ndarray) -> None:
        flat = torch.as_tensor(flat, dtype=torch.float32)
        off = 0
        with torch.no_grad():
            for p in self.parameters():
                k = p.numel()
                p.copy_(flat[off : off + k].view_as(p).to(p.device, p.dtype))
                off += k
        if off != flat.numel():
            raise ValueError(f"parameter vector has {flat.numel()} entries, expected {off}")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
