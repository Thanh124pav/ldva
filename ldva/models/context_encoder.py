"""Permutation-invariant context encoder (PLAN.md 4.2; SETUP.md 13).

DeepSets, with one design choice that matters downstream: the pooling statistics
are restricted to ones that support **exact leave-one-out** in O(N) rather than
O(N^2). The effect readout needs `h_{B\\i}` for every member of the batch, and

    mean_{B\\i} = (n * mean_B - phi_i) / (n - 1)

is exact for mean / sum pooling and for the second moment. Max pooling would
force a recompute per member, so it is offered only via `exact_loo=False`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ldva.models.common import masked_mean, mlp


@dataclass
class ContextEncoderConfig:
    latent_dim: int = 32
    hidden: Sequence[int] = field(default_factory=lambda: (128, 128))
    phi_dim: int = 128
    out_dim: int = 64
    pool: str = "mean"  # "mean" | "sum" | "max"
    use_second_moment: bool = True
    #: append the (normalized) batch size, which pooling alone discards
    use_size_feature: bool = True
    activation: str = "relu"
    size_scale: float = 16.0


class ContextEncoder(nn.Module):
    """ContextEncoder(Z_B) -> h_B, plus exact per-member leave-one-out."""

    def __init__(self, cfg: ContextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.phi = mlp(
            cfg.latent_dim, cfg.hidden, cfg.phi_dim, activation=cfg.activation,
            final_activation=True,
        )
        n_stats = 2 if cfg.use_second_moment else 1
        rho_in = cfg.phi_dim * n_stats + (1 if cfg.use_size_feature else 0)
        self.rho = mlp(rho_in, cfg.hidden, cfg.out_dim, activation=cfg.activation)

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    @property
    def exact_loo(self) -> bool:
        return self.cfg.pool in ("mean", "sum")

    # ---- pooling -------------------------------------------------------
    def _stats(self, z: torch.Tensor, mask: torch.Tensor):
        """Return (phi, first moment sum, second moment sum, true member count).

        `counts` is deliberately *un*clamped: an empty set must report size 0,
        otherwise the brute-force and exact leave-one-out paths disagree on
        single-member batches.
        """
        p = self.phi(z)
        m = mask.unsqueeze(-1).to(p.dtype)
        return p, (p * m).sum(1), (p * p * m).sum(1), m.sum(1)

    def _assemble(self, s1, s2, n_eff):
        """Build the rho input from raw sums and the member count `n_eff`.

        An empty set (`n_eff == 0`) yields an all-zero feature vector in every
        pooling mode, which is the canonical "no context" encoding.
        """
        denom = n_eff.clamp(min=1.0)
        if self.cfg.pool == "mean":
            f1, f2 = s1 / denom, s2 / denom
        else:  # sum
            f1, f2 = s1, s2
        parts = [f1]
        if self.cfg.use_second_moment:
            parts.append(f2)
        if self.cfg.use_size_feature:
            parts.append(n_eff / self.cfg.size_scale)
        return torch.cat(parts, dim=-1)

    # ---- forward -------------------------------------------------------
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`z` is (B, N, latent_dim), `mask` is (B, N) -> (B, out_dim)."""
        if self.cfg.pool == "max":
            p = self.phi(z)
            neg = torch.finfo(p.dtype).min
            f1 = p.masked_fill(~mask.unsqueeze(-1), neg).max(1).values
            parts = [f1]
            if self.cfg.use_second_moment:
                parts.append(masked_mean(p * p, mask))
            if self.cfg.use_size_feature:
                parts.append(mask.sum(1, keepdim=True).to(p.dtype) / self.cfg.size_scale)
            empty = ~mask.any(1, keepdim=True)
            feats = torch.cat(parts, dim=-1).masked_fill(empty, 0.0)
            return self.rho(feats)
        _, s1, s2, counts = self._stats(z, mask)
        return self.rho(self._assemble(s1, s2, counts))

    def leave_one_out(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h_{B\\i} for every i -> (B, N, out_dim).

        Members whose batch has a single element get the pooled statistics of an
        empty set (zeros), which is the correct "no context" signal.
        """
        if not self.exact_loo:
            return self._leave_one_out_bruteforce(z, mask)
        p, s1, s2, counts = self._stats(z, mask)
        # removing member i leaves sum - phi_i and one fewer member
        loo_s1 = s1.unsqueeze(1) - p
        loo_s2 = s2.unsqueeze(1) - p * p
        n_eff = (counts.unsqueeze(1) - 1.0).clamp(min=0.0).expand(-1, z.shape[1], -1)
        h = self.rho(self._assemble(loo_s1, loo_s2, n_eff))
        # padding slots hold meaningless statistics; zero them so they cannot
        # leak into a loss through a mis-applied mask
        return h * mask.unsqueeze(-1).to(h.dtype)

    def _leave_one_out_bruteforce(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """O(N^2) fallback for pooling that is not LOO-decomposable."""
        outs = []
        for i in range(mask.shape[1]):
            m = mask.clone()
            m[:, i] = False
            outs.append(self.forward(z, m))
        return torch.stack(outs, dim=1) * mask.unsqueeze(-1).to(z.dtype)


class SetTransformerContextEncoder(nn.Module):
    """Set Transformer alternative (PLAN.md 4.2 "alternative later").

    Attention makes pooling non-decomposable, so leave-one-out is the O(N^2)
    recompute. Kept available for the encoder ablation, not the MVP default.
    """

    def __init__(self, cfg: ContextEncoderConfig, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.latent_dim, cfg.phi_dim)
        layer = nn.TransformerEncoderLayer(
            cfg.phi_dim, n_heads, dim_feedforward=4 * cfg.phi_dim, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.rho = mlp(cfg.phi_dim + 1, cfg.hidden, cfg.out_dim, activation=cfg.activation)

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim

    @property
    def exact_loo(self) -> bool:
        return False

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.proj(z), src_key_padding_mask=~mask)
        h = masked_mean(h, mask)
        size = mask.sum(1, keepdim=True).to(h.dtype) / self.cfg.size_scale
        return self.rho(torch.cat([h, size], dim=-1))

    def leave_one_out(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(mask.shape[1]):
            m = mask.clone()
            m[:, i] = False
            outs.append(self.forward(z, m))
        return torch.stack(outs, dim=1)


def build_context_encoder(cfg: ContextEncoderConfig, kind: str = "deepsets") -> nn.Module:
    if kind == "deepsets":
        return ContextEncoder(cfg)
    if kind == "set_transformer":
        return SetTransformerContextEncoder(cfg)
    raise ValueError(f"unknown context encoder {kind!r}")
