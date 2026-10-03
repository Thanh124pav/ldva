"""Batch utility model F_psi(Z_B, D, theta) -> V_hat(B) (PLAN.md 4.4).

This is the model the acquisition planner actually queries, so it has to
represent redundancy, complementarity and saturation rather than a sum of
per-sample values. Three variants:

- `BatchUtilityModel`  : DeepSets over Z_B, free to learn interactions.
- `AdditiveBatchUtility`: V = pool_i f(z_i). Structurally additive, and the
  control for ablation 3 / failure mode F2 - if it matches the DeepSets model,
  the batch-interaction motivation is not supported by the data.
- `PairwiseBatchUtility`: explicit first order + pairwise kernel, which is
  interpretable enough to read redundancy and complementarity off directly.

All three take an optional `dataset_context` summarizing D_t, because the value
of a *new* batch depends on what is already owned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ldva.models.common import masked_mean, masked_sum, mlp


@dataclass
class BatchUtilityConfig:
    latent_dim: int = 32
    hidden: Sequence[int] = field(default_factory=lambda: (128, 128))
    phi_dim: int = 128
    policy_dim: int = 0
    dataset_dim: int = 0
    pool: str = "mean"
    use_second_moment: bool = True
    use_size_feature: bool = True
    activation: str = "relu"
    size_scale: float = 16.0
    dropout: float = 0.0


class _UtilityBase(nn.Module):
    def __init__(self, cfg: BatchUtilityConfig):
        super().__init__()
        self.cfg = cfg

    def _side_inputs(
        self,
        batch: int,
        policy_context: torch.Tensor | None,
        dataset_context: torch.Tensor | None,
        device,
        dtype,
    ) -> list[torch.Tensor]:
        parts = []
        if self.cfg.policy_dim > 0:
            if policy_context is None:
                raise ValueError("policy_dim > 0 but no policy_context was passed")
            parts.append(policy_context)
        if self.cfg.dataset_dim > 0:
            if dataset_context is None:
                raise ValueError("dataset_dim > 0 but no dataset_context was passed")
            if dataset_context.ndim == 1:
                dataset_context = dataset_context.unsqueeze(0).expand(batch, -1)
            parts.append(dataset_context)
        return parts


class BatchUtilityModel(_UtilityBase):
    """DeepSets utility; free to represent arbitrary set interactions."""

    def __init__(self, cfg: BatchUtilityConfig):
        super().__init__(cfg)
        self.phi = mlp(
            cfg.latent_dim, cfg.hidden, cfg.phi_dim, activation=cfg.activation,
            final_activation=True,
        )
        n_stats = 2 if cfg.use_second_moment else 1
        head_in = (
            cfg.phi_dim * n_stats
            + (1 if cfg.use_size_feature else 0)
            + cfg.policy_dim
            + cfg.dataset_dim
        )
        self.head = mlp(
            head_in, cfg.hidden, 1, activation=cfg.activation, dropout=cfg.dropout
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        policy_context: torch.Tensor | None = None,
        dataset_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = self.phi(z)
        pooled = masked_mean(p, mask) if self.cfg.pool == "mean" else masked_sum(p, mask)
        parts = [pooled]
        if self.cfg.use_second_moment:
            second = (
                masked_mean(p * p, mask)
                if self.cfg.pool == "mean"
                else masked_sum(p * p, mask)
            )
            parts.append(second)
        if self.cfg.use_size_feature:
            parts.append(mask.sum(1, keepdim=True).to(p.dtype) / self.cfg.size_scale)
        parts += self._side_inputs(z.shape[0], policy_context, dataset_context, z.device, z.dtype)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


class AdditiveBatchUtility(_UtilityBase):
    """V = pool_i f(z_i) (+ side terms). Cannot express any interaction.

    Side inputs are added through a separate branch rather than concatenated
    before pooling, so the per-sample contribution stays strictly independent
    of the other members - otherwise the "additive" control would leak
    interactions through the head.
    """

    def __init__(self, cfg: BatchUtilityConfig):
        super().__init__(cfg)
        self.f = mlp(cfg.latent_dim, cfg.hidden, 1, activation=cfg.activation)
        side = cfg.policy_dim + cfg.dataset_dim
        self.side = (
            mlp(side, cfg.hidden, 1, activation=cfg.activation) if side > 0 else None
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        policy_context: torch.Tensor | None = None,
        dataset_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per = self.f(z).squeeze(-1) * mask.to(z.dtype)
        v = (
            per.sum(1) / mask.sum(1).clamp(min=1)
            if self.cfg.pool == "mean"
            else per.sum(1)
        )
        parts = self._side_inputs(z.shape[0], policy_context, dataset_context, z.device, z.dtype)
        if self.side is not None and parts:
            v = v + self.side(torch.cat(parts, dim=-1)).squeeze(-1)
        return v

    def per_sample_values(self, z: torch.Tensor) -> torch.Tensor:
        return self.f(z).squeeze(-1)


class PairwiseBatchUtility(_UtilityBase):
    """V = mean_i g(z_i) + mean_{i<j} k(z_i, z_j), with a symmetric kernel.

    Interpretable by construction: the pairwise term is exactly the redundancy /
    complementarity budget, and `pairwise_matrix` exposes it for analysis.
    """

    def __init__(self, cfg: BatchUtilityConfig):
        super().__init__(cfg)
        self.g = mlp(cfg.latent_dim, cfg.hidden, 1, activation=cfg.activation)
        self.pair = mlp(2 * cfg.latent_dim, cfg.hidden, 1, activation=cfg.activation)
        side = cfg.policy_dim + cfg.dataset_dim
        self.side = (
            mlp(side, cfg.hidden, 1, activation=cfg.activation) if side > 0 else None
        )

    def pairwise_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Symmetrized k(z_i, z_j) -> (B, N, N)."""
        n = z.shape[1]
        zi = z.unsqueeze(2).expand(-1, -1, n, -1)
        zj = z.unsqueeze(1).expand(-1, n, -1, -1)
        a = self.pair(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return 0.5 * (a + a.transpose(1, 2))

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        policy_context: torch.Tensor | None = None,
        dataset_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = mask.to(z.dtype)
        first = (self.g(z).squeeze(-1) * m).sum(1) / m.sum(1).clamp(min=1)
        k = self.pairwise_matrix(z)
        pm = m.unsqueeze(1) * m.unsqueeze(2)
        eye = torch.eye(z.shape[1], device=z.device, dtype=z.dtype).unsqueeze(0)
        pm = pm * (1 - eye)
        n_pairs = pm.sum((1, 2)).clamp(min=1)
        v = first + (k * pm).sum((1, 2)) / n_pairs
        parts = self._side_inputs(z.shape[0], policy_context, dataset_context, z.device, z.dtype)
        if self.side is not None and parts:
            v = v + self.side(torch.cat(parts, dim=-1)).squeeze(-1)
        return v


def build_batch_utility(cfg: BatchUtilityConfig, kind: str = "deepsets") -> nn.Module:
    if kind == "deepsets":
        return BatchUtilityModel(cfg)
    if kind == "additive":
        return AdditiveBatchUtility(cfg)
    if kind == "pairwise":
        return PairwiseBatchUtility(cfg)
    raise ValueError(f"unknown batch utility model {kind!r}")
