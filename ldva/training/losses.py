"""Training objectives (PLAN.md 5).

    L = L_effect + lb * L_batch + lm * L_metric + ls * L_smooth [+ lmeta * L_meta]

Notes on two subtleties:

- `L_metric` compares a latent distance measured *at one checkpoint* against an
  effect distance averaged over all contexts the pair shares. The mismatch is
  deliberate and documented in PLAN.md 5.3: the target is a summary of
  optimization behaviour, not a per-checkpoint quantity. Pairs always exist for
  members of the same context, since the table is built from co-occurrences.

- `L_smooth` is a Lipschitz-style penalty, not an equality constraint. It asks
  that a *small* change of batch and policy context produce a small change of
  predicted effect; forcing it to zero would destroy the contextuality the rest
  of the objective is trying to create, which is why PLAN.md 5.4 warns against
  over-weighting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    effect: float = 1.0
    batch: float = 1.0
    metric: float = 0.1
    smooth: float = 0.01
    meta: float = 0.0

    def as_dict(self) -> dict:
        return {
            "effect": self.effect,
            "batch": self.batch,
            "metric": self.metric,
            "smooth": self.smooth,
            "meta": self.meta,
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype)
    n = m.sum().clamp(min=1.0)
    return (((pred - target) ** 2) * m).sum() / n


def effect_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L_effect: MSE of contextual effect predictions over valid members."""
    return masked_mse(pred, target, mask)


def batch_gain_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_batch: MSE of batch-gain predictions."""
    return F.mse_loss(pred, target)


def metric_loss(
    z: torch.Tensor,
    sample_ids: torch.Tensor,
    mask: torch.Tensor,
    distance_lookup,
    max_pairs_per_context: int = 28,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """L_metric = mean | ||z_i - z_j|| - d_effect_norm(i,j) |.

    Returns `(loss, n_pairs_used)`; the count lets the trainer skip the term
    cleanly when a minibatch yields no usable pair instead of silently adding a
    zero that drags the running average down.
    """
    rng = rng or np.random.default_rng(0)
    n_ctx, n_max = mask.shape
    zi_idx, zj_idx, ctx_idx = [], [], []
    ids_np = sample_ids.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    pair_keys = []

    for c in range(n_ctx):
        members = np.nonzero(mask_np[c])[0]
        if len(members) < 2:
            continue
        a, b = np.triu_indices(len(members), k=1)
        if len(a) > max_pairs_per_context:
            sel = rng.choice(len(a), size=max_pairs_per_context, replace=False)
            a, b = a[sel], b[sel]
        for ia, ib in zip(a, b):
            mi, mj = int(members[ia]), int(members[ib])
            pair_keys.append((int(ids_np[c, mi]), int(ids_np[c, mj])))
            zi_idx.append(mi)
            zj_idx.append(mj)
            ctx_idx.append(c)

    if not pair_keys:
        return z.sum() * 0.0, 0

    target, found = distance_lookup(np.array(pair_keys, dtype=np.int64))
    if not found.any():
        return z.sum() * 0.0, 0

    ctx_idx = torch.as_tensor(np.array(ctx_idx)[found], device=z.device)
    zi_idx = torch.as_tensor(np.array(zi_idx)[found], device=z.device)
    zj_idx = torch.as_tensor(np.array(zj_idx)[found], device=z.device)
    t = torch.as_tensor(target[found], dtype=z.dtype, device=z.device)

    d = torch.norm(z[ctx_idx, zi_idx] - z[ctx_idx, zj_idx], dim=-1)
    return torch.mean(torch.abs(d - t)), int(found.sum())


def smoothness_loss(
    effect_pred: torch.Tensor,
    effect_pred_perturbed: torch.Tensor,
    shared_mask: torch.Tensor,
) -> torch.Tensor:
    """L_smooth: |s_hat(B, theta) - s_hat(B', theta')| on members present in both."""
    m = shared_mask.to(effect_pred.dtype)
    n = m.sum().clamp(min=1.0)
    return (torch.abs(effect_pred - effect_pred_perturbed) * m).sum() / n


def metadata_direction_loss(
    model,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    m_i: torch.Tensor,
    m_j: torch.Tensor,
) -> torch.Tensor:
    """L_meta = || (z_j - z_i) - G_omega(z_i, m_i, m_j - m_i) ||^2 (PLAN.md 5.5)."""
    delta_m = m_j - m_i
    pred = model(z_i, m_i, delta_m)
    return torch.mean(torch.sum((z_j - z_i - pred) ** 2, dim=-1))


def latent_norm_penalty(z: torch.Tensor, mask: torch.Tensor, target: float = 0.0) -> torch.Tensor:
    """Optional guard against latent-scale blow-up when metric loss is off."""
    m = mask.to(z.dtype).unsqueeze(-1)
    norms = torch.norm(z * m, dim=-1)
    if target <= 0:
        return (norms**2).sum() / m.sum().clamp(min=1.0)
    return ((norms - target) ** 2).sum() / m.sum().clamp(min=1.0)
