"""Policy evaluation on a *fixed, pre-declared* distribution (SETUP.md 33)."""

from __future__ import annotations

import numpy as np
import torch

from ldva.policy.bc import MLPPolicy


@torch.no_grad()
def evaluate_bc(
    policy: MLPPolicy,
    val_obs: torch.Tensor,
    val_act: torch.Tensor,
    success_threshold: float = 0.1,
) -> dict:
    """Validation BC loss plus a thresholded success proxy.

    In simulation the real metric is rollout success; for the state-based linear
    stages we report per-chunk action error and the fraction of chunks whose mean
    error is below `success_threshold`, which behaves like a success rate and
    keeps the acquisition plots comparable across stages.
    """
    device = next(policy.parameters()).device
    val_obs = val_obs.to(device)
    val_act = val_act.to(device)
    pred = policy(val_obs)
    err = ((pred - val_act) ** 2).mean(dim=tuple(range(1, val_act.ndim)))
    return {
        "val_loss": float(err.mean().item()),
        "val_loss_std": float(err.std().item()),
        "success_rate": float((err < success_threshold).float().mean().item()),
        "utility": float(-err.mean().item()),
        "n_eval": int(err.numel()),
    }


def utility_from_flat_params(
    policy: MLPPolicy,
    flat: np.ndarray,
    val_obs: torch.Tensor,
    val_act: torch.Tensor,
) -> float:
    """Utility of a parameter vector without disturbing `policy`."""
    backup = policy.flat_params().clone()
    try:
        policy.load_flat_params(flat)
        return evaluate_bc(policy, val_obs, val_act)["utility"]
    finally:
        policy.load_flat_params(backup)
