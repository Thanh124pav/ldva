"""Cheap effect targets (PLAN.md 3.2, "cheap").

Three proxies, all first order and all computable from one backward pass per
sample:

- `cosine`    : cos(g_i, g_val)      - direction only, scale free
- `dot`       : lr * <g_i, g_val>    - first-order predicted *rise* in utility
- `grad_norm` : ||g_i||              - a context-free magnitude baseline

Sign: a step along `-g_B` changes utility by `-g_val . (-lr g_B) = +lr <g_B, g_val>`
(utility is minus the validation loss), so aligned gradients give a positive gain.

Only the first two are contextual in the weak sense that they depend on the
checkpoint; none of them depend on the *other* samples in the batch. That is
exactly why they are cheap and why PLAN.md 19/F3 asks us to compare them with
the leave-one-out oracle before trusting them.
"""

from __future__ import annotations

import numpy as np
import torch

from ldva.supervision.base import (
    EffectEstimator,
    SupervisionTask,
    apply_update,
    flat_grad,
    per_sample_grads,
)


class GradientAlignmentEstimator(EffectEstimator):
    estimator_id = "gradient_alignment"

    def __init__(self, mode: str = "cosine", lr: float = 0.1, eps: float = 1e-12):
        if mode not in ("cosine", "dot", "grad_norm"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.lr = float(lr)
        self.eps = float(eps)
        self.estimator_id = f"gradient_alignment:{mode}"

    def _val_grad(self, task: SupervisionTask) -> torch.Tensor:
        params = list(task.parameters())
        return flat_grad(task.val_loss(), params).detach()

    def sample_effect(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> np.ndarray:
        g = per_sample_grads(task, batch_sample_ids)  # (n, d)
        if self.mode == "grad_norm":
            return g.norm(dim=1).cpu().numpy()
        gv = self._val_grad(task)
        if self.mode == "cosine":
            num = g @ gv
            den = g.norm(dim=1) * gv.norm() + self.eps
            return (num / den).cpu().numpy()
        return ((g @ gv) * self.lr).cpu().numpy()

    def batch_gain(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> float:
        """First-order predicted utility change from one SGD step on the batch."""
        params = list(task.parameters())
        gb = flat_grad(task.train_loss(batch_sample_ids), params).detach()
        gv = self._val_grad(task)
        return float((gb @ gv).item() * self.lr)

    @property
    def cost_tier(self) -> str:
        return "cheap"


class ValidationGradientAlignmentEstimator(GradientAlignmentEstimator):
    """Alias kept explicit because PLAN.md 3.2 lists it as its own target."""

    def __init__(self, lr: float = 0.1):
        super().__init__(mode="dot", lr=lr)
        self.estimator_id = "validation_gradient_alignment"


class OneStepUtilityEstimator(EffectEstimator):
    """Actually take the step and measure U, instead of trusting first order.

    Sits between the gradient proxies and full leave-one-out: one update per
    sample, no retraining.
    """

    estimator_id = "one_step_utility"

    def __init__(self, lr: float = 0.1):
        self.lr = float(lr)

    def sample_effect(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> np.ndarray:
        params = list(task.parameters())
        u0 = task.utility()
        out = []
        for sid in np.asarray(batch_sample_ids, dtype=np.int64):
            g = flat_grad(task.train_loss(np.array([sid])), params).detach()
            out.append(task.utility(apply_update(params, g, self.lr)) - u0)
        return np.array(out, dtype=np.float64)

    def batch_gain(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> float:
        params = list(task.parameters())
        g = flat_grad(task.train_loss(batch_sample_ids), params).detach()
        return float(task.utility(apply_update(params, g, self.lr)) - task.utility())

    @property
    def cost_tier(self) -> str:
        return "medium"
