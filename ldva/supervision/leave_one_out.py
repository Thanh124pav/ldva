"""Leave-one-out effect targets (SETUP.md 12, "expensive"; PLAN.md 3.1).

The preferred definition:

    Delta_i(B, theta) = U(Update(theta, B)) - U(Update(theta, B \\ {i}))

This is the only target that is genuinely *contextual* - remove the same sample
from a different batch and you get a different number, because the remaining
batch members change the update. It is also `len(B) + 1` updates per context,
hence "use expensive targets on a subset for calibration" (SETUP.md 12).

`ShortHorizonRetrainEstimator` is the same idea with `n_steps` updates instead
of one, which captures a little of the longer-horizon effect at a proportional
cost.
"""

from __future__ import annotations

import numpy as np
import torch

from ldva.supervision.base import EffectEstimator, SupervisionTask, flat_grad


class LeaveOneOutEstimator(EffectEstimator):
    estimator_id = "leave_one_out"

    def __init__(self, lr: float = 0.1, n_steps: int = 1):
        self.lr = float(lr)
        self.n_steps = int(n_steps)

    def _updated_utility(
        self, task: SupervisionTask, ids: np.ndarray
    ) -> float:
        """U after `n_steps` full-batch gradient steps on `ids`."""
        if len(ids) == 0:
            return task.utility()
        params = list(task.parameters())
        current = [p.detach().clone() for p in params]
        for _ in range(self.n_steps):
            g = _grad_at(task, current, ids)
            current = apply_update_from(current, g, self.lr)
        return task.utility(current)

    def sample_effect(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(batch_sample_ids, dtype=np.int64)
        u_full = self._updated_utility(task, ids)
        out = np.empty(len(ids), dtype=np.float64)
        for k in range(len(ids)):
            without = np.delete(ids, k)
            out[k] = u_full - self._updated_utility(task, without)
        return out

    def batch_gain(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> float:
        ids = np.asarray(batch_sample_ids, dtype=np.int64)
        return float(self._updated_utility(task, ids) - task.utility())

    @property
    def cost_tier(self) -> str:
        return "expensive"


class ShortHorizonRetrainEstimator(LeaveOneOutEstimator):
    """Leave-one-out with a multi-step retraining horizon."""

    estimator_id = "short_horizon_retrain"

    def __init__(self, lr: float = 0.1, n_steps: int = 5):
        super().__init__(lr=lr, n_steps=n_steps)


def _grad_at(
    task: SupervisionTask, params: list[torch.Tensor], ids: np.ndarray
) -> torch.Tensor:
    """Gradient of the training loss on `ids`, evaluated at `params`.

    Implemented by temporarily writing `params` into the task so that tasks
    which are not written functionally still work.
    """
    live = list(task.parameters())
    backup = [p.detach().clone() for p in live]
    try:
        with torch.no_grad():
            for p, new in zip(live, params):
                p.copy_(new)
        return flat_grad(task.train_loss(ids), live).detach()
    finally:
        with torch.no_grad():
            for p, old in zip(live, backup):
                p.copy_(old)


def apply_update_from(
    params: list[torch.Tensor], direction: torch.Tensor, lr: float
) -> list[torch.Tensor]:
    out, off = [], 0
    for p in params:
        k = p.numel()
        out.append((p.reshape(-1) - lr * direction[off : off + k]).view_as(p))
        off += k
    return out
