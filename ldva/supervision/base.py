"""Effect-label interface (PLAN.md 3.2, 3.1).

Everything downstream consumes `ContextRecord`s, so the only thing an
environment has to provide is a `SupervisionTask`: how to get the current
parameters, a training loss on an arbitrary subset of samples, and a scalar
downstream utility. Gradient alignment, influence and leave-one-out are then
written once against that interface and reused from the synthetic sanity test
all the way to a real policy.

Sign convention: effects and gains are **gains** - larger is better. A task
whose `utility` is a loss should return its negation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch


class SupervisionTask(Protocol):
    """A differentiable learner we can probe for data effects."""

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        """Trainable parameters of the current checkpoint."""
        ...

    def train_loss(self, sample_ids: np.ndarray) -> torch.Tensor:
        """Mean training loss over `sample_ids` (differentiable)."""
        ...

    def utility(self, params: Sequence[torch.Tensor] | None = None) -> float:
        """Downstream utility U(theta); larger is better.

        `params` lets a caller evaluate a hypothetical parameter vector without
        mutating the task.
        """
        ...

    def val_loss(self, params: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        """Differentiable validation loss, used for validation-gradient targets."""
        ...

    @property
    def checkpoint_id(self) -> str:
        ...

    def policy_features(self) -> np.ndarray:
        """Numeric descriptors of this checkpoint (step, loss, grad norm, ...)."""
        ...


@dataclass
class EffectLabels:
    """Labels for one context."""

    per_sample_effects: np.ndarray
    batch_gain: float
    estimator_id: str
    diagnostics: dict | None = None


class EffectEstimator(ABC):
    """Base class for all effect-label generators."""

    #: short identifier recorded on every `ContextRecord`
    estimator_id: str = "base"

    @abstractmethod
    def sample_effect(
        self, task: SupervisionTask, batch_sample_ids: np.ndarray
    ) -> np.ndarray:
        """Per-sample contextual effects s_i(B, theta) for one batch."""

    @abstractmethod
    def batch_gain(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> float:
        """Batch-level gain target, e.g. U(Update(theta, B)) - U(theta)."""

    def label(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> EffectLabels:
        ids = np.asarray(batch_sample_ids, dtype=np.int64)
        return EffectLabels(
            per_sample_effects=np.asarray(self.sample_effect(task, ids), dtype=np.float64),
            batch_gain=float(self.batch_gain(task, ids)),
            estimator_id=self.estimator_id,
        )

    @property
    def cost_tier(self) -> str:
        """"cheap" | "medium" | "expensive" - used to budget label generation."""
        return "cheap"


# ---- shared autograd helpers ------------------------------------------


def flat_grad(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """Flattened gradient of `loss` wrt `params`, zeros for unused params."""
    grads = torch.autograd.grad(
        loss,
        list(params),
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    flat = [
        torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
        for p, g in zip(params, grads)
    ]
    return torch.cat(flat)


def per_sample_grads(
    task: SupervisionTask, sample_ids: np.ndarray
) -> torch.Tensor:
    """(n, n_params) matrix of per-sample loss gradients.

    Looping is intentional: it keeps the estimator usable for any task that can
    only score one sample at a time. `torch.func.vmap` + `grad` would be faster
    but requires a functional task definition.
    """
    params = list(task.parameters())
    rows = []
    for sid in np.asarray(sample_ids, dtype=np.int64):
        loss = task.train_loss(np.array([sid]))
        rows.append(flat_grad(loss, params).detach())
    return torch.stack(rows)


def apply_update(
    params: Sequence[torch.nn.Parameter], direction: torch.Tensor, lr: float
) -> list[torch.Tensor]:
    """theta - lr * direction, returned as a list shaped like `params`."""
    out, off = [], 0
    for p in params:
        k = p.numel()
        out.append((p.detach().reshape(-1) - lr * direction[off : off + k]).view_as(p))
        off += k
    return out
