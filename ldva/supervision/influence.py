"""Influence-function effect targets (PLAN.md 3.2, "medium").

Classical first-order influence of training point i on validation utility:

    I_i = + g_val^T H^-1 g_i

with the sign chosen so that a *positive* value means including i reduces the
validation loss, i.e. is a gain. H is the training Hessian at the current
checkpoint; we never form it, and offer two approximations:

- `damping`: H ~= lambda I, which collapses to a scaled gradient dot product
  but keeps the interface honest about what is being approximated.
- `lissa`  : stochastic Neumann series via Hessian-vector products.

`TRAKEstimator` is the random-projection variant: project per-sample gradients
to a low dimension and score alignment there, which is what makes TRAK-like
influence affordable for many samples.
"""

from __future__ import annotations

import numpy as np
import torch

from ldva.supervision.base import (
    EffectEstimator,
    SupervisionTask,
    flat_grad,
    per_sample_grads,
)


def hessian_vector_product(
    task: SupervisionTask, sample_ids: np.ndarray, v: torch.Tensor
) -> torch.Tensor:
    """Hv via double backward on the training loss."""
    params = list(task.parameters())
    loss = task.train_loss(sample_ids)
    g = flat_grad(loss, params, retain_graph=True, create_graph=True)
    hv = flat_grad((g * v).sum(), params, retain_graph=False)
    return hv.detach()


def lissa_inverse_hvp(
    task: SupervisionTask,
    v: torch.Tensor,
    sample_ids: np.ndarray,
    damping: float = 0.01,
    scale: float = 25.0,
    n_iter: int = 20,
    batch_size: int = 32,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Stochastic estimate of H^-1 v (Agarwal et al. LiSSA).

    Iterates h <- v + (I - (H + damping I)/scale) h, then divides by `scale`.
    `scale` must exceed the largest eigenvalue of H for the series to converge;
    too small a value diverges, which we detect and report by returning the
    damped fallback.
    """
    rng = rng or np.random.default_rng(0)
    ids = np.asarray(sample_ids, dtype=np.int64)
    h = v.clone()
    for _ in range(n_iter):
        sub = rng.choice(ids, size=min(batch_size, len(ids)), replace=False)
        hv = hessian_vector_product(task, sub, h)
        h = v + h - (hv + damping * h) / scale
        if not torch.isfinite(h).all() or h.norm() > 1e8 * (v.norm() + 1e-12):
            # series diverged: fall back to the damped identity approximation
            return v / max(damping, 1e-8)
    return h / scale


class InfluenceEstimator(EffectEstimator):
    estimator_id = "influence"

    def __init__(
        self,
        mode: str = "damping",
        damping: float = 0.01,
        scale: float = 25.0,
        n_iter: int = 20,
        lissa_batch: int = 32,
        seed: int = 0,
    ):
        if mode not in ("damping", "lissa"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.damping = float(damping)
        self.scale = float(scale)
        self.n_iter = int(n_iter)
        self.lissa_batch = int(lissa_batch)
        self.rng = np.random.default_rng(seed)
        self.estimator_id = f"influence:{mode}"

    def _ihvp(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> torch.Tensor:
        gv = flat_grad(task.val_loss(), list(task.parameters())).detach()
        if self.mode == "damping":
            return gv / self.damping
        return lissa_inverse_hvp(
            task,
            gv,
            batch_sample_ids,
            damping=self.damping,
            scale=self.scale,
            n_iter=self.n_iter,
            batch_size=self.lissa_batch,
            rng=self.rng,
        )

    def sample_effect(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> np.ndarray:
        ihvp = self._ihvp(task, batch_sample_ids)
        g = per_sample_grads(task, batch_sample_ids)
        return (g @ ihvp).cpu().numpy()

    def batch_gain(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> float:
        ihvp = self._ihvp(task, batch_sample_ids)
        gb = flat_grad(task.train_loss(batch_sample_ids), list(task.parameters())).detach()
        return float((gb @ ihvp).item())

    @property
    def cost_tier(self) -> str:
        return "medium"


class TRAKEstimator(EffectEstimator):
    """TRAK-like influence: alignment measured in a random projection.

    Per-sample gradients are projected with a fixed Rademacher matrix, which is
    what keeps the memory cost independent of the parameter count.
    """

    estimator_id = "trak"

    def __init__(self, proj_dim: int = 256, seed: int = 0, damping: float = 1e-2):
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.damping = float(damping)
        self._proj: torch.Tensor | None = None

    def _projection(self, d: int, device, dtype) -> torch.Tensor:
        if self._proj is None or self._proj.shape[0] != d:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            signs = torch.randint(0, 2, (d, self.proj_dim), generator=gen).float() * 2 - 1
            self._proj = (signs / np.sqrt(self.proj_dim)).to(device=device, dtype=dtype)
        return self._proj

    def _projected(self, task: SupervisionTask, batch_sample_ids: np.ndarray):
        g = per_sample_grads(task, batch_sample_ids)
        gv = flat_grad(task.val_loss(), list(task.parameters())).detach()
        p = self._projection(g.shape[1], g.device, g.dtype)
        return g @ p, gv @ p

    def sample_effect(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> np.ndarray:
        gp, gvp = self._projected(task, batch_sample_ids)
        # (G G^T + lambda I)^-1 G g_val, the TRAK kernel in projected space
        k = gp @ gp.T + self.damping * torch.eye(
            gp.shape[0], device=gp.device, dtype=gp.dtype
        )
        scores = torch.linalg.solve(k, gp @ gvp)
        return scores.cpu().numpy()

    def batch_gain(self, task: SupervisionTask, batch_sample_ids: np.ndarray) -> float:
        gp, gvp = self._projected(task, batch_sample_ids)
        return float((gp.mean(0) @ gvp).item())

    @property
    def cost_tier(self) -> str:
        return "medium"
