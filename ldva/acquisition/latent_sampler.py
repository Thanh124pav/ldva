"""Hypothetical future latents (PLAN.md 9, 8).

    z_new = z_boundary + delta * v + epsilon,   epsilon ~ N(0, sigma^2 Sigma_local)

The noise term is the point of this module. A direction represents data we have
not collected yet, so representing it with a single deterministic latent point
would make the planner confident about a sample it has never seen. Drawing the
residual from the *local* covariance keeps the uncertainty anisotropic in the
same way the observed cluster is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ldva.acquisition.clustering import ClusterState
from ldva.acquisition.directions import AcquisitionDirection


@dataclass
class LatentSamplerConfig:
    #: scale of the local residual noise, relative to the cluster covariance
    sigma: float = 0.3
    #: jitter on the step length, so a direction is a cone rather than a ray
    delta_jitter: float = 0.15
    #: shrink the local covariance to its diagonal by this amount (0 = full)
    diagonal_shrinkage: float = 0.0
    seed: int = 0


class LatentSampler:
    """Draws hypothetical latents for candidate directions."""

    def __init__(
        self,
        clusters: list[ClusterState],
        cfg: LatentSamplerConfig | None = None,
    ):
        self.cfg = cfg or LatentSamplerConfig()
        self.clusters = {c.cluster_id: c for c in clusters}
        self._chol: dict[int, np.ndarray] = {}
        self.rng = np.random.default_rng(self.cfg.seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _cholesky(self, cluster_id: int) -> np.ndarray:
        """Cached Cholesky factor of the (regularized) local covariance."""
        if cluster_id in self._chol:
            return self._chol[cluster_id]
        cov = np.atleast_2d(self.clusters[cluster_id].covariance).copy()
        d = cov.shape[0]
        if self.cfg.diagonal_shrinkage > 0:
            s = self.cfg.diagonal_shrinkage
            cov = (1 - s) * cov + s * np.diag(np.diag(cov))
        # jitter until positive definite; clusters can be rank deficient
        scale = max(np.trace(cov) / max(d, 1), 1e-12)
        for extra in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            try:
                L = np.linalg.cholesky(cov + extra * scale * np.eye(d))
                self._chol[cluster_id] = L
                return L
            except np.linalg.LinAlgError:
                continue
        L = np.eye(d) * np.sqrt(scale)
        self._chol[cluster_id] = L
        return L

    def sample(self, direction: AcquisitionDirection, n: int) -> np.ndarray:
        """Draw `n` hypothetical latents for one direction -> (n, latent_dim)."""
        if n <= 0:
            return np.zeros((0, direction.vector.shape[0]))
        anchors = direction.anchors
        idx = self.rng.integers(0, len(anchors), size=n)
        base = anchors[idx]
        delta = direction.delta * (
            1.0 + self.cfg.delta_jitter * self.rng.normal(size=(n, 1))
        )
        L = self._cholesky(direction.cluster_id)
        eps = self.rng.normal(size=(n, L.shape[0])) @ L.T * self.cfg.sigma
        return base + delta * direction.vector[None, :] + eps

    def sample_allocation(
        self, directions: list[AcquisitionDirection], allocation: np.ndarray
    ) -> np.ndarray:
        """Draw one hypothetical acquisition batch for an allocation `n`.

        Returns the concatenated latents of the whole composition, which is what
        the batch utility model must score jointly (PLAN.md 8).
        """
        allocation = np.asarray(allocation, dtype=np.int64)
        if len(allocation) != len(directions):
            raise ValueError(
                f"allocation has {len(allocation)} entries but there are "
                f"{len(directions)} directions"
            )
        parts = [
            self.sample(d, int(k)) for d, k in zip(directions, allocation) if int(k) > 0
        ]
        if not parts:
            return np.zeros((0, directions[0].vector.shape[0]))
        return np.concatenate(parts, axis=0)
