"""Latent clustering into local effect domains (PLAN.md 3.3, 7, 6).

A `ClusterState` carries everything the direction generator needs, so the
clustering method itself stays swappable - PLAN.md 7 is explicit that no
particular clustering algorithm is part of the contribution. Boundary points
matter most: outward expansion has to start from the edge of the support, not
from the centroid, or the proposed samples land inside data we already own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture


@dataclass
class ClusterState:
    cluster_id: int
    member_ids: np.ndarray
    centroid: np.ndarray
    covariance: np.ndarray
    pca_basis: np.ndarray  # (n_components, latent_dim), rows are eigenvectors
    pca_eigenvalues: np.ndarray
    boundary_ids: np.ndarray
    metadata_mean: np.ndarray | None = None
    metadata_std: np.ndarray | None = None
    metadata: np.ndarray | None = None  # raw metadata of members
    extra: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(len(self.member_ids))

    @property
    def radius(self) -> float:
        """Mean distance from the centroid; the natural scale for step sizes."""
        return float(self.extra.get("mean_radius", 0.0))

    def explained_variance_ratio(self) -> np.ndarray:
        total = self.pca_eigenvalues.sum()
        if total <= 0:
            return np.zeros_like(self.pca_eigenvalues)
        return self.pca_eigenvalues / total

    def intrinsic_rank(self, rho: float = 0.90, r_max: int = 5) -> int:
        """Smallest r with cumulative explained variance >= rho, capped at r_max."""
        cum = np.cumsum(self.explained_variance_ratio())
        r = int(np.searchsorted(cum, rho) + 1)
        return int(min(max(r, 1), r_max, len(self.pca_eigenvalues)))

    def summary(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "mean_radius": self.radius,
            "n_boundary": int(len(self.boundary_ids)),
            "explained_variance_ratio": self.explained_variance_ratio().tolist(),
            "intrinsic_rank_90": self.intrinsic_rank(),
            "metadata_mean": None if self.metadata_mean is None else self.metadata_mean.tolist(),
            "metadata_std": None if self.metadata_std is None else self.metadata_std.tolist(),
        }


@dataclass
class ClusteringConfig:
    method: str = "kmeans"  # "kmeans" | "gmm" | "hdbscan"
    n_clusters: int = 8
    #: fraction of each cluster's members, furthest from the centroid, kept as
    #: boundary anchors for outward expansion
    boundary_quantile: float = 0.25
    min_cluster_size: int = 10
    seed: int = 0


class LatentClustering:
    def __init__(self, cfg: ClusteringConfig | None = None):
        self.cfg = cfg or ClusteringConfig()
        self.labels_: np.ndarray | None = None
        self.clusters_: list[ClusterState] = []
        self.model_ = None

    def fit(self, z: np.ndarray, metadata: np.ndarray | None = None) -> list[ClusterState]:
        z = np.asarray(z, dtype=np.float64)
        cfg = self.cfg
        if cfg.method == "kmeans":
            self.model_ = KMeans(n_clusters=cfg.n_clusters, n_init=10, random_state=cfg.seed)
            labels = self.model_.fit_predict(z)
        elif cfg.method == "gmm":
            self.model_ = GaussianMixture(
                n_components=cfg.n_clusters, covariance_type="full", random_state=cfg.seed
            )
            labels = self.model_.fit_predict(z)
        elif cfg.method == "hdbscan":
            self.model_ = HDBSCAN(min_cluster_size=max(cfg.min_cluster_size, 2))
            labels = self.model_.fit_predict(z)
        else:
            raise ValueError(f"unknown clustering method {cfg.method!r}")

        self.labels_ = labels
        self.clusters_ = []
        # label -1 is HDBSCAN's noise class and is intentionally not a domain
        for cid in sorted(int(c) for c in np.unique(labels) if c >= 0):
            ids = np.nonzero(labels == cid)[0]
            if len(ids) < 2:
                continue
            self.clusters_.append(self._build_state(cid, ids, z, metadata))
        return self.clusters_

    def _build_state(
        self, cid: int, ids: np.ndarray, z: np.ndarray, metadata: np.ndarray | None
    ) -> ClusterState:
        pts = z[ids]
        centroid = pts.mean(0)
        cov = np.atleast_2d(np.cov(pts - centroid, rowvar=False))
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]

        d = np.linalg.norm(pts - centroid, axis=1)
        q = np.quantile(d, 1.0 - self.cfg.boundary_quantile) if len(d) > 1 else 0.0
        boundary = ids[d >= q]
        if len(boundary) == 0:
            boundary = ids[[int(np.argmax(d))]]

        meta_mean = meta_std = meta_raw = None
        if metadata is not None:
            meta_raw = np.asarray(metadata)[ids]
            meta_mean, meta_std = meta_raw.mean(0), meta_raw.std(0)

        return ClusterState(
            cluster_id=cid,
            member_ids=ids,
            centroid=centroid,
            covariance=cov,
            pca_basis=evecs.T,
            pca_eigenvalues=evals,
            boundary_ids=boundary,
            metadata_mean=meta_mean,
            metadata_std=meta_std,
            metadata=meta_raw,
            extra={
                "mean_radius": float(d.mean()),
                "max_radius": float(d.max()),
                "boundary_threshold": float(q),
            },
        )

    def report(self) -> dict:
        sizes = [c.size for c in self.clusters_]
        return {
            "method": self.cfg.method,
            "n_clusters": len(self.clusters_),
            "sizes": sizes,
            "size_min": int(min(sizes)) if sizes else 0,
            "size_max": int(max(sizes)) if sizes else 0,
            "n_noise": int((self.labels_ == -1).sum()) if self.labels_ is not None else 0,
            "clusters": [c.summary() for c in self.clusters_],
        }


def support_density(
    z_query: np.ndarray, z_support: np.ndarray, k: int = 10
) -> np.ndarray:
    """Local density proxy: inverse mean distance to the k nearest support points.

    Used by the outward filter - a proposed point is only an *expansion* if the
    support density there is lower than at the anchor it came from.
    """
    from sklearn.neighbors import NearestNeighbors

    z_query = np.atleast_2d(np.asarray(z_query, dtype=np.float64))
    z_support = np.asarray(z_support, dtype=np.float64)
    k = min(k, len(z_support))
    nn = NearestNeighbors(n_neighbors=k).fit(z_support)
    d, _ = nn.kneighbors(z_query)
    return 1.0 / (d.mean(axis=1) + 1e-12)


def distance_to_support(z_query: np.ndarray, z_support: np.ndarray) -> np.ndarray:
    """Distance to the single nearest support point (the trust-region measure)."""
    from sklearn.neighbors import NearestNeighbors

    z_query = np.atleast_2d(np.asarray(z_query, dtype=np.float64))
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(z_support, dtype=np.float64))
    d, _ = nn.kneighbors(z_query)
    return d[:, 0]
