"""Candidate acquisition directions (PLAN.md 8, 7).

For each local effect domain we take the local PCA basis, keep the smallest rank
explaining `rho` of the variance (capped at `r_max`), and propose both signs of
each retained eigenvector. A direction then has to survive three filters before
it becomes a candidate:

1. **outward** - moving along it increases the distance to the cluster centroid;
2. **density decreasing** - the proposal sits in sparser support than its anchor,
   so it genuinely expands coverage instead of thickening what we already own;
3. **trust region** - two caps, both active, because neither alone is enough
   (PLAN.md 19/F5, PLAN.md 7):
   - *relative*: the nearest observed sample is within
     `epsilon_expand_scale * delta`. Stepping `delta` beyond a boundary anchor
     normally leaves you about `delta` from the support, so a much larger
     distance means the direction points sideways into a void.
   - *absolute*: the nearest observed sample is within
     `epsilon_expand_absolute_scale * cluster_radius`. The relative cap scales
     with `delta` and so can never reject a merely enormous step; this one is
     what actually forbids uncontrolled long-range extrapolation.

Step sizes are expressed in units of the cluster radius, so one `delta` setting
behaves consistently across domains of different scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ldva.acquisition.clustering import ClusterState, distance_to_support, support_density


@dataclass
class AcquisitionDirection:
    """One candidate direction `a = (cluster k, direction v)` (PLAN.md 9)."""

    direction_id: int
    cluster_id: int
    #: unit vector in latent space
    vector: np.ndarray
    #: index of the PCA component it came from (-1 for random/other sources)
    component: int
    sign: int
    #: latent anchors on the cluster boundary that this direction starts from
    anchor_ids: np.ndarray
    anchors: np.ndarray
    #: absolute step length in latent units
    delta: float
    eigenvalue: float = 0.0
    explained_variance: float = 0.0
    #: filter diagnostics
    outward_score: float = 0.0
    density_ratio: float = 1.0
    support_distance: float = 0.0
    #: monetary cost per acquired trajectory along this direction (PLAN.md 10)
    cost: float = 1.0
    source: str = "pca"
    extra: dict = field(default_factory=dict)

    def proposal_center(self) -> np.ndarray:
        """Mean proposed latent position: anchor mean + delta * v."""
        return self.anchors.mean(0) + self.delta * self.vector

    def summary(self) -> dict:
        return {
            "direction_id": self.direction_id,
            "cluster_id": self.cluster_id,
            "component": self.component,
            "sign": self.sign,
            "delta": self.delta,
            "explained_variance": self.explained_variance,
            "outward_score": self.outward_score,
            "density_ratio": self.density_ratio,
            "support_distance": self.support_distance,
            "cost": self.cost,
            "source": self.source,
            "n_anchors": int(len(self.anchor_ids)),
        }


@dataclass
class DirectionConfig:
    rho: float = 0.90
    r_max: int = 5
    #: Step length as a multiple of the cluster's mean radius. Smaller steps
    #: keep the local linear metadata map valid: on the Stage 0 world the
    #: realized direction cosine is 0.33 at 0.4 but only 0.18 at 1.5. The
    #: trade-off is that a smaller step expands the support less, so this is a
    #: fidelity-versus-reach knob, not a free parameter.
    delta_scale: float = 0.4
    n_anchors: int = 5
    #: trust region: max distance to the support, as a multiple of the step `delta`
    epsilon_expand_scale: float = 1.5
    #: absolute cap, as a multiple of the cluster radius; `None` disables it
    epsilon_expand_absolute_scale: float | None = 2.0
    density_k: int = 10
    #: filters
    require_outward: bool = True
    require_density_decrease: bool = True
    require_trust_region: bool = True
    #: Reject a candidate whose |cos| with an already-accepted direction
    #: exceeds this. Without it the candidate set is redundant, and redundancy
    #: caps the only interpretable measure of direction control.
    #:
    #: Measured before this filter existed: 12-15 candidates spanned ~2.5
    #: effective dimensions with a mean pairwise |cos| of ~0.5, so many
    #: candidates were near duplicates. Specificity - does collection move the
    #: latents along the direction asked for rather than one that was not -
    #: then cannot clear 2 standard deviations even with perfect execution,
    #: because the null is full of near-copies of the direction being tested.
    #: In raw metadata space the realized cosine was 1.000 and the z-score
    #: still only 1.79-1.91. `None` disables the filter.
    max_pairwise_cosine: float | None = 0.8
    #: ablation 6 of PLAN.md 18: random directions instead of local PCA
    use_random_directions: bool = False
    n_random_per_cluster: int = 4
    seed: int = 0


class DirectionGenerator:
    def __init__(self, cfg: DirectionConfig | None = None):
        self.cfg = cfg or DirectionConfig()
        self.rejected_: list[dict] = []

    def generate(
        self,
        clusters: list[ClusterState],
        z_support: np.ndarray,
        costs: dict[int, float] | None = None,
    ) -> list[AcquisitionDirection]:
        """Produce the filtered candidate set across all domains."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        z_support = np.asarray(z_support, dtype=np.float64)
        # global spacing is only used for the optional absolute cap
        support_scale = _support_scale(z_support)

        self.rejected_ = []
        out: list[AcquisitionDirection] = []
        did = 0
        for c in clusters:
            vectors = (
                self._random_vectors(c, rng)
                if cfg.use_random_directions
                else self._pca_vectors(c)
            )
            delta = cfg.delta_scale * max(c.radius, 1e-8)
            for comp, sign, v, ev, evr in vectors:
                # anchors are chosen *per direction*: the boundary points that
                # are extreme along v. Picking them once per cluster (globally
                # furthest from the centroid) would straddle both ends of an
                # elongated domain, so +v and -v would each average to zero
                # outward movement and both would be rejected.
                anchors_ids = self._anchor_ids(c, z_support, v)
                anchors = z_support[anchors_ids]
                cand = AcquisitionDirection(
                    direction_id=did,
                    cluster_id=c.cluster_id,
                    vector=v,
                    component=comp,
                    sign=sign,
                    anchor_ids=anchors_ids,
                    anchors=anchors,
                    delta=delta,
                    eigenvalue=ev,
                    explained_variance=evr,
                    cost=float((costs or {}).get(c.cluster_id, 1.0)),
                    source="random" if cfg.use_random_directions else "pca",
                )
                verdict = self._evaluate(cand, c, z_support, support_scale)
                if verdict["keep"] and cfg.max_pairwise_cosine is not None:
                    dup = self._too_similar(cand, out, cfg.max_pairwise_cosine)
                    if dup is not None:
                        verdict = {"keep": False,
                                   "reason": f"duplicate_of_direction_{dup}"}
                if verdict["keep"]:
                    out.append(cand)
                    did += 1
                else:
                    self.rejected_.append({**cand.summary(), "reason": verdict["reason"]})
        return out

    @staticmethod
    def _too_similar(cand, accepted: list, max_cos: float) -> int | None:
        """Direction id of an accepted candidate this one nearly duplicates.

        Compared across clusters as well as within one: two domains sitting on
        the same elongated manifold produce near-identical leading PCA
        directions, which is where most of the redundancy came from.

        The comparison is **signed**, deliberately. `+v` and `-v` have
        |cos| = 1 but are opposite requests - expand outward on one side of a
        domain or the other - and both are meaningful, so an absolute-value
        test would delete half of every candidate set. Only a candidate
        pointing the *same* way as an accepted one is a duplicate.
        """
        v = np.asarray(cand.vector, dtype=np.float64)
        v = v / max(np.linalg.norm(v), 1e-12)
        for other in accepted:
            w = np.asarray(other.vector, dtype=np.float64)
            w = w / max(np.linalg.norm(w), 1e-12)
            if float(np.dot(v, w)) > max_cos:
                return int(other.direction_id)
        return None

    # ---- direction proposals -------------------------------------------
    def _pca_vectors(self, c: ClusterState):
        r = c.intrinsic_rank(self.cfg.rho, self.cfg.r_max)
        evr = c.explained_variance_ratio()
        for j in range(r):
            v = c.pca_basis[j]
            v = v / max(np.linalg.norm(v), 1e-12)
            for sign in (1, -1):
                yield j, sign, sign * v, float(c.pca_eigenvalues[j]), float(evr[j])

    def _random_vectors(self, c: ClusterState, rng: np.random.Generator):
        d = c.centroid.shape[0]
        for j in range(self.cfg.n_random_per_cluster):
            v = rng.normal(size=d)
            v /= max(np.linalg.norm(v), 1e-12)
            yield -1, 1, v, 0.0, 0.0

    def _anchor_ids(
        self, c: ClusterState, z_support: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Boundary anchors furthest along `v` (PLAN.md 8.2).

        Ranking by projection onto `v` rather than by distance from the centroid
        is what makes `z_b + delta * v` an actual expansion of the support: the
        step starts from the edge of the domain *in the direction we intend to
        move*.
        """
        pool = c.boundary_ids if len(c.boundary_ids) >= self.cfg.n_anchors else c.member_ids
        if len(pool) == 0:
            return c.member_ids[:1]
        proj = (z_support[pool] - c.centroid) @ v
        n = min(self.cfg.n_anchors, len(pool))
        top = np.argsort(proj)[::-1][:n]
        return np.asarray(pool)[top]

    # ---- filters ---------------------------------------------------------
    def _evaluate(
        self,
        cand: AcquisitionDirection,
        c: ClusterState,
        z_support: np.ndarray,
        support_scale: float,
    ) -> dict:
        cfg = self.cfg
        proposals = cand.anchors + cand.delta * cand.vector

        # 1. outward: does the step move away from the centroid?
        d_before = np.linalg.norm(cand.anchors - c.centroid, axis=1).mean()
        d_after = np.linalg.norm(proposals - c.centroid, axis=1).mean()
        cand.outward_score = float(d_after - d_before)
        if cfg.require_outward and cand.outward_score <= 0:
            return {"keep": False, "reason": "not_outward"}

        # 2. density: is the proposed region sparser than the anchor region?
        dens_anchor = support_density(cand.anchors, z_support, k=cfg.density_k).mean()
        dens_prop = support_density(proposals, z_support, k=cfg.density_k).mean()
        cand.density_ratio = float(dens_prop / max(dens_anchor, 1e-12))
        if cfg.require_density_decrease and cand.density_ratio >= 1.0:
            return {"keep": False, "reason": "density_not_decreasing"}

        # 3. trust region: close enough to the support to be predictable
        cand.support_distance = float(distance_to_support(proposals, z_support).mean())
        eps_rel = cfg.epsilon_expand_scale * max(cand.delta, 1e-12)
        eps_abs = (
            cfg.epsilon_expand_absolute_scale * max(c.radius, 1e-12)
            if cfg.epsilon_expand_absolute_scale is not None
            else np.inf
        )
        eps_expand = min(eps_rel, eps_abs)
        trust_ratio = cand.support_distance / max(cand.delta, 1e-12)
        cand.extra.update(
            {
                "eps_expand": float(eps_expand),
                "eps_relative": float(eps_rel),
                "eps_absolute": float(eps_abs),
                "trust_ratio": float(trust_ratio),
                "radius_ratio": float(cand.support_distance / max(c.radius, 1e-12)),
                "support_scale": float(support_scale),
                "centroid_distance_before": float(d_before),
                "centroid_distance_after": float(d_after),
            }
        )
        if cfg.require_trust_region and cand.support_distance > eps_expand:
            reason = (
                "outside_trust_region_absolute"
                if cand.support_distance > eps_abs
                else "outside_trust_region_relative"
            )
            return {"keep": False, "reason": reason}
        return {"keep": True, "reason": "ok"}

    def report(self) -> dict:
        reasons: dict[str, int] = {}
        for r in self.rejected_:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        return {"n_rejected": len(self.rejected_), "rejection_reasons": reasons}


def _support_scale(z: np.ndarray, k: int = 5) -> float:
    """Typical nearest-neighbour spacing, used as the trust-region unit."""
    from sklearn.neighbors import NearestNeighbors

    if len(z) < 2:
        return 1.0
    k = min(k + 1, len(z))
    nn = NearestNeighbors(n_neighbors=k).fit(z)
    d, _ = nn.kneighbors(z)
    return float(np.median(d[:, 1:].mean(axis=1)))


def directions_report(directions: list[AcquisitionDirection]) -> dict:
    if not directions:
        return {"n_directions": 0}
    per_cluster: dict[int, int] = {}
    for d in directions:
        per_cluster[d.cluster_id] = per_cluster.get(d.cluster_id, 0) + 1
    return {
        "n_directions": len(directions),
        "directions_per_cluster": per_cluster,
        "mean_delta": float(np.mean([d.delta for d in directions])),
        "mean_support_distance": float(np.mean([d.support_distance for d in directions])),
        "mean_density_ratio": float(np.mean([d.density_ratio for d in directions])),
        "mean_trust_ratio": float(
            np.mean([d.extra.get("trust_ratio", float("nan")) for d in directions])
        ),
        "costs": [d.cost for d in directions],
    }
