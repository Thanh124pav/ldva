"""Latent directions -> actionable metadata (PLAN.md 12; SETUP.md 17).

Without this module the planner produces latent directions nobody can collect.
We fit a **local** linear map per effect domain,

    delta_z ~= J_k delta_m

and invert it under the feasibility constraints of the acquisition interface:

    delta_m* = argmin || J_k delta_m - alpha v ||^2
               s.t.  m_anchor + delta_m in [low, high],  |delta_m| <= step cap

Locality is what makes this legitimate: PLAN.md 14/A2 only claims that *small*
controllable metadata changes produce predictable latent changes, never that the
global metadata->latent map is invertible. The fit therefore happens per cluster,
in normalized metadata units so the regression is not dominated by whichever
field happens to have the largest range.

The honest test of the whole module is `cosine(desired v, realized delta_z)`
after actually collecting at `m_anchor + delta_m*` - PLAN.md 24 and SETUP.md 22.
Only `evaluate_realized_direction` reports that, and it needs real new data.

Two things learned the hard way, both encoded in the defaults:

- Fit on *random* within-cluster pairs, not nearest-neighbour pairs. A pair
  difference has signal `J delta_m` and noise `2 * var(z | m)`, so taking the
  *nearest* neighbours drives `delta_m -> 0` and minimizes signal-to-noise. On
  the Stage 0 world, kNN pairs give a held-out R^2 of ~0.0 where random pairs
  give ~0.15.
- A latent direction is only worth planning for if it is *reachable*. Encoded
  latents contain per-chunk sampling noise, so a cluster's leading PCA
  directions are partly noise directions that no metadata change can produce.
  `reachability_cosine` measures this and `filter_actionable_directions`
  implements the fourth filter of SETUP.md 16. If almost nothing survives, that
  is failure mode F4 in PLAN.md 19, and it should be reported, not worked around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from ldva.acquisition.clustering import ClusterState
from ldva.acquisition.directions import AcquisitionDirection
from ldva.data.metadata import MetadataSpec


@dataclass
class MetadataMapperConfig:
    #: ridge strength for the local linear fit
    ridge: float = 1e-2
    #: fit on pair differences instead of raw within-cluster regression
    use_pair_differences: bool = True
    #: "random" draws pairs anywhere in the cluster (best SNR); "knn" uses
    #: nearest neighbours, which is more local but nearly signal-free
    pair_mode: str = "random"
    n_neighbors: int = 8
    max_pairs_per_cluster: int = 4000
    #: held-out fraction used to report an honest Jacobian R^2
    validation_fraction: float = 0.3
    #: cap on |delta_m| in normalized units, i.e. a trust region in metadata space
    max_step_norm: float = 0.35
    #: minimum members needed before a cluster's Jacobian is trusted
    min_members: int = 8
    seed: int = 0


@dataclass
class LocalJacobian:
    cluster_id: int
    #: (latent_dim, meta_dim) in normalized metadata units
    J: np.ndarray
    m_mean: np.ndarray
    z_mean: np.ndarray
    n_fit: int
    #: in-sample fraction of latent variance the local linear map explains
    r2: float = 0.0
    #: the same on held-out pairs; this is the number to trust
    r2_heldout: float = 0.0
    well_conditioned: bool = True
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def predict(self, delta_m_norm: np.ndarray) -> np.ndarray:
        return (self.J @ np.asarray(delta_m_norm, dtype=np.float64).T).T

    def reachability_cosine(self, v: np.ndarray, rcond: float = 1e-8) -> float:
        """How much of `v` lies in the column space of `J`.

        `J` maps `meta_dim` inputs into a `latent_dim` space, so at most
        `meta_dim` latent directions are reachable at all. This projects `v`
        onto `range(J)` and returns `cos(v, proj)` - 1.0 means fully
        actionable, 0.0 means no metadata change can move along `v`.
        """
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return 0.0
        u, sv, _ = np.linalg.svd(self.J, full_matrices=False)
        keep = sv > rcond * max(sv[0], 1e-300) if len(sv) else np.zeros(0, dtype=bool)
        if not np.any(keep):
            return 0.0
        basis = u[:, keep]
        proj = basis @ (basis.T @ v)
        return float(np.linalg.norm(proj) / nv)


@dataclass
class MetadataPlan:
    """A concrete, purchasable acquisition request for one direction."""

    direction_id: int
    cluster_id: int
    #: raw-unit metadata to collect at (one row per requested sample)
    metadata: np.ndarray
    #: the anchor each row was derived from
    anchor_metadata: np.ndarray
    #: sample id of the anchor behind each row, so a realized-movement
    #: measurement can be matched anchor by anchor instead of pooling
    anchor_sample_ids: np.ndarray
    delta_m_norm: np.ndarray
    desired_latent_direction: np.ndarray
    predicted_latent_delta: np.ndarray
    #: cosine between the desired direction and the displacement predicted by
    #: the *same* Jacobian the plan was solved with. It is therefore circular
    #: and close to 1 whenever the solve succeeded - it says the optimizer did
    #: its job, NOT that collection will move the latents that way. Judge
    #: reliability by `reachability_cosine` and `jacobian_r2_heldout`, and
    #: truth by `evaluate_realized_direction`.
    achievable_cosine: float
    #: fraction of the desired direction that lies in range(J) at all
    reachability_cosine: float
    jacobian_r2_heldout: float
    alpha: float
    residual: float
    feasible: bool
    info: dict = field(default_factory=dict)


class MetadataMapper:
    def __init__(self, spec: MetadataSpec, cfg: MetadataMapperConfig | None = None):
        self.spec = spec
        self.cfg = cfg or MetadataMapperConfig()
        self.jacobians: dict[int, LocalJacobian] = {}

    # ---- fitting ---------------------------------------------------------
    def fit(
        self,
        z: np.ndarray,
        metadata: np.ndarray,
        clusters: list[ClusterState],
    ) -> dict[int, LocalJacobian]:
        """Fit one local Jacobian per effect domain."""
        z = np.asarray(z, dtype=np.float64)
        m_norm = self.spec.normalize(np.asarray(metadata, dtype=np.float64))
        rng = np.random.default_rng(self.cfg.seed)
        self.jacobians = {}
        for c in clusters:
            ids = c.member_ids
            if len(ids) < self.cfg.min_members:
                continue
            if self.cfg.use_pair_differences:
                dM, dZ = self._pair_differences(z[ids], m_norm[ids], rng)
            else:
                dM = m_norm[ids] - m_norm[ids].mean(0)
                dZ = z[ids] - z[ids].mean(0)
            if len(dM) < max(self.cfg.min_members, len(self.spec) + 1):
                continue
            self.jacobians[c.cluster_id] = self._ridge_fit(
                c.cluster_id, dM, dZ, m_norm[ids].mean(0), z[ids].mean(0), rng
            )
        return self.jacobians

    def _pair_differences(
        self, z: np.ndarray, m_norm: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differences between pairs of samples inside the cluster.

        `pair_mode="random"` is the default because a pair difference's
        signal-to-noise ratio is `||J delta_m||^2 / 2 var(z | m)`: choosing the
        *nearest* neighbours sends `delta_m` to zero and leaves almost pure
        encoder noise. Staying inside one cluster already provides the locality
        that the linear approximation needs.
        """
        n = len(m_norm)
        if self.cfg.pair_mode == "knn":
            from sklearn.neighbors import NearestNeighbors

            k = min(self.cfg.n_neighbors + 1, n)
            _, idx = NearestNeighbors(n_neighbors=k).fit(m_norm).kneighbors(m_norm)
            a = np.repeat(np.arange(n), k - 1)
            b = idx[:, 1:].reshape(-1)
        elif self.cfg.pair_mode == "random":
            n_draw = min(self.cfg.max_pairs_per_cluster, max(n * self.cfg.n_neighbors, 64))
            a = rng.integers(0, n, size=n_draw)
            b = rng.integers(0, n, size=n_draw)
            keep = a != b
            a, b = a[keep], b[keep]
        else:
            raise ValueError(f"unknown pair_mode {self.cfg.pair_mode!r}")

        if len(a) > self.cfg.max_pairs_per_cluster:
            sel = rng.choice(len(a), size=self.cfg.max_pairs_per_cluster, replace=False)
            a, b = a[sel], b[sel]
        return m_norm[b] - m_norm[a], z[b] - z[a]

    def _ridge_fit(
        self,
        cluster_id: int,
        dM: np.ndarray,
        dZ: np.ndarray,
        m_mean: np.ndarray,
        z_mean: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> LocalJacobian:
        rng = rng or np.random.default_rng(self.cfg.seed)
        d_m = dM.shape[1]

        def solve(X, Y):
            A = X.T @ X + self.cfg.ridge * np.eye(d_m)
            return np.linalg.solve(A, X.T @ Y).T

        J = solve(dM, dZ)

        def r2_of(J_, X, Y):
            ss_res = float(np.sum((Y - X @ J_.T) ** 2))
            ss_tot = float(np.sum((Y - Y.mean(0)) ** 2))
            return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-20 else 0.0

        # held-out R^2: the in-sample value is optimistic, and the planner's
        # confidence in a direction should rest on generalization
        r2_out = 0.0
        n_val = int(self.cfg.validation_fraction * len(dM))
        if n_val >= 5 and len(dM) - n_val > d_m:
            perm = rng.permutation(len(dM))
            va, tr = perm[:n_val], perm[n_val:]
            r2_out = r2_of(solve(dM[tr], dZ[tr]), dM[va], dZ[va])

        sv = np.linalg.svd(J, compute_uv=False)
        return LocalJacobian(
            cluster_id=cluster_id,
            J=J,
            m_mean=m_mean,
            z_mean=z_mean,
            n_fit=len(dM),
            r2=r2_of(J, dM, dZ),
            r2_heldout=r2_out,
            well_conditioned=bool(sv[-1] > 1e-8 * max(sv[0], 1e-12)) if len(sv) else False,
            singular_values=sv,
        )

    # ---- inversion --------------------------------------------------------
    def solve_delta_m(
        self,
        cluster_id: int,
        v: np.ndarray,
        alpha: float,
        m_anchor_raw: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        """Constrained least-squares inversion for one anchor.

        Returns `(delta_m in normalized units, residual, feasible)`.
        """
        jac = self.jacobians.get(cluster_id)
        if jac is None:
            return np.zeros(len(self.spec)), np.inf, False

        target = alpha * np.asarray(v, dtype=np.float64)
        m_anchor_norm = self.spec.normalize(np.asarray(m_anchor_raw, dtype=np.float64).reshape(-1))

        # feasibility: stay in the box, respect the metadata-space trust region,
        # and never move a field the acquisition interface cannot control
        cap = self.cfg.max_step_norm
        lo = np.maximum(-m_anchor_norm, -cap)
        hi = np.minimum(1.0 - m_anchor_norm, cap)
        fixed = ~self.spec.controllable_mask
        lo[fixed] = 0.0
        hi[fixed] = 0.0

        # `lsq_linear` requires lb < ub strictly, so pinned variables are
        # *eliminated* from the problem rather than given a fudged window. A
        # field is pinned when it is uncontrollable, or when the anchor already
        # sits against the box edge in the only direction the step could go.
        free = (hi - lo) > 1e-12
        delta_m = np.zeros(len(self.spec), dtype=np.float64)
        if not np.any(free):
            return delta_m, float(np.linalg.norm(target)), False

        res = lsq_linear(
            jac.J[:, free], target, bounds=(lo[free], hi[free]), max_iter=200
        )
        delta_m[free] = np.asarray(res.x, dtype=np.float64)
        achieved = jac.J @ delta_m
        residual = float(np.linalg.norm(achieved - target))
        return delta_m, residual, bool(res.success)

    def plan_direction(
        self,
        direction: AcquisitionDirection,
        n_samples: int,
        metadata_all: np.ndarray,
        alpha: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> MetadataPlan:
        """Turn one latent direction into metadata to collect at."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        alpha = float(direction.delta if alpha is None else alpha)
        metadata_all = np.asarray(metadata_all, dtype=np.float64)

        anchors = direction.anchor_ids
        pick = rng.integers(0, len(anchors), size=max(n_samples, 1))
        anchor_ids_per_row = np.asarray(anchors)[pick]
        anchor_raw = metadata_all[anchor_ids_per_row]

        rows, deltas, residuals, feasible = [], [], [], True
        for a_raw in anchor_raw:
            d_m, resid, ok = self.solve_delta_m(
                direction.cluster_id, direction.vector, alpha, a_raw
            )
            feasible = feasible and ok
            residuals.append(resid)
            deltas.append(d_m)
            new_norm = np.clip(self.spec.normalize(a_raw) + d_m, 0.0, 1.0)
            rows.append(self.spec.clip(self.spec.denormalize(new_norm)))

        deltas = np.array(deltas)
        jac = self.jacobians.get(direction.cluster_id)
        pred = jac.predict(deltas).mean(0) if jac is not None else np.zeros_like(direction.vector)
        cos = _cosine(pred, direction.vector)
        return MetadataPlan(
            direction_id=direction.direction_id,
            cluster_id=direction.cluster_id,
            metadata=np.array(rows),
            anchor_metadata=anchor_raw,
            anchor_sample_ids=anchor_ids_per_row,
            delta_m_norm=deltas,
            desired_latent_direction=direction.vector,
            predicted_latent_delta=pred,
            achievable_cosine=cos,
            reachability_cosine=(
                jac.reachability_cosine(direction.vector) if jac is not None else 0.0
            ),
            jacobian_r2_heldout=float(jac.r2_heldout) if jac is not None else float("nan"),
            alpha=alpha,
            residual=float(np.mean(residuals)),
            feasible=feasible,
            info={
                "jacobian_r2": float(jac.r2) if jac is not None else float("nan"),
                "jacobian_r2_heldout": float(jac.r2_heldout) if jac is not None else float("nan"),
                "n_fit": int(jac.n_fit) if jac is not None else 0,
                "mean_step_norm": float(np.linalg.norm(deltas, axis=1).mean()),
            },
        )

    def plan_allocation(
        self,
        directions: list[AcquisitionDirection],
        allocation: np.ndarray,
        metadata_all: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> list[MetadataPlan]:
        """Metadata requests for a whole allocation (step 9 of PLAN.md 13)."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        out = []
        for d, n in zip(directions, np.asarray(allocation, dtype=np.int64)):
            if int(n) <= 0:
                continue
            out.append(self.plan_direction(d, int(n), metadata_all, rng=rng))
        return out

    def report(self) -> dict:
        if not self.jacobians:
            return {"n_jacobians": 0}
        r2 = [j.r2 for j in self.jacobians.values()]
        r2o = [j.r2_heldout for j in self.jacobians.values()]
        return {
            "n_jacobians": len(self.jacobians),
            "jacobian_r2_mean": float(np.mean(r2)),
            "jacobian_r2_min": float(np.min(r2)),
            "jacobian_r2_heldout_mean": float(np.mean(r2o)),
            "jacobian_r2_heldout_min": float(np.min(r2o)),
            "pair_mode": self.cfg.pair_mode,
            "n_fit_mean": float(np.mean([j.n_fit for j in self.jacobians.values()])),
            "all_well_conditioned": bool(
                all(j.well_conditioned for j in self.jacobians.values())
            ),
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_realized_direction(
    z_before_anchor: np.ndarray,
    z_after_collected: np.ndarray,
    desired_direction: np.ndarray,
) -> dict:
    """Did the collected data actually move where we asked? (PLAN.md 24)

    `z_before_anchor` are the latents of the anchors a plan was built from and
    `z_after_collected` the latents of what the environment actually returned.
    The cosine between the mean realized displacement and the desired direction
    is the metric SETUP.md 22 names; the per-sample distribution is reported too
    because a good mean can hide a very noisy realization.
    """
    z_before_anchor = np.atleast_2d(z_before_anchor)
    z_after_collected = np.atleast_2d(z_after_collected)
    delta = z_after_collected.mean(0) - z_before_anchor.mean(0)
    per_sample = [
        _cosine(z_after_collected[i] - z_before_anchor.mean(0), desired_direction)
        for i in range(len(z_after_collected))
    ]
    v = np.asarray(desired_direction, dtype=np.float64)
    v_unit = v / max(np.linalg.norm(v), 1e-12)
    along = float(np.dot(delta, v_unit))
    return {
        "direction_cosine": _cosine(delta, v),
        "per_sample_cosine_mean": float(np.mean(per_sample)),
        "per_sample_cosine_std": float(np.std(per_sample)),
        "frac_samples_positive_cosine": float(np.mean(np.array(per_sample) > 0)),
        "displacement_norm": float(np.linalg.norm(delta)),
        "displacement_along_direction": along,
        # orthogonal leakage: how much of the movement went somewhere else
        "orthogonal_error": float(
            np.linalg.norm(delta - along * v_unit)
        ),
    }


@dataclass
class ActionabilityConfig:
    """Thresholds for the metadata-actionability filter (SETUP.md 16).

    The primary test is `min_achievable_cosine`, measured by actually solving
    the constrained inversion at the direction's own anchors. Pure subspace
    reachability is not enough: a direction can lie entirely inside `range(J)`
    and still be unachievable because the required metadata change leaves the
    feasible box or exceeds the metadata-space step cap. Filtering on the
    constrained solution raised the *realized* direction cosine from 0.19 to
    0.44 on the Stage 0 world.
    """

    #: minimum cos(J delta_m*, v) from the constrained solve at the anchors
    min_achievable_cosine: float = 0.9
    #: minimum fraction of the direction that lies in range(J)
    min_reachability_cosine: float = 0.5
    #: minimum held-out R^2 of the domain's local linear map
    min_jacobian_r2_heldout: float = 0.0
    #: anchors to test per direction
    n_probe_anchors: int = 4
    #: keep at least this many directions even if all fail, so the planner
    #: still has something to compare; they stay flagged as unactionable
    keep_at_least: int = 2


def filter_actionable_directions(
    directions: list[AcquisitionDirection],
    mapper: MetadataMapper,
    metadata_all: np.ndarray,
    cfg: ActionabilityConfig | None = None,
) -> tuple[list[AcquisitionDirection], dict]:
    """Drop latent directions no metadata change can actually produce.

    This is the filter SETUP.md 16 lists alongside outward movement, density
    decrease and the trust region, and it is the one that decides whether
    directional acquisition is executable at all. Encoded latents carry
    per-chunk sampling noise, so a cluster's leading PCA directions are partly
    noise directions; asking the simulator to move along one of those produces
    an essentially random displacement, which is what shows up as a negative
    realized cosine.

    A low survival rate here is failure mode F4 of PLAN.md 19 and is returned in
    the report rather than hidden.
    """
    cfg = cfg or ActionabilityConfig()
    metadata_all = np.asarray(metadata_all, dtype=np.float64)
    kept, rejected = [], []
    for d in directions:
        jac = mapper.jacobians.get(d.cluster_id)
        if jac is None:
            rejected.append(
                {**d.summary(), "reason": "no_local_jacobian", "reachability": 0.0, "achievable": 0.0}
            )
            continue
        reach = jac.reachability_cosine(d.vector)

        # solve the real constrained problem at this direction's own anchors
        cosines = []
        for a_id in np.asarray(d.anchor_ids)[: cfg.n_probe_anchors]:
            delta_m, _, _ = mapper.solve_delta_m(
                d.cluster_id, d.vector, d.delta, metadata_all[int(a_id)]
            )
            achieved = jac.J @ delta_m
            cosines.append(_cosine(achieved, d.vector))
        achievable = float(np.mean(cosines)) if cosines else 0.0

        d.extra["reachability_cosine"] = float(reach)
        d.extra["achievable_cosine"] = achievable
        d.extra["jacobian_r2_heldout"] = float(jac.r2_heldout)
        entry = {**d.summary(), "reachability": reach, "achievable": achievable}
        if achievable < cfg.min_achievable_cosine:
            rejected.append({**entry, "reason": "not_achievable_under_constraints"})
        elif reach < cfg.min_reachability_cosine:
            rejected.append({**entry, "reason": "not_metadata_actionable"})
        elif jac.r2_heldout < cfg.min_jacobian_r2_heldout:
            rejected.append({**entry, "reason": "unreliable_local_map"})
        else:
            kept.append(d)

    fallback_used = False
    if len(kept) < cfg.keep_at_least:
        # rank the rejects by reachability so the planner keeps the best of a
        # bad set rather than nothing at all
        extra = sorted(rejected, key=lambda r: -r["achievable"])[: cfg.keep_at_least - len(kept)]
        extra_ids = {r["direction_id"] for r in extra}
        for d in directions:
            if d.direction_id in extra_ids and d not in kept:
                d.extra["actionability_fallback"] = True
                kept.append(d)
        rejected = [r for r in rejected if r["direction_id"] not in extra_ids]
        fallback_used = True

    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    report = {
        "n_input": len(directions),
        "n_actionable": len(kept),
        "n_rejected": len(rejected),
        "rejection_reasons": reasons,
        "survival_rate": len(kept) / max(len(directions), 1),
        "fallback_used": fallback_used,
        "mean_reachability_kept": float(
            np.mean([d.extra.get("reachability_cosine", 0.0) for d in kept])
        )
        if kept
        else 0.0,
        "mean_achievable_kept": float(
            np.mean([d.extra.get("achievable_cosine", 0.0) for d in kept])
        )
        if kept
        else 0.0,
        "thresholds": {
            "min_achievable_cosine": cfg.min_achievable_cosine,
            "min_reachability_cosine": cfg.min_reachability_cosine,
            "min_jacobian_r2_heldout": cfg.min_jacobian_r2_heldout,
        },
        "F4_warning": len(kept) < 0.25 * max(len(directions), 1),
    }
    return kept, report
