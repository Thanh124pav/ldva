"""Did the acquisition plan actually move the data where we asked? (PLAN.md 24)

Environment-agnostic: it talks to an `EnvAdapter`, so the same check runs on the
synthetic world, DMC and MetaWorld. The chain being validated is

    desired latent direction -> metadata perturbation -> collected sample
    -> realized latent movement

and the metric is `cosine(desired, realized)` (PLAN.md 18).

Two measurement details decide whether the number means anything:

- **Anchor matching.** Each planned row came from one specific anchor via its
  own `delta_m`, so the displacement to measure is
  `E[z | m_anchor + delta_m] - E[z | m_anchor]`, estimated anchor by anchor.
  Pooling all anchors mixes in the offset between the anchors a plan happened
  to use and the anchor set as a whole, which can flip the sign.
- **Paired sampling.** Both sides are collected with the same random stream. On
  the synthetic world a single chunk's latent noise (0.50) exceeds a planned
  displacement (0.22), so an unpaired 12-vs-12 comparison is pure noise; paired,
  12 samples match 256 unpaired ones. For MetaWorld and DMC the rollout is
  deterministic once the reset state is pinned and the expert is scripted, so
  pairing costs nothing and removes the remaining variance.
"""

from __future__ import annotations

import numpy as np

from ldva.acquisition.directions import AcquisitionDirection
from ldva.acquisition.metadata_mapper import MetadataPlan, evaluate_realized_direction
from ldva.envs.base import EnvAdapter


def measure_realized_latent_movement(
    model,
    adapter: EnvAdapter,
    plan: MetadataPlan,
    direction: AcquisitionDirection,
    policy_features: np.ndarray,
    rng: np.random.Generator,
    n_per_anchor: int = 8,
    paired: bool = True,
) -> dict:
    """Collect at the plan's metadata and check where the latents land."""
    deltas = []
    for row_m, anchor_m in zip(plan.metadata, plan.anchor_metadata):
        shared = int(rng.integers(0, 2**31 - 1))
        rng_base = np.random.default_rng(shared)
        rng_new = np.random.default_rng(shared) if paired else rng

        base_store = adapter.collect(
            np.tile(anchor_m, (n_per_anchor, 1)), rng_base, round_id=-2)
        new_store = adapter.collect(
            np.tile(row_m, (n_per_anchor, 1)), rng_new, round_id=-2)
        z_base = model.encode_store(base_store, policy_features).mean(0)
        z_new = model.encode_store(new_store, policy_features).mean(0)
        deltas.append(z_new - z_base)
    deltas = np.asarray(deltas)

    out = evaluate_realized_direction(
        np.zeros((1, deltas.shape[1])), deltas, direction.vector)
    out.update(
        {
            "direction_id": direction.direction_id,
            "cluster_id": direction.cluster_id,
            "achievable_cosine": plan.achievable_cosine,
            "reachability_cosine": plan.reachability_cosine,
            "jacobian_r2_heldout": plan.jacobian_r2_heldout,
            "n_anchors": int(len(deltas)),
            "n_per_anchor": int(n_per_anchor),
            "paired_sampling": bool(paired),
            # the mean realized displacement itself, so a caller can score
            # this direction against the OTHER candidate directions. Without
            # the vector, only the raw cosine is available, and that overstates
            # control whenever the latent space is collapsed - see
            # `direction_specificity`.
            "realized_delta": deltas.mean(0).tolist(),
        }
    )
    return out


def validate_all_directions(
    model,
    adapter: EnvAdapter,
    mapper,
    directions: list[AcquisitionDirection],
    metadata_all: np.ndarray,
    policy_features: np.ndarray,
    rng: np.random.Generator,
    n_anchors_per_direction: int = 3,
    n_per_anchor: int = 8,
) -> dict:
    """Run the check over every candidate direction and summarize.

    Reports the mean cosine over all directions *and* over the subset the
    actionability filter would keep, because that difference is the filter's
    whole justification.
    """
    rows = []
    for d in directions:
        plan = mapper.plan_direction(d, n_anchors_per_direction, metadata_all, rng=rng)
        rows.append(
            measure_realized_latent_movement(
                model, adapter, plan, d, policy_features, rng, n_per_anchor=n_per_anchor)
        )
    cos = np.array([r["direction_cosine"] for r in rows])
    ach = np.array([r["achievable_cosine"] for r in rows])
    keep = ach >= 0.9
    return {
        "per_direction": rows,
        "direction_cosine_mean": float(cos.mean()) if len(cos) else float("nan"),
        "direction_cosine_median": float(np.median(cos)) if len(cos) else float("nan"),
        "frac_directions_positive": float((cos > 0).mean()) if len(cos) else float("nan"),
        "actionable_only_cosine_mean": float(cos[keep].mean()) if keep.any() else None,
        "n_actionable": int(keep.sum()),
        "achievable_cosine_mean": float(ach.mean()) if len(ach) else float("nan"),
        "reachability_cosine_mean": float(
            np.mean([r["reachability_cosine"] for r in rows])) if rows else float("nan"),
        "jacobian_r2_heldout_mean": float(
            np.mean([r["jacobian_r2_heldout"] for r in rows])) if rows else float("nan"),
    }


# ---- is the movement SPECIFIC to the direction asked for? -----------------


def participation_ratio(z: np.ndarray) -> float:
    """Effective number of latent dimensions in use.

    `(sum lambda)^2 / sum lambda^2` over the covariance eigenvalues: equal to
    the latent dimension when variance is spread evenly, and to 1 when one
    direction dominates. Measured on this project it comes out at 1.0-2.2 on a
    32-dimensional latent space, i.e. the encoder collapses to roughly one
    effective dimension - which is why a raw cosine overstates control.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        return float("nan")
    cov = np.cov(z - z.mean(0), rowvar=False)
    lam = np.linalg.eigvalsh(np.atleast_2d(cov))
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def direction_specificity(
    realized_deltas: dict[int, np.ndarray],
    direction_vectors: dict[int, np.ndarray],
) -> dict:
    """How much better is the realized movement than requesting *another*
    direction?

    A raw `cos(delta_i, v_i)` is not interpretable on its own. Between two
    vectors in a d-dimensional space the chance level is about
    `sqrt(2/(pi*d))`, so a collapsed latent space inflates it for free: with an
    effective dimensionality near 1, every candidate direction points into the
    same narrow subspace and any displacement aligns with any direction.
    Measured here, a cosine of +0.910 - which reads as near-perfect control -
    sat only 1.3 standard deviations above requesting a different direction.

    The null therefore has to be the **other candidate directions**, not
    isotropic noise: "did collection move the latents along what we asked for,
    rather than along something else we might have asked for". That controls
    for the dimensionality and for the structure of the direction set at once.
    An isotropic null does neither, and using one inflated the z-scores
    threefold.

    Returns the mean z-score over directions; `>= 2` is the point at which the
    requested direction is distinguishable from an arbitrary one.
    """
    ids = [i for i in realized_deltas if i in direction_vectors]
    rows = []
    for i in ids:
        delta = np.asarray(realized_deltas[i], dtype=np.float64).reshape(-1)
        dn = delta / (np.linalg.norm(delta) + 1e-12)
        own = direction_vectors[i]
        own = np.asarray(own, dtype=np.float64).reshape(-1)
        cos_own = float(np.dot(dn, own / (np.linalg.norm(own) + 1e-12)))
        null = []
        for j in ids:
            if j == i:
                continue
            v = np.asarray(direction_vectors[j], dtype=np.float64).reshape(-1)
            null.append(float(np.dot(dn, v / (np.linalg.norm(v) + 1e-12))))
        if not null:
            continue
        null = np.asarray(null, dtype=np.float64)
        sd = float(null.std())
        # The primary statistic is the GAP, in cosine units:
        #
        #     gap_i = cos(delta_i, v_i) - mean_j |cos(delta_i, v_j)|
        #
        # Dividing by the null's spread instead - a z-score - has two failure
        # modes that a sweep exposed. When candidate directions are nearly
        # identical the spread collapses and the z-score explodes (values of
        # 2e8 were produced). When they are orthogonal the spread is also zero,
        # but that is the *best* case, not a degenerate one. The null's spread
        # is information about how diverse the direction set is; it is not the
        # measurement noise, so it does not belong in the denominator.
        #
        # The gap handles every case in one scale: perfect execution of
        # orthogonal directions gives 1.0 - 0.0 = 1.0; a collapsed space gives
        # 0.988 - 0.985 = 0.003; and a direction realized *worse* than the
        # alternatives gives a negative gap (measured: 0.232 - 0.660 = -0.43).
        null_abs = float(np.abs(null).mean())
        rows.append({
            "direction_id": int(i),
            "cos_desired": cos_own,
            "null_mean": float(null.mean()),
            "null_abs_mean": null_abs,
            "null_std": sd,
            "gap": float(cos_own - null_abs),
            # capped: a near-orthogonal null sends the raw ratio to 1e12, which
            # is why the gap and not the ratio is the criterion
            "ratio": float(min(cos_own / max(null_abs, 1e-3), 1e3)),
            # kept for reference; unreliable when `null_std` is small, which is
            # why it is no longer the criterion
            "z_score": float((cos_own - null.mean()) / sd) if sd > 1e-9 else float("nan"),
        })
    if not rows:
        return {"n_directions": 0, "z_score_mean": float("nan")}
    z = np.array([r["z_score"] for r in rows], dtype=np.float64)
    gap = np.array([r["gap"] for r in rows], dtype=np.float64)
    ratio = np.array([r["ratio"] for r in rows], dtype=np.float64)
    cos = np.array([r["cos_desired"] for r in rows], dtype=np.float64)
    nullabs = np.array([r["null_abs_mean"] for r in rows], dtype=np.float64)
    return {
        "n_directions": len(rows),
        "cos_desired_mean": float(np.nanmean(cos)),
        "null_abs_mean": float(np.nanmean(nullabs)),
        "cos_over_null": float(np.nanmean(cos) / max(np.nanmean(nullabs), 1e-9)),
        # primary: the gap in cosine units
        "gap_mean": float(np.nanmean(gap)),
        "gap_sem": float(np.nanstd(gap) / max(np.sqrt(len(gap)), 1)),
        "frac_directions_gap_positive": float(np.mean(gap > 0)),
        "ratio_mean": float(np.nanmean(ratio[np.isfinite(ratio)]))
        if np.isfinite(ratio).any() else float("nan"),
        # reference only; see the note on `gap` above
        "z_score_mean": (float(np.nanmean(z)) if np.isfinite(z).any()
                         else float("nan")),
        "null": "permutation over the other candidate directions",
        "per_direction": rows,
    }


# ---- how far can a LOCAL LINEAR map be trusted? --------------------------


def estimate_reach(
    encode_fn,
    centre: np.ndarray,
    steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05, 0.02, 0.01),
    r2_threshold: float = 0.95,
    n_probe: int = 400,
    rng: np.random.Generator | None = None,
) -> dict:
    """Largest step over which the map is still linear to `r2_threshold`.

    This is an empirical estimate of the manifold's **reach** (Federer;
    Niyogi-Smale-Weinberger): the radius within which a curved manifold is well
    approximated by its tangent space. It is the quantity that should set the
    planner's step, because `MetadataMapper` solves `J delta_m ~ alpha v` with a
    *local linear* `J` - a step longer than the reach asks that solve for
    something the linearisation cannot deliver.

    Why it matters here: richness and locality are not independently tunable.
    Both are set by the map's curvature - for a random-Fourier map, by one
    kernel bandwidth (Bochner) - so buying effective dimensions costs reach.
    Measured on the synthetic world's metadata->latent map, with 3-dimensional
    metadata:

        omega   effective dim   reach (R^2 > 0.95)
        0.5     2.05            <= 0.4
        2.0     3.19            <= 0.1
        8.0     14.40           <= 0.02

    So a configuration can be rich *or* take long steps, and the step has to be
    chosen from the measurement rather than fixed by hand. Compressing a rich
    map back down does **not** recover reach: reach depends on how fast the
    Jacobian turns with the *input*, and a fixed linear projection of the output
    does not change that - measured, PCA from 32 to 3 dimensions moved local
    R^2 only from 0.27 to 0.35 while destroying the richness (18.2 to 3.0
    effective dimensions).

    `steps` is scanned from large to small and the first passing value is
    returned, so the result is the coarsest trustworthy step.
    """
    rng = rng or np.random.default_rng(0)
    centre = np.asarray(centre, dtype=np.float64).reshape(-1)
    d = centre.shape[0]
    rows = []
    reach = None
    for eps in sorted(steps, reverse=True):
        pts = centre + rng.uniform(-eps, eps, size=(n_probe, d))
        y = np.asarray(encode_fn(pts), dtype=np.float64)
        x = np.concatenate([pts - centre, np.ones((n_probe, 1))], axis=1)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        denom = float(((y - y.mean(0)) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / max(denom, 1e-12)
        rows.append({"step": float(eps), "linear_r2": r2})
        if reach is None and r2 >= r2_threshold:
            reach = float(eps)
    return {
        "reach": reach,
        "r2_threshold": r2_threshold,
        "per_step": rows,
        "largest_step_tested": float(max(steps)),
        "smallest_step_tested": float(min(steps)),
        # None means even the smallest step tested was too curved; the caller
        # must not silently fall back to a default in that case
        "reach_below_tested_range": reach is None,
    }
