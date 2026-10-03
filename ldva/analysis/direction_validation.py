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
