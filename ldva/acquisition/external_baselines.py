"""Independent external baselines (PLAN.md 12.2, 15 P0.3).

The distinction PLAN.md 12 insists on, and that this module exists to enforce:

- `ldva/acquisition/baselines.py` holds **LDVA internal ablations**. Every
  scoring rule there calls `model.effect_from_latents` or
  `model.utility_from_latents`, so it inherits LDVA's representation, its
  learned readout and its hypothetical-latent sampler, and differs only in how
  one scalar per direction becomes an allocation. Those isolate a design
  choice. They are not reproductions of published methods, and PLAN.md 12.1
  says not to present them as such.
- this module holds **independent baselines**. Nothing here touches the LDVA
  data model. Scores come from real gradients of the real policy on the real
  collected data against the real evaluation set.

What the two groups *must* share is the action space: the same candidate
directions and the same budget, or a difference in final performance would
reflect a difference in what each method was allowed to request rather than in
how well it chose. That is experimental control, not borrowing.

**The prospective adaptation, stated plainly.** Direct gradient alignment and
direct influence are retrospective: both score a sample that already exists by
differentiating the loss at it. Neither has anything to say about a region of
metadata space from which nothing has been collected - which is the whole
problem LDVA is built for. Pretending otherwise would make the baseline a straw
man, so the adaptation here is the strongest honest one:

    a candidate direction is scored by the real direct score of the real
    samples already nearest to the metadata it would collect at.

The nearest-neighbour lookup runs in *normalized metadata space*, which is the
environment's own coordinate system - not the LDVA latent space - so no part of
the score depends on the learned model. Where a direction points at genuinely
unpopulated metadata the nearest real samples are far away and the score
degrades to that of the closest populated region. That is a faithful
representation of the method's blind spot, and `mean_neighbor_distance` is
recorded per direction so the write-up can say how far the extrapolation
reached rather than leaving it implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ldva.acquisition.directions import AcquisitionDirection
from ldva.acquisition.exact_search import SolverResult
from ldva.acquisition.objective import AllocationObjective, BudgetSpec
from ldva.supervision.base import SupervisionTask, flat_grad, per_sample_grads

# ---- direct per-sample scores (no LDVA model anywhere) -------------------


def direct_gradient_alignment_scores(
    task: SupervisionTask,
    sample_ids: np.ndarray,
    mode: str = "dot",
    lr: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    """`<g_i, g_val>` per real sample, at the real current checkpoint.

    The published quantity, computed the published way: one backward pass per
    sample for `g_i`, one for the evaluation loss `g_val`, then their inner
    product. `mode="dot"` is the first-order predicted utility rise from an SGD
    step; `mode="cosine"` is the scale-free version.
    """
    ids = np.asarray(sample_ids, dtype=np.int64)
    if len(ids) == 0:
        return np.zeros(0, dtype=np.float64)
    params = list(task.parameters())
    g = per_sample_grads(task, ids)
    gv = flat_grad(task.val_loss(), params).detach()
    if mode == "cosine":
        return ((g @ gv) / (g.norm(dim=1) * gv.norm() + eps)).cpu().numpy()
    if mode == "grad_norm":
        return g.norm(dim=1).cpu().numpy()
    return ((g @ gv) * lr).cpu().numpy()


def direct_influence_scores(
    task: SupervisionTask,
    sample_ids: np.ndarray,
    damping: float = 1e-2,
    n_lissa: int = 8,
    scale: float = 10.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """`-g_val^T H^-1 g_i` per real sample, with a LiSSA inverse-HVP.

    This is what makes influence a *different* method rather than a rescaling
    of gradient alignment: the curvature term reweights directions the
    validation gradient is sensitive to. The inverse Hessian-vector product
    uses the standard LiSSA recursion

        h_0 = v,    h_{k+1} = v + (I - (H + damping I) / scale) h_k

    with Hessian-vector products taken by double backward through the training
    loss on the scored samples. `scale` must exceed the largest eigenvalue for
    the recursion to converge; the default is deliberately conservative and
    `n_lissa` is small, so this is a cheap approximation - stated here rather
    than implied to be exact.
    """
    ids = np.asarray(sample_ids, dtype=np.int64)
    if len(ids) == 0:
        return np.zeros(0, dtype=np.float64)
    params = list(task.parameters())
    gv = flat_grad(task.val_loss(), params, retain_graph=True).detach()

    def hvp(vec: torch.Tensor) -> torch.Tensor:
        loss = task.train_loss(ids)
        g = flat_grad(loss, params, retain_graph=True, create_graph=True)
        return flat_grad((g * vec).sum(), params, retain_graph=True).detach()

    h = gv.clone()
    for _ in range(max(n_lissa, 1)):
        h = gv + h - (hvp(h) + damping * h) / scale
    ihvp = h / scale

    g = per_sample_grads(task, ids)
    # utility is minus the validation loss, so a positive value is a gain
    return (g @ ihvp).cpu().numpy() * 1.0 / (1.0 + eps)


# ---- direction scoring through environment-native metadata ---------------


@dataclass
class ProspectiveProxyConfig:
    """How a direction borrows the scores of nearby real samples."""

    #: real samples averaged per direction
    k_neighbors: int = 8
    #: metadata rows drawn per direction to represent where it would collect
    n_probe: int = 4
    #: inverse-distance weighting instead of a plain mean over the k neighbours
    distance_weighted: bool = True
    seed: int = 0


@dataclass
class DirectionProxy:
    """Which real samples stood in for each candidate direction."""

    direction_id: int
    neighbor_ids: np.ndarray
    distances: np.ndarray
    score: float
    extra: dict = field(default_factory=dict)


def _target_metadata(
    direction: AcquisitionDirection,
    mapper,
    metadata_all: np.ndarray,
    n_probe: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Where this direction would collect, in raw metadata units.

    Uses the same `MetadataMapper` the real planner uses, because that is the
    environment's answer to "what does this direction mean concretely" and it
    is shared by every method under comparison. Should the mapper fail to
    produce a feasible plan, the anchors themselves are the fallback: a
    direction that cannot be executed is then scored where it starts.
    """
    try:
        plan = mapper.plan_direction(direction, n_probe, metadata_all, rng=rng)
        rows = np.asarray(plan.metadata, dtype=np.float64)
        if rows.size:
            return np.atleast_2d(rows)
    except Exception:
        pass
    return np.atleast_2d(np.asarray(metadata_all, dtype=np.float64)[direction.anchor_ids])


def score_directions_by_proxy(
    directions: list[AcquisitionDirection],
    per_sample_scores: np.ndarray,
    metadata_all: np.ndarray,
    metadata_spec,
    mapper,
    cfg: ProspectiveProxyConfig | None = None,
) -> tuple[np.ndarray, list[DirectionProxy]]:
    """Average the real per-sample scores of each direction's nearest real data.

    The distance is Euclidean in normalized metadata space, so every metadata
    field contributes on the same scale regardless of its raw units - a goal
    radius in metres and an angle in radians would otherwise be incomparable.
    """
    cfg = cfg or ProspectiveProxyConfig()
    rng = np.random.default_rng(cfg.seed)
    metadata_all = np.asarray(metadata_all, dtype=np.float64)
    scores = np.asarray(per_sample_scores, dtype=np.float64)
    m_norm = metadata_spec.normalize(metadata_all)

    out_scores, proxies = [], []
    for d in directions:
        targets = _target_metadata(d, mapper, metadata_all, cfg.n_probe, rng)
        t_norm = metadata_spec.normalize(targets)
        # (n_targets, n_samples) distances, then the k nearest over all targets
        dist = np.linalg.norm(t_norm[:, None, :] - m_norm[None, :, :], axis=-1)
        best = dist.min(axis=0)
        k = int(min(cfg.k_neighbors, len(best)))
        nn = np.argsort(best)[:k]
        dn = best[nn]
        if cfg.distance_weighted:
            w = 1.0 / (dn + 1e-6)
            w = w / w.sum()
            s = float(np.sum(w * scores[nn]))
        else:
            s = float(np.mean(scores[nn]))
        out_scores.append(s)
        proxies.append(
            DirectionProxy(
                direction_id=d.direction_id,
                neighbor_ids=nn,
                distances=dn,
                score=s,
                extra={
                    "mean_neighbor_distance": float(dn.mean()),
                    "max_neighbor_distance": float(dn.max()) if k else float("nan"),
                    "n_targets": int(len(targets)),
                },
            )
        )
    return np.asarray(out_scores, dtype=np.float64), proxies


# ---- allocation ----------------------------------------------------------


def _allocate_top_scores(scores: np.ndarray, budget: BudgetSpec) -> np.ndarray:
    """Spend the budget on the highest-scoring directions.

    Deliberately the same rule the additive LDVA ablations use, so the only
    thing separating this baseline from `gradient_alignment_acquisition` is
    *where the score came from* - measured gradients here, predicted effects
    there. That is the comparison PLAN.md 12 is asking for.
    """
    alloc = np.zeros(len(scores), dtype=np.int64)
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    for _ in range(budget.budget):
        placed = False
        for a in order:
            if budget.can_add(alloc, int(a)):
                alloc[int(a)] += 1
                placed = True
                break
        if not placed:
            break
    return alloc


def _result(
    name: str,
    alloc: np.ndarray,
    objective: AllocationObjective,
    scores: np.ndarray,
    proxies: list[DirectionProxy],
) -> SolverResult:
    """Package an allocation.

    `best_value` is LDVA's prediction for the allocation this baseline chose,
    recorded *only* so every method in the report carries the same column. It
    plays no part in the choice, and the honest comparison between methods is
    the realized rollout metric, not this number.
    """
    v = objective.predict(alloc)
    return SolverResult(
        best_allocation=alloc,
        best_value=v.value,
        solver=name,
        n_evaluations=1,
        scored=[v],
        info={
            "best_cost": v.cost,
            "best_std": v.std,
            "scores": np.asarray(scores).tolist(),
            "independent_of_ldva_model": True,
            "mean_neighbor_distance": float(
                np.mean([p.extra["mean_neighbor_distance"] for p in proxies])
            )
            if proxies
            else float("nan"),
            "note": "score measured on real data; ldva prediction recorded, not used",
        },
    )


def direct_gradient_alignment_acquisition(
    objective: AllocationObjective,
    task: SupervisionTask,
    metadata_all: np.ndarray,
    metadata_spec,
    mapper,
    sample_ids: np.ndarray | None = None,
    cfg: ProspectiveProxyConfig | None = None,
    mode: str = "dot",
) -> SolverResult:
    """PLAN.md 12.2 "Direct Gradient Alignment", adapted to acquisition."""
    ids = (
        np.arange(len(metadata_all), dtype=np.int64)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.int64)
    )
    per_sample = direct_gradient_alignment_scores(task, ids, mode=mode)
    full = np.zeros(len(metadata_all), dtype=np.float64)
    full[ids] = per_sample
    scores, proxies = score_directions_by_proxy(
        objective.directions, full, metadata_all, metadata_spec, mapper, cfg
    )
    alloc = _allocate_top_scores(scores, objective.budget)
    return _result("direct_gradient_alignment", alloc, objective, scores, proxies)


def direct_influence_acquisition(
    objective: AllocationObjective,
    task: SupervisionTask,
    metadata_all: np.ndarray,
    metadata_spec,
    mapper,
    sample_ids: np.ndarray | None = None,
    cfg: ProspectiveProxyConfig | None = None,
    damping: float = 1e-2,
    n_lissa: int = 8,
) -> SolverResult:
    """PLAN.md 12.2 "Direct Influence", adapted to acquisition."""
    ids = (
        np.arange(len(metadata_all), dtype=np.int64)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.int64)
    )
    per_sample = direct_influence_scores(
        task, ids, damping=damping, n_lissa=n_lissa
    )
    full = np.zeros(len(metadata_all), dtype=np.float64)
    full[ids] = per_sample
    scores, proxies = score_directions_by_proxy(
        objective.directions, full, metadata_all, metadata_spec, mapper, cfg
    )
    alloc = _allocate_top_scores(scores, objective.budget)
    return _result("direct_influence", alloc, objective, scores, proxies)


#: PLAN.md 12.2's minimum local-stage set. Random and Diversity live in
#: `baselines.py` because neither one needs a score at all - random ignores the
#: model by construction and core-set coverage uses only distances - so they
#: are already independent of LDVA's predictions.
EXTERNAL_METHODS = ("direct_gradient_alignment", "direct_influence")

#: everything in `baselines.py` whose score is read out of the LDVA model.
#: PLAN.md 12.1: report these as ablations, never as published methods.
LDVA_ABLATION_METHODS = (
    "uncertainty",
    "gradient_norm",
    "gradient_alignment",
    "influence_cupid_style",
    "domain_mixture_remix_style",
    "predicted_utility_datamil_style",
)

#: methods that need neither the LDVA model nor a gradient computation
MODEL_FREE_METHODS = ("random", "equal", "diversity")


def classify_method(name: str) -> str:
    """"external" | "ldva_ablation" | "model_free" | "ldva" - for the report.

    Keeping this in code rather than in a comment is what stops an ablation
    from drifting into a results table as a published baseline.
    """
    if name in EXTERNAL_METHODS:
        return "external"
    if name in LDVA_ABLATION_METHODS:
        return "ldva_ablation"
    if name in MODEL_FREE_METHODS:
        return "model_free"
    if name.startswith("ldva"):
        return "ldva"
    return "unknown"
