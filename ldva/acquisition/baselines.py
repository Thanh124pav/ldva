"""LDVA **internal ablations** (PLAN.md 12.1).

Despite the filename, nothing in this module is a published baseline. Every
scoring rule here reads its per-direction score out of the LDVA data model -
`model.effect_from_latents` or `model.utility_from_latents` applied to
hypothetical latents - so each one inherits LDVA's representation, its learned
readout and its sampler, and differs from LDVA only in how one scalar per
direction becomes an allocation. That makes them clean ablations of the
set-level planner and nothing more.

PLAN.md 12.1 is explicit about the consequence:

    Do **not** present them as reproductions of published methods.

The `*_style` names record which published *principle* each rule borrows, and
each docstring names the deviation. Independent baselines, which compute their
own scores from their own assumptions without consulting the LDVA model, live
in `ldva/acquisition/external_baselines.py` and are the ones PLAN.md 12.2 asks
for.

`random`, `equal` and `diversity` are the exception within this file: none of
them queries the model at all - random ignores it by construction and core-set
coverage uses only distances - so they are independent of LDVA predictions and
`classify_method` tags them `model_free`.

Every rule returns an allocation over the *same* candidate directions under the
*same* budget, so a comparison isolates the allocation rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ldva.acquisition.directions import AcquisitionDirection
from ldva.acquisition.exact_search import SolverResult
from ldva.acquisition.latent_sampler import LatentSampler
from ldva.acquisition.objective import AllocationObjective, BudgetSpec, uniform_allocation


def _allocate_from_scores(
    scores: np.ndarray,
    budget: BudgetSpec,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Spend the budget on the highest-scoring directions.

    `temperature > 0` draws each unit from a softmax over scores instead of
    always taking the argmax, which is what keeps a stochastic baseline from
    collapsing onto one direction. The feasibility check is inside the loop so
    a monetary budget can exclude a direction that is still the top scorer.
    """
    A = len(scores)
    alloc = np.zeros(A, dtype=np.int64)
    if temperature > 0:
        rng = rng or np.random.default_rng(0)
        logits = (np.asarray(scores, dtype=np.float64) - np.max(scores)) / temperature
        p = np.exp(logits)
        p /= p.sum()
    for _ in range(budget.budget):
        if temperature > 0:
            order = rng.choice(A, size=A, replace=False, p=p)
        else:
            order = np.argsort(-np.asarray(scores))
        placed = False
        for a in order:
            if budget.can_add(alloc, int(a)):
                alloc[int(a)] += 1
                placed = True
                break
        if not placed:
            break
    return alloc


def _result(name: str, alloc: np.ndarray, objective: AllocationObjective, info=None) -> SolverResult:
    """Score a baseline's allocation with the shared objective, for comparability."""
    v = objective.predict(alloc)
    return SolverResult(
        best_allocation=alloc,
        best_value=v.value,
        solver=name,
        n_evaluations=1,
        scored=[v],
        info={"best_cost": v.cost, "best_std": v.std, **(info or {})},
    )


# ---- allocation-rule baselines ------------------------------------------


def random_acquisition(objective: AllocationObjective, seed: int = 0) -> SolverResult:
    """Uniform random allocation (PLAN.md 12 "Random / Uniform")."""
    rng = np.random.default_rng(seed)
    budget = objective.budget
    alloc = np.zeros(objective.n_directions, dtype=np.int64)
    for _ in range(budget.budget):
        choices = [a for a in range(objective.n_directions) if budget.can_add(alloc, a)]
        if not choices:
            break
        alloc[rng.choice(choices)] += 1
    return _result("random", alloc, objective, {"seed": seed})


def equal_allocation(objective: AllocationObjective) -> SolverResult:
    """Equal budget per candidate direction (PLAN.md 12 "Equal allocation")."""
    alloc = uniform_allocation(objective.n_directions, objective.budget.budget)
    while not objective.budget.feasible(alloc) and alloc.sum() > 0:
        alloc[int(np.argmax(alloc))] -= 1
    return _result("equal", alloc, objective)


def diversity_acquisition(
    objective: AllocationObjective, z_support: np.ndarray, seed: int = 0
) -> SolverResult:
    """Core-set / coverage: favour directions furthest from existing support.

    Deviation: classic core-set selects *samples* to maximize coverage radius;
    here the same greedy max-min criterion is applied to candidate directions'
    proposal centres, and chosen centres join the covered set so the rule keeps
    spreading rather than piling onto one extreme direction.
    """
    centers = np.stack([d.proposal_center() for d in objective.directions])
    covered = list(np.asarray(z_support, dtype=np.float64))
    budget = objective.budget
    alloc = np.zeros(objective.n_directions, dtype=np.int64)
    for _ in range(budget.budget):
        cov = np.asarray(covered)
        d = np.min(
            np.linalg.norm(centers[:, None, :] - cov[None, :, :], axis=-1), axis=1
        )
        order = np.argsort(-d)
        placed = False
        for a in order:
            if budget.can_add(alloc, int(a)):
                alloc[int(a)] += 1
                covered.append(centers[int(a)])
                placed = True
                break
        if not placed:
            break
    return _result("diversity", alloc, objective, {"seed": seed})


# ---- score-based baselines ----------------------------------------------


def _direction_scores(
    model,
    sampler: LatentSampler,
    directions: list[AcquisitionDirection],
    policy_features: np.ndarray | None,
    n_draw: int = 32,
    mode: str = "effect",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-direction score and dispersion from hypothetical latents."""
    means, stds = [], []
    for d in directions:
        z = sampler.sample(d, n_draw)
        with torch.no_grad():
            if mode == "effect":
                s = model.effect_from_latents(
                    z[None, ...], policy_features=policy_features
                )[0].cpu().numpy()
            else:
                # marginal utility of each hypothetical sample on its own
                s = np.array(
                    [
                        float(
                            model.utility_from_latents(
                                z[i][None, None, :], policy_features=policy_features
                            ).item()
                        )
                        for i in range(len(z))
                    ]
                )
        means.append(float(np.mean(s)))
        stds.append(float(np.std(s)))
    return np.array(means), np.array(stds)


def uncertainty_acquisition(
    objective: AllocationObjective, model, n_draw: int = 32
) -> SolverResult:
    """Allocate to directions whose predicted effect is most uncertain.

    Deviation: uncertainty is the spread of the readout over hypothetical draws
    from the direction (a dispersion proxy), not an ensemble or a Bayesian
    posterior. It is the cheapest honest version and is clearly labelled as such.
    """
    _, stds = _direction_scores(
        model, objective.sampler, objective.directions, objective.policy_features, n_draw
    )
    alloc = _allocate_from_scores(stds, objective.budget)
    return _result("uncertainty", alloc, objective, {"scores": stds.tolist()})


def gradient_norm_acquisition(
    objective: AllocationObjective, model, n_draw: int = 32
) -> SolverResult:
    """Allocate by predicted effect *magnitude* (a gradient-norm analogue).

    Deviation: the true baseline scores real samples by ||g_i||, which cannot be
    computed for data that does not exist yet. The predicted absolute effect is
    the latent-space stand-in: large magnitude regardless of sign.
    """
    means, _ = _direction_scores(
        model, objective.sampler, objective.directions, objective.policy_features, n_draw
    )
    alloc = _allocate_from_scores(np.abs(means), objective.budget)
    return _result("gradient_norm", alloc, objective, {"scores": np.abs(means).tolist()})


def gradient_alignment_acquisition(
    objective: AllocationObjective, model, n_draw: int = 32
) -> SolverResult:
    """Allocate by *signed* predicted effect (a gradient-alignment analogue).

    This is the strongest purely additive scoring rule: it uses the same learned
    representation as LDVA but collapses each direction to one scalar and
    ignores the composition, so the gap to beam search is exactly the value of
    set-level reasoning.
    """
    means, _ = _direction_scores(
        model, objective.sampler, objective.directions, objective.policy_features, n_draw
    )
    alloc = _allocate_from_scores(means, objective.budget)
    return _result("gradient_alignment", alloc, objective, {"scores": means.tolist()})


def influence_acquisition(
    objective: AllocationObjective, model, n_draw: int = 32
) -> SolverResult:
    """CUPID-style policy-aware influence allocation.

    Deviation: CUPID scores already-collected trajectories by their influence on
    policy performance. Transposed to acquisition, each direction is scored by
    the model's predicted contextual effect under the *current* policy context
    and the budget goes to the top-scoring directions. The principle kept is
    "rank by policy-aware per-sample influence"; the deviation is that the score
    is predicted for hypothetical data rather than measured on real data.
    """
    means, _ = _direction_scores(
        model,
        objective.sampler,
        objective.directions,
        objective.policy_features,
        n_draw,
        mode="utility",
    )
    alloc = _allocate_from_scores(means, objective.budget)
    return _result("influence_cupid_style", alloc, objective, {"scores": means.tolist()})


def domain_mixture_acquisition(
    objective: AllocationObjective, model, n_draw: int = 32
) -> SolverResult:
    """Re-Mix-style source/domain mixture allocation.

    Deviation: Re-Mix optimizes mixture weights over *dataset domains* with a
    proxy objective. Here the domains are the latent clusters, each cluster gets
    a weight proportional to its mean predicted per-sample utility, and the
    budget is split across clusters in proportion to those weights before being
    spread evenly over that cluster's directions. The composition *within* a
    draw is still never evaluated jointly, which is the point of the comparison.
    """
    dirs = objective.directions
    means, _ = _direction_scores(
        model, objective.sampler, dirs, objective.policy_features, n_draw, mode="utility"
    )
    clusters = sorted({d.cluster_id for d in dirs})
    cl_score = np.array(
        [np.mean([means[i] for i, d in enumerate(dirs) if d.cluster_id == c]) for c in clusters]
    )
    w = np.exp(cl_score - cl_score.max())
    w = w / w.sum()
    budget = objective.budget
    alloc = np.zeros(len(dirs), dtype=np.int64)
    quota = np.floor(w * budget.budget).astype(np.int64)
    for ci, c in enumerate(clusters):
        members = [i for i, d in enumerate(dirs) if d.cluster_id == c]
        for j in range(int(quota[ci])):
            a = members[j % len(members)]
            if budget.can_add(alloc, a):
                alloc[a] += 1
    # distribute the rounding remainder to the best-scoring directions
    order = np.argsort(-means)
    while alloc.sum() < budget.budget:
        placed = False
        for a in order:
            if budget.can_add(alloc, int(a)):
                alloc[int(a)] += 1
                placed = True
                break
        if not placed:
            break
    return _result("domain_mixture_remix_style", alloc, objective, {"cluster_weights": w.tolist()})


def predicted_utility_greedy_acquisition(
    objective: AllocationObjective, model, n_draw: int = 32
) -> SolverResult:
    """DataMIL-style predicted-utility allocation.

    Deviation: DataMIL trains a utility predictor over data sources and selects
    by predicted utility. Here the additive per-sample predicted utility is used
    to rank directions, which is the additive counterpart of LDVA's set-level
    planner.
    """
    res = gradient_alignment_acquisition(objective, model, n_draw)
    res.solver = "predicted_utility_datamil_style"
    return res


# ---- registry -------------------------------------------------------------


@dataclass
class BaselineSuite:
    """Run the PLAN.md 12 minimum baseline suite in one call."""

    z_support: np.ndarray
    n_draw: int = 32
    seed: int = 0

    def run(self, objective: AllocationObjective, model) -> dict[str, SolverResult]:
        out: dict[str, SolverResult] = {}
        out["random"] = random_acquisition(objective, self.seed)
        out["equal"] = equal_allocation(objective)
        out["diversity"] = diversity_acquisition(objective, self.z_support, self.seed)
        out["uncertainty"] = uncertainty_acquisition(objective, model, self.n_draw)
        out["gradient_norm"] = gradient_norm_acquisition(objective, model, self.n_draw)
        out["gradient_alignment"] = gradient_alignment_acquisition(objective, model, self.n_draw)
        out["influence_cupid_style"] = influence_acquisition(objective, model, self.n_draw)
        out["domain_mixture_remix_style"] = domain_mixture_acquisition(objective, model, self.n_draw)
        out["predicted_utility_datamil_style"] = predicted_utility_greedy_acquisition(
            objective, model, self.n_draw
        )
        return out
