"""Greedy allocation (PLAN.md 11.3; SETUP.md 19).

One unit at a time to the direction with the largest one-step marginal gain.
This is a **baseline**, not the planner: its purpose is to expose whether
complementarity matters. If greedy matches beam search everywhere, the joint
composition is not doing any work and the set-level story is weak.

Cost-aware mode divides the marginal gain by the direction's cost, which is the
standard knapsack-style ratio rule, so a monetary budget is spent where gain per
dollar is highest rather than where raw gain is highest.
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from ldva.acquisition.exact_search import SolverResult
from ldva.acquisition.objective import AllocationObjective, AllocationValue


def greedy_search(
    objective: AllocationObjective,
    cost_aware: bool = False,
    progress: bool = False,
) -> SolverResult:
    """Allocate greedily until the budget cannot take another unit."""
    objective.reset_stats()
    budget = objective.budget
    A = objective.n_directions
    alloc = objective.zero_allocation()
    current = objective.predict(alloc)
    trace: list[AllocationValue] = []

    steps = range(budget.budget)
    if progress:
        steps = tqdm(steps, desc="greedy", leave=False)

    for _ in steps:
        best_a, best_v, best_score = None, None, -np.inf
        for a in range(A):
            if not budget.can_add(alloc, a):
                continue
            trial = alloc.copy()
            trial[a] += 1
            v = objective.predict(trial)
            marginal = v.value - current.value
            unit_cost = 1.0 if budget.costs is None else float(budget.costs[a])
            score = marginal / max(unit_cost, 1e-12) if cost_aware else marginal
            if score > best_score:
                best_a, best_v, best_score = a, v, score
        if best_a is None:
            break
        alloc = alloc.copy()
        alloc[best_a] += 1
        current = best_v
        trace.append(current)

    return SolverResult(
        best_allocation=alloc,
        best_value=current.value,
        solver="greedy_cost_aware" if cost_aware else "greedy",
        n_evaluations=objective.n_evaluations,
        scored=sorted(trace, key=lambda s: -s.value)[:50],
        info={
            "best_cost": current.cost,
            "best_std": current.std,
            "n_steps": len(trace),
            "cost_aware": cost_aware,
            "value_trace": [float(t.value) for t in trace],
        },
    )
