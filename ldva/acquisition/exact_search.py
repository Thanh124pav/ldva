"""Exact allocation search (PLAN.md 11.1, 9).

Enumerates every feasible allocation. This is the *oracle optimum under the
learned utility model* - not the true optimum - and its only jobs are to give
beam search something to be measured against on small problems and to show how
much the search, as opposed to the model, costs us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from ldva.acquisition.objective import (
    AllocationObjective,
    AllocationValue,
    enumerate_allocations,
    n_allocations,
)


@dataclass
class SolverResult:
    """Common return type for all three planners."""

    best_allocation: np.ndarray
    best_value: float
    solver: str
    n_evaluations: int
    #: every allocation the solver scored, best first (may be truncated)
    scored: list[AllocationValue] = None  # type: ignore[assignment]
    info: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scored is None:
            self.scored = []
        if self.info is None:
            self.info = {}

    @property
    def best_cost(self) -> float:
        return float(self.info.get("best_cost", 0.0))

    def top_k(self, k: int = 5) -> list[dict]:
        return [s.as_dict() for s in self.scored[:k]]

    def as_dict(self) -> dict:
        return {
            "solver": self.solver,
            "best_allocation": np.asarray(self.best_allocation).tolist(),
            "best_value": float(self.best_value),
            "n_evaluations": int(self.n_evaluations),
            **self.info,
        }


def exact_search(
    objective: AllocationObjective,
    max_allocations: int | None = 200_000,
    keep_top: int = 50,
    progress: bool = False,
) -> SolverResult:
    """Exhaustive search over feasible allocations."""
    A = objective.n_directions
    budget = objective.budget
    space = n_allocations(A, budget.budget)
    if max_allocations is not None and space > max_allocations and budget.monetary_budget is None:
        raise RuntimeError(
            f"exact search space is C({budget.budget}+{A}-1, {A}-1) = {space} > "
            f"max_allocations={max_allocations}; use beam search instead"
        )

    objective.reset_stats()
    it = enumerate_allocations(A, budget, max_count=max_allocations)
    if progress:
        it = tqdm(it, total=space, desc="exact", leave=False)

    scored: list[AllocationValue] = []
    best: AllocationValue | None = None
    for alloc in it:
        v = objective.predict(alloc)
        scored.append(v)
        if best is None or v.value > best.value:
            best = v

    if best is None:
        raise RuntimeError("no feasible allocation found")
    scored.sort(key=lambda s: -s.value)
    return SolverResult(
        best_allocation=best.allocation,
        best_value=best.value,
        solver="exact",
        n_evaluations=objective.n_evaluations,
        scored=scored[:keep_top],
        info={
            "search_space_size": int(space),
            "n_feasible_scored": len(scored),
            "best_cost": best.cost,
            "best_std": best.std,
        },
    )
