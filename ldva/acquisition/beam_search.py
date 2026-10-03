"""Beam-search allocation (PLAN.md 11.2; SETUP.md 19).

The primary practical planner. It keeps the `H` best partial allocations at each
step, so unlike greedy it can hold on to a composition that only pays off once
a complementary direction is added.

Two details that matter for the comparison against exact search:

- partial allocations are **deduplicated** by their counts. Without that, the
  same allocation is reached by `A!`-many orders and the beam fills with
  duplicates, which silently reduces the effective width to 1.
- when the monetary budget binds before the count budget, a beam entry that can
  no longer grow is retained as a finished candidate instead of being dropped.
"""

from __future__ import annotations

from tqdm.auto import tqdm

from ldva.acquisition.exact_search import SolverResult
from ldva.acquisition.objective import AllocationObjective, AllocationValue


def beam_search(
    objective: AllocationObjective,
    beam_width: int = 10,
    progress: bool = False,
    keep_top: int = 50,
) -> SolverResult:
    objective.reset_stats()
    budget = objective.budget
    A = objective.n_directions

    beam: list[AllocationValue] = [objective.predict(objective.zero_allocation())]
    finished: list[AllocationValue] = []
    all_scored: dict[tuple[int, ...], AllocationValue] = {}

    steps = range(budget.budget)
    if progress:
        steps = tqdm(steps, desc=f"beam(H={beam_width})", leave=False)

    for _ in steps:
        candidates: dict[tuple[int, ...], AllocationValue] = {}
        for entry in beam:
            grew = False
            for a in range(A):
                if not budget.can_add(entry.allocation, a):
                    continue
                trial = entry.allocation.copy()
                trial[a] += 1
                key = tuple(int(x) for x in trial)
                if key in candidates:
                    grew = True
                    continue
                v = objective.predict(trial)
                candidates[key] = v
                all_scored[key] = v
                grew = True
            if not grew:
                # budget exhausted for this entry: keep it as a final answer
                finished.append(entry)
        if not candidates:
            break
        beam = sorted(candidates.values(), key=lambda s: -s.value)[:beam_width]

    pool = [s for s in beam + finished if budget.complete(s.allocation)]
    if not pool:
        pool = beam + finished
    best = max(pool, key=lambda s: s.value)

    return SolverResult(
        best_allocation=best.allocation,
        best_value=best.value,
        solver=f"beam_{beam_width}",
        n_evaluations=objective.n_evaluations,
        scored=sorted(all_scored.values(), key=lambda s: -s.value)[:keep_top],
        info={
            "beam_width": beam_width,
            "best_cost": best.cost,
            "best_std": best.std,
            "n_final_candidates": len(pool),
            "final_beam": [s.as_dict() for s in beam[: min(beam_width, 10)]],
        },
    )


def beam_width_sweep(
    objective: AllocationObjective,
    widths: tuple[int, ...] = (5, 10, 20, 50),
    progress: bool = False,
) -> dict:
    """SETUP.md 19 initial beam widths, reported together.

    The objective cache is kept across widths on purpose: evaluation counts then
    show the *incremental* cost of a wider beam.
    """
    out = {}
    for h in widths:
        res = beam_search(objective, beam_width=h, progress=progress)
        out[h] = res
    return out
