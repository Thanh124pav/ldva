"""The one scoring function every planner shares (PLAN.md 10; SETUP.md 18).

    V_hat(n | D, theta) = mean_m F_psi(Z_future^(m), D, theta)

Three properties matter and are enforced here rather than in each solver:

- the **whole composition** is scored jointly, never as a sum of per-direction
  scores, which is the entire reason for a set-level utility model;
- the Monte Carlo average over hypothetical draws is what turns a direction into
  a distribution of possible future samples rather than one latent point;
- exact, greedy and beam search all call `predict`, so a difference between
  them is a difference in *search*, not in objective. That is what makes the
  "beam approximately matches exact" check in SETUP.md 30 meaningful.

`BudgetSpec` carries both the count budget of PLAN.md 11 (`sum n_a <= B`) and
the monetary budget of SETUP.md 31 (`sum c_a n_a <= C`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ldva.acquisition.directions import AcquisitionDirection
from ldva.acquisition.latent_sampler import LatentSampler


@dataclass
class BudgetSpec:
    #: maximum number of acquired samples (B)
    budget: int
    #: per-direction monetary cost c_a; None means all costs are 1
    costs: np.ndarray | None = None
    #: monetary budget C; None means only the count budget applies
    monetary_budget: float | None = None
    #: require spending the full count budget (the usual fixed-B comparison)
    require_full_budget: bool = True

    def cost_of(self, allocation: np.ndarray) -> float:
        if self.costs is None:
            return float(np.sum(allocation))
        return float(np.dot(self.costs, allocation))

    def feasible(self, allocation: np.ndarray) -> bool:
        allocation = np.asarray(allocation)
        if allocation.min() < 0 or allocation.sum() > self.budget:
            return False
        if self.monetary_budget is not None and self.cost_of(allocation) > self.monetary_budget + 1e-9:
            return False
        return True

    def complete(self, allocation: np.ndarray) -> bool:
        """Is this a valid *final* answer?"""
        if not self.feasible(allocation):
            return False
        if self.require_full_budget and np.sum(allocation) != self.budget:
            # with a monetary cap the count budget may be unreachable
            return self.monetary_budget is not None and not self.can_add_any(allocation)
        return True

    def can_add(self, allocation: np.ndarray, a: int) -> bool:
        trial = np.asarray(allocation).copy()
        trial[a] += 1
        return self.feasible(trial)

    def can_add_any(self, allocation: np.ndarray) -> bool:
        return any(self.can_add(allocation, a) for a in range(len(allocation)))

    @classmethod
    def from_directions(
        cls,
        directions: list[AcquisitionDirection],
        budget: int,
        monetary_budget: float | None = None,
        use_costs: bool = False,
        require_full_budget: bool = True,
    ) -> "BudgetSpec":
        costs = (
            np.array([d.cost for d in directions], dtype=np.float64) if use_costs else None
        )
        return cls(
            budget=budget,
            costs=costs,
            monetary_budget=monetary_budget,
            require_full_budget=require_full_budget,
        )


@dataclass
class ObjectiveConfig:
    #: Monte Carlo draws per allocation
    n_mc: int = 16
    #: resample the hypothetical batches on every call (noisier, unbiased) or
    #: keep a fixed common random seed per allocation (comparable across solvers)
    common_random_numbers: bool = True
    seed: int = 0
    cache: bool = True
    #: safety valve against combinatorial blow-up in exact search
    max_evaluations: int | None = None
    #: also return the MC standard error, used by the uncertainty-aware report
    track_std: bool = True


@dataclass
class AllocationValue:
    allocation: np.ndarray
    value: float
    std: float = 0.0
    n_mc: int = 0
    cost: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "allocation": np.asarray(self.allocation).tolist(),
            "value": float(self.value),
            "std": float(self.std),
            "n_mc": int(self.n_mc),
            "cost": float(self.cost),
            **self.extra,
        }


class AllocationObjective:
    """`predict_allocation_utility(allocation)` for every planner."""

    def __init__(
        self,
        model,
        sampler: LatentSampler,
        directions: list[AcquisitionDirection],
        budget: BudgetSpec,
        policy_features: np.ndarray | None = None,
        ckpt_index: int = 0,
        cfg: ObjectiveConfig | None = None,
    ):
        if not directions:
            raise ValueError("need at least one candidate direction")
        self.model = model
        self.sampler = sampler
        self.directions = directions
        self.budget = budget
        self.policy_features = policy_features
        self.ckpt_index = int(ckpt_index)
        self.cfg = cfg or ObjectiveConfig()
        self.n_directions = len(directions)
        self._cache: dict[tuple[int, ...], AllocationValue] = {}
        self.n_evaluations = 0
        self.n_cache_hits = 0

    # ---- core ------------------------------------------------------------
    def predict(self, allocation: np.ndarray) -> AllocationValue:
        """Monte Carlo estimate of V_hat for one allocation."""
        alloc = np.asarray(allocation, dtype=np.int64)
        key = tuple(int(x) for x in alloc)
        if self.cfg.cache and key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]

        if self.cfg.max_evaluations is not None and self.n_evaluations >= self.cfg.max_evaluations:
            raise RuntimeError(
                f"allocation objective exceeded max_evaluations="
                f"{self.cfg.max_evaluations}; reduce the budget or use beam search"
            )

        total = int(alloc.sum())
        if total == 0:
            val = AllocationValue(alloc, 0.0, 0.0, 0, 0.0, {"empty": True})
            if self.cfg.cache:
                self._cache[key] = val
            return val

        if self.cfg.common_random_numbers:
            # a deterministic per-allocation seed keeps comparisons between
            # solvers free of MC noise without fixing one draw for all
            self.sampler.reseed(self.cfg.seed + (hash(key) % 1_000_003))

        draws = np.stack(
            [self.sampler.sample_allocation(self.directions, alloc) for _ in range(self.cfg.n_mc)]
        )
        with torch.no_grad():
            v = self.model.utility_from_latents(
                draws,
                policy_features=self.policy_features,
                ckpt_index=self.ckpt_index,
            ).cpu().numpy()

        self.n_evaluations += 1
        val = AllocationValue(
            allocation=alloc,
            value=float(np.mean(v)),
            std=float(np.std(v) / np.sqrt(len(v))) if self.cfg.track_std else 0.0,
            n_mc=self.cfg.n_mc,
            cost=self.budget.cost_of(alloc),
        )
        if self.cfg.cache:
            self._cache[key] = val
        return val

    def value(self, allocation: np.ndarray) -> float:
        return self.predict(allocation).value

    def predict_many(self, allocations: list[np.ndarray]) -> list[AllocationValue]:
        return [self.predict(a) for a in allocations]

    # ---- helpers ----------------------------------------------------------
    def zero_allocation(self) -> np.ndarray:
        return np.zeros(self.n_directions, dtype=np.int64)

    def reset_stats(self) -> None:
        self.n_evaluations = 0
        self.n_cache_hits = 0

    def clear_cache(self) -> None:
        self._cache.clear()

    def stats(self) -> dict:
        return {
            "n_evaluations": self.n_evaluations,
            "n_cache_hits": self.n_cache_hits,
            "n_cached": len(self._cache),
            "n_mc": self.cfg.n_mc,
            "n_directions": self.n_directions,
        }


def uniform_allocation(n_directions: int, budget: int) -> np.ndarray:
    """Equal split with the remainder spread over the first directions."""
    base = budget // n_directions
    rem = budget - base * n_directions
    alloc = np.full(n_directions, base, dtype=np.int64)
    alloc[:rem] += 1
    return alloc


def enumerate_allocations(
    n_directions: int, budget: BudgetSpec, max_count: int | None = None
):
    """Yield every feasible allocation, pruning on both budget forms.

    With unit costs and `require_full_budget`, this is exactly the
    `C(B + A - 1, A - 1)` compositions of PLAN.md 11.1.
    """
    costs = budget.costs if budget.costs is not None else np.ones(n_directions)
    alloc = np.zeros(n_directions, dtype=np.int64)
    count = 0

    def rec(a: int, remaining: int, spent: float):
        nonlocal count
        if a == n_directions - 1:
            take_range = [remaining] if budget.require_full_budget else range(remaining + 1)
            for k in take_range:
                if budget.monetary_budget is not None and spent + costs[a] * k > budget.monetary_budget + 1e-9:
                    continue
                alloc[a] = k
                if budget.complete(alloc):
                    count += 1
                    if max_count is not None and count > max_count:
                        raise RuntimeError(
                            f"enumeration exceeded max_count={max_count}; "
                            "use beam search for this problem size"
                        )
                    yield alloc.copy()
            alloc[a] = 0
            return
        for k in range(remaining + 1):
            if budget.monetary_budget is not None and spent + costs[a] * k > budget.monetary_budget + 1e-9:
                break
            alloc[a] = k
            yield from rec(a + 1, remaining - k, spent + costs[a] * k)
        alloc[a] = 0

    yield from rec(0, budget.budget, 0.0)


def n_allocations(n_directions: int, budget: int) -> int:
    """C(B + A - 1, A - 1), the size of the exact search space."""
    from math import comb

    return comb(budget + n_directions - 1, n_directions - 1)
