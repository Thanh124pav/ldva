"""Allocation planners (PLAN.md 11; SETUP.md 19, 30 criterion 4).

The planners are tested against an objective with a *known analytic optimum*,
so "beam matches exact" and "greedy can miss complementarity" are checked
against ground truth rather than against each other.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldva.acquisition.beam_search import beam_search, beam_width_sweep
from ldva.acquisition.exact_search import exact_search
from ldva.acquisition.greedy import greedy_search
from ldva.acquisition.objective import (
    AllocationObjective,
    BudgetSpec,
    ObjectiveConfig,
    enumerate_allocations,
    n_allocations,
    uniform_allocation,
)

A, B = 5, 8
#: asymmetric weights plus a bonus for pairing directions 2 and 4, so the
#: optimum is unique, non-uniform, and reachable only by holding a partial
#: allocation that looks worse than the greedy choice
W = np.array([1.0, 0.6, 2.2, 0.5, 0.9])
PAIR = (2, 4)
PAIR_BONUS = 1.8


def true_value(counts) -> float:
    c = np.asarray(counts, dtype=np.float64)
    return float(np.sum(W * np.sqrt(c)) + PAIR_BONUS * min(c[PAIR[0]], c[PAIR[1]]))


class _CountModel:
    """A utility model that reads the allocation back out of the latents."""

    def __init__(self, centers):
        self.centers = centers

    def utility_from_latents(self, Z, policy_features=None, ckpt_index=0, mask=None):
        Z = np.asarray(Z)
        out = []
        for b in range(Z.shape[0]):
            which = np.argmin(((Z[b][:, None, :] - self.centers[None]) ** 2).sum(-1), axis=1)
            out.append(true_value(np.bincount(which, minlength=A)))
        return torch.tensor(out, dtype=torch.float32)


class _Direction:
    def __init__(self, i, center):
        self.direction_id = i
        self.cluster_id = i
        self.vector = np.eye(len(center))[0]
        self.cost = 1.0
        self.anchors = center[None, :]
        self.anchor_ids = np.array([i])
        self.delta = 0.0

    def proposal_center(self):
        return self.anchors[0]


class _Sampler:
    """Deterministic-ish sampler: draws tight around each direction's centre."""

    def __init__(self, centers, sigma=0.02, seed=0):
        self.centers = centers
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self, d, n):
        return self.centers[d.direction_id] + self.sigma * self.rng.normal(
            size=(n, self.centers.shape[1]))

    def sample_allocation(self, directions, allocation):
        parts = [self.sample(d, int(k)) for d, k in zip(directions, allocation) if int(k) > 0]
        return np.concatenate(parts) if parts else np.zeros((0, self.centers.shape[1]))


@pytest.fixture
def setup():
    centers = np.eye(A) * 10.0
    dirs = [_Direction(i, centers[i]) for i in range(A)]
    budget = BudgetSpec(budget=B)
    obj = AllocationObjective(_CountModel(centers), _Sampler(centers), dirs, budget,
                              policy_features=np.zeros(4),
                              cfg=ObjectiveConfig(n_mc=3, seed=0))
    best = max((a.copy() for a in enumerate_allocations(A, budget)), key=true_value)
    return obj, best


def test_exact_search_recovers_the_analytic_optimum(setup):
    obj, best = setup
    res = exact_search(obj)
    assert np.array_equal(res.best_allocation, best)
    assert res.n_evaluations == n_allocations(A, B)
    assert res.info["search_space_size"] == n_allocations(A, B)


def test_all_solvers_spend_the_whole_budget(setup):
    obj, _ = setup
    for res in (exact_search(obj), greedy_search(obj), beam_search(obj, 10)):
        assert res.best_allocation.sum() == B, res.solver
        assert (res.best_allocation >= 0).all()


def test_beam_matches_exact_and_is_cheaper(setup):
    """SETUP.md 30 criterion 4, plus the point of using beam search at all."""
    obj, _ = setup
    ex = exact_search(obj)
    obj.clear_cache()
    bm = beam_search(obj, beam_width=10)
    assert bm.best_value >= ex.best_value - 1e-6
    assert bm.n_evaluations < ex.n_evaluations


def test_beam_width_one_equals_greedy(setup):
    """A width-1 beam is greedy by construction; if it differs, one of them is
    not calling the shared objective the way it claims to."""
    obj, _ = setup
    gr = greedy_search(obj)
    b1 = beam_search(obj, beam_width=1)
    assert np.array_equal(gr.best_allocation, b1.best_allocation)


def test_greedy_can_be_beaten_by_beam_on_complementarity(setup):
    """If greedy always matched beam, the set-level planner would be pointless,
    so the test fixture itself has to exhibit complementarity."""
    obj, _ = setup
    gr = greedy_search(obj)
    bm = beam_search(obj, beam_width=10)
    assert bm.best_value > gr.best_value + 1e-6


def test_wider_beams_do_not_get_worse(setup):
    obj, _ = setup
    results = beam_width_sweep(obj, widths=(1, 3, 5, 10, 20))
    vals = [results[h].best_value for h in (1, 3, 5, 10, 20)]
    assert max(vals) == pytest.approx(vals[-1], abs=1e-6) or vals[-1] >= vals[0]


def test_uniform_allocation_is_valid_but_suboptimal(setup):
    obj, best = setup
    u = uniform_allocation(A, B)
    assert u.sum() == B
    assert obj.value(u) < obj.value(best)


def test_objective_is_cached_and_deterministic(setup):
    obj, best = setup
    v1 = obj.value(best)
    hits = obj.n_cache_hits
    v2 = obj.value(best)
    assert v1 == v2
    assert obj.n_cache_hits == hits + 1


def test_objective_scores_the_whole_composition_not_a_sum(setup):
    """PLAN.md 10: V_hat(n) must not reduce to independent per-direction scores."""
    obj, _ = setup
    mixed = np.zeros(A, dtype=np.int64)
    mixed[PAIR[0]] = mixed[PAIR[1]] = 4
    solo_a = np.zeros(A, dtype=np.int64)
    solo_a[PAIR[0]] = 8
    solo_b = np.zeros(A, dtype=np.int64)
    solo_b[PAIR[1]] = 8
    # the pairing bonus makes the mixture worth more than either pure option
    assert obj.value(mixed) > max(obj.value(solo_a), obj.value(solo_b))


def test_exact_search_refuses_an_intractable_space(setup):
    obj, _ = setup
    with pytest.raises(RuntimeError, match="exact search space"):
        exact_search(obj, max_allocations=10)


def test_enumeration_size_matches_the_combinatorial_formula():
    budget = BudgetSpec(budget=6)
    allocs = list(enumerate_allocations(4, budget))
    assert len(allocs) == n_allocations(4, 6)
    assert all(a.sum() == 6 for a in allocs)
    assert len({tuple(a) for a in allocs}) == len(allocs)


def test_monetary_budget_is_respected_by_every_solver():
    """SETUP.md 31: sum_a c_a n_a <= C, not only sum_a n_a <= B."""
    centers = np.eye(A) * 10.0
    dirs = [_Direction(i, centers[i]) for i in range(A)]
    costs = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
    for i, c in enumerate(costs):
        dirs[i].cost = float(c)
    budget = BudgetSpec(budget=B, costs=costs, monetary_budget=8.0,
                        require_full_budget=False)
    obj = AllocationObjective(_CountModel(centers), _Sampler(centers), dirs, budget,
                              policy_features=np.zeros(4),
                              cfg=ObjectiveConfig(n_mc=3, seed=0))
    for res in (exact_search(obj, max_allocations=10**6),
                greedy_search(obj, cost_aware=True),
                beam_search(obj, beam_width=10)):
        cost = budget.cost_of(res.best_allocation)
        assert cost <= 8.0 + 1e-9, (res.solver, cost)
        assert budget.feasible(res.best_allocation), res.solver


def test_budget_spec_feasibility_rules():
    b = BudgetSpec(budget=4)
    assert b.feasible(np.array([2, 2]))
    assert not b.feasible(np.array([3, 3]))
    assert not b.feasible(np.array([-1, 2]))
    assert b.complete(np.array([2, 2]))
    assert not b.complete(np.array([1, 1]))
    assert b.can_add(np.array([1, 1]), 0)
    assert not b.can_add(np.array([2, 2]), 0)

    bc = BudgetSpec(budget=10, costs=np.array([3.0, 1.0]), monetary_budget=5.0,
                    require_full_budget=False)
    assert bc.feasible(np.array([1, 2]))
    assert not bc.feasible(np.array([2, 0]))
    assert bc.cost_of(np.array([1, 2])) == 5.0
