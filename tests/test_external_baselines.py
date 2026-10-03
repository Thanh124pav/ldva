"""Independent external baselines, and the MetaWorld reset-hang guard.

PLAN.md 12 forbids mixing LDVA ablations with published baselines. The first
group of tests holds that line in code rather than in a comment: an external
baseline must produce its allocation without the data model existing at all.

The last group guards a failure that costs hours rather than producing a wrong
number - see `test_metaworld_*_does_not_hang`.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldva.acquisition.directions import AcquisitionDirection
from ldva.acquisition.external_baselines import (
    EXTERNAL_METHODS,
    LDVA_ABLATION_METHODS,
    ProspectiveProxyConfig,
    classify_method,
    direct_gradient_alignment_acquisition,
    direct_gradient_alignment_scores,
    direct_influence_acquisition,
    direct_influence_scores,
    score_directions_by_proxy,
)
from ldva.acquisition.objective import BudgetSpec
from ldva.supervision.bc_task import BCSupervisionTask

# ---- the scores come from real gradients --------------------------------


def test_direct_scores_respond_to_the_real_policy(store, trained_policy, eval_set):
    """The whole point of an independent baseline: the number is measured on
    this policy and this data, with no learned predictor in between."""
    policy, _ = trained_policy
    vo, va = eval_set
    task = BCSupervisionTask(policy, store, vo, va)
    ids = np.arange(min(24, len(store)))

    ga = direct_gradient_alignment_scores(task, ids)
    inf = direct_influence_scores(task, ids, n_lissa=3)
    assert ga.shape == inf.shape == ids.shape
    assert np.isfinite(ga).all() and np.isfinite(inf).all()
    # not all identical: a constant score would mean the gradients were not
    # actually being read
    assert ga.std() > 0
    assert inf.std() > 0


def test_direct_influence_differs_from_gradient_alignment(store, trained_policy, eval_set):
    """If the curvature term changed nothing, influence would be a rescaling of
    gradient alignment and reporting both as separate baselines would be
    misleading."""
    policy, _ = trained_policy
    vo, va = eval_set
    task = BCSupervisionTask(policy, store, vo, va)
    ids = np.arange(min(24, len(store)))

    ga = direct_gradient_alignment_scores(task, ids)
    inf = direct_influence_scores(task, ids, n_lissa=5)
    # compare rankings, since the two live on different scales
    from scipy.stats import spearmanr

    rho = spearmanr(ga, inf).statistic
    assert not np.isnan(rho)
    assert abs(rho) < 0.999, "influence is a monotone rescaling of gradient alignment"


def test_direct_scores_handle_an_empty_id_set(store, trained_policy, eval_set):
    policy, _ = trained_policy
    vo, va = eval_set
    task = BCSupervisionTask(policy, store, vo, va)
    assert direct_gradient_alignment_scores(task, np.array([], dtype=np.int64)).shape == (0,)
    assert direct_influence_scores(task, np.array([], dtype=np.int64)).shape == (0,)


# ---- the proxy is environment-native ------------------------------------


class _FakeMapper:
    """Returns a target offset from the anchors, with no latent model."""

    def __init__(self, shift):
        self.shift = np.asarray(shift, dtype=np.float64)

    def plan_direction(self, direction, n, metadata_all, rng=None):
        rows = np.asarray(metadata_all)[direction.anchor_ids] + self.shift

        class _P:
            metadata = rows

        return _P()


def _direction(did, cluster, anchors, dim=2):
    v = np.zeros(dim)
    v[0] = 1.0
    return AcquisitionDirection(
        direction_id=did, cluster_id=cluster, vector=v, component=0, sign=1,
        anchor_ids=np.asarray(anchors), anchors=np.zeros((len(anchors), dim)),
        delta=0.5,
    )


def test_proxy_prefers_the_direction_near_high_scoring_real_data():
    """A direction pointing at well-scoring real data must outscore one
    pointing at badly-scoring real data."""
    from ldva.data.metadata import MetadataField, MetadataSpec

    spec = MetadataSpec([MetadataField("x", 0.0, 1.0), MetadataField("y", 0.0, 1.0)])
    metadata = np.array([[0.1, 0.1], [0.15, 0.1], [0.9, 0.9], [0.85, 0.9]])
    scores = np.array([10.0, 10.0, -10.0, -10.0])

    good = _direction(0, 0, [0, 1])
    bad = _direction(1, 1, [2, 3])
    out, proxies = score_directions_by_proxy(
        [good, bad], scores, metadata, spec, _FakeMapper([0.0, 0.0]),
        ProspectiveProxyConfig(k_neighbors=2, n_probe=2))
    assert out[0] > out[1]
    assert len(proxies) == 2
    # the report has to say how far the extrapolation reached
    assert "mean_neighbor_distance" in proxies[0].extra


def test_proxy_records_how_far_it_had_to_reach():
    """Where a direction points at genuinely unpopulated metadata, the nearest
    real samples are far away. That is the method's blind spot and must be
    visible in the report, not hidden."""
    from ldva.data.metadata import MetadataField, MetadataSpec

    spec = MetadataSpec([MetadataField("x", 0.0, 1.0), MetadataField("y", 0.0, 1.0)])
    metadata = np.array([[0.05, 0.05], [0.1, 0.05]])
    scores = np.array([1.0, 1.0])
    near = _direction(0, 0, [0, 1])
    far = _direction(1, 0, [0, 1])
    _, p_near = score_directions_by_proxy(
        [near], scores, metadata, spec, _FakeMapper([0.01, 0.0]),
        ProspectiveProxyConfig(k_neighbors=1, n_probe=1))
    _, p_far = score_directions_by_proxy(
        [far], scores, metadata, spec, _FakeMapper([0.8, 0.8]),
        ProspectiveProxyConfig(k_neighbors=1, n_probe=1))
    assert p_far[0].extra["mean_neighbor_distance"] > p_near[0].extra["mean_neighbor_distance"]


def test_classification_keeps_the_two_groups_apart():
    """PLAN.md 12: the `*_style` rules score through the LDVA model and are
    ablations; the `direct_*` rules are independent baselines."""
    for m in EXTERNAL_METHODS:
        assert classify_method(m) == "external"
    for m in LDVA_ABLATION_METHODS:
        assert classify_method(m) == "ldva_ablation"
    assert set(EXTERNAL_METHODS).isdisjoint(LDVA_ABLATION_METHODS)


def test_external_baselines_never_consult_the_data_model(store, trained_policy, eval_set):
    """The strongest form of the PLAN.md 12.2 requirement: the allocation is
    produced with a model that raises on *any* attribute access, so touching it
    would fail the test rather than quietly borrowing a prediction.
    """
    from ldva.data.metadata import MetadataField, MetadataSpec

    policy, _ = trained_policy
    vo, va = eval_set
    task = BCSupervisionTask(policy, store, vo, va)

    class _Explode:
        def __getattr__(self, name):
            raise AssertionError(
                f"an external baseline consulted the LDVA model (.{name}); "
                "PLAN.md 12.2 forbids borrowing LDVA predictions"
            )

    class _Objective:
        """Only the attributes a baseline is licensed to read."""

        def __init__(self, directions, budget):
            self.directions = directions
            self.budget = budget
            self.model = _Explode()
            self.n_directions = len(directions)

        def predict(self, alloc):
            from ldva.acquisition.objective import AllocationValue

            return AllocationValue(alloc, 0.0, 0.0, 0, 0.0, {})

    spec = MetadataSpec([
        MetadataField(f"m{i}", float(store.metadata[:, i].min()),
                      float(store.metadata[:, i].max()) + 1e-6)
        for i in range(store.metadata.shape[1])
    ])
    dim = store.metadata.shape[1]
    dirs = [_direction(0, 0, [0, 1], dim), _direction(1, 0, [2, 3], dim)]
    budget = BudgetSpec(budget=4)
    obj = _Objective(dirs, budget)
    cfg = ProspectiveProxyConfig(k_neighbors=3, n_probe=2)

    for fn in (direct_gradient_alignment_acquisition, direct_influence_acquisition):
        res = fn(obj, task, store.metadata, spec, _FakeMapper(np.zeros(dim)),
                 sample_ids=np.arange(min(16, len(store))), cfg=cfg)
        assert int(res.best_allocation.sum()) == budget.budget
        assert res.info["independent_of_ldva_model"] is True


# ---- MetaWorld: the reset hang ------------------------------------------

metaworld = pytest.importorskip("metaworld")


@pytest.fixture(scope="module")
def mw():
    from ldva.envs.metaworld.adapter import MetaWorldAdapter, MetaWorldConfig

    return MetaWorldAdapter(MetaWorldConfig(tasks=("push-v3",), seed=0, max_steps=30))


def test_metaworld_repairs_the_object_goal_separation(mw):
    """reach/push/pick-place contain

        while ||obj_xy - goal_xy|| < 0.15: rand_vec = _get_state_rand_vec()

    and under `_freeze_rand_vec` that call returns the same vector forever, so
    `env.reset()` spins without ever raising. Every pinned vector must
    therefore come out of `_pin` already satisfying the constraint.
    """
    from ldva.envs.metaworld.adapter import _MIN_OBJ_GOAL_SEP

    # object and goal 0.10 apart: inside the forbidden disc
    bad = np.array([0.0, 0.70, 0.02, 0.0, 0.80, 0.015])
    fixed = mw._repair_separation("push-v3", bad)
    assert np.linalg.norm(fixed[0:2] - fixed[3:5]) >= _MIN_OBJ_GOAL_SEP

    # an already-feasible request must be left alone
    ok = np.array([0.0, 0.60, 0.02, 0.0, 0.90, 0.015])
    np.testing.assert_allclose(mw._repair_separation("push-v3", ok), ok)


def test_metaworld_repairs_exact_overlap(mw):
    """Coincident object and goal have no outward direction to slide along."""
    from ldva.envs.metaworld.adapter import _MIN_OBJ_GOAL_SEP

    same = np.array([0.0, 0.65, 0.02, 0.0, 0.65, 0.015])
    fixed = mw._repair_separation("push-v3", same)
    assert np.linalg.norm(fixed[0:2] - fixed[3:5]) >= _MIN_OBJ_GOAL_SEP


def test_metaworld_collect_does_not_hang_on_a_violating_request(mw):
    """The end-to-end guard. Without the repair this call never returns."""
    bad = np.array([[0.0, 0.70, 0.02, 0.0, 0.80, 0.015]])
    store = mw.collect(bad, np.random.default_rng(0), round_id=1)
    assert len(store) >= 1
    from ldva.envs.metaworld.adapter import _MIN_OBJ_GOAL_SEP

    sep = np.linalg.norm(store.metadata[0, 0:2] - store.metadata[0, 3:5])
    # the RECORDED metadata is the repaired one, because the mapper must learn
    # the map that actually holds
    assert sep >= _MIN_OBJ_GOAL_SEP


def test_metaworld_eval_conditions_are_all_feasible(mw):
    """Fixed conditions must be feasible as declared; one repaired at run time
    would no longer match its own fingerprint."""
    from ldva.envs.metaworld.adapter import _MIN_OBJ_GOAL_SEP

    cond = mw.eval_conditions(24, np.random.default_rng(3))
    sep = np.linalg.norm(cond.metadata[:, 0:2] - cond.metadata[:, 3:5], axis=1)
    assert (sep >= _MIN_OBJ_GOAL_SEP).all()


def test_metaworld_rollout_eval_separates_expert_from_untrained(mw):
    from ldva.policy.bc import MLPPolicy

    cond = mw.eval_conditions(3, np.random.default_rng(0))
    untrained = mw.evaluate_policy(
        MLPPolicy(mw._obs_dim, mw._act_dim, hidden=(32,)), cond)
    expert = mw.expert_reference(cond)
    assert expert.mean_return > untrained.mean_return
    assert untrained.info["metric"] == "metaworld_success"
