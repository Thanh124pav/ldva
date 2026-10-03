"""Effect-label estimators (PLAN.md 3.2, 3.1).

The sign convention is the thing most likely to be silently wrong: every target
must be a *gain* (larger is better), so a sample that helps has a positive
effect. These tests pin that down by calibrating the cheap first-order proxies
against the leave-one-out definition at a learning rate small enough for a
first-order expansion to be valid.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from ldva.supervision.bc_task import BCSupervisionTask
from ldva.supervision.gradient_alignment import (
    GradientAlignmentEstimator,
    OneStepUtilityEstimator,
)
from ldva.supervision.influence import InfluenceEstimator, TRAKEstimator
from ldva.supervision.leave_one_out import LeaveOneOutEstimator

SMALL_LR = 0.01  # small enough that the first-order proxies are valid


@pytest.fixture
def task(store, eval_set, trained_policy):
    policy, ckpts = trained_policy
    vo, va = eval_set
    t = BCSupervisionTask(policy, store, vo, va)
    c = ckpts[1]
    t.set_checkpoint(c.ckpt_id, c.flat_params, c.features)
    return t, ckpts


def _batches(store, n=12, size=6, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.choice(len(store), size=size, replace=False) for _ in range(n)]


def test_all_estimators_return_the_right_shapes(task, store):
    t, _ = task
    ids = _batches(store, n=1)[0]
    for est in (
        GradientAlignmentEstimator("cosine"),
        GradientAlignmentEstimator("dot", lr=SMALL_LR),
        GradientAlignmentEstimator("grad_norm"),
        OneStepUtilityEstimator(lr=SMALL_LR),
        InfluenceEstimator("damping"),
        InfluenceEstimator("lissa", n_iter=5),
        TRAKEstimator(proj_dim=8),
        LeaveOneOutEstimator(lr=SMALL_LR),
    ):
        lab = est.label(t, ids)
        assert lab.per_sample_effects.shape == (len(ids),), est.estimator_id
        assert np.isfinite(lab.per_sample_effects).all(), est.estimator_id
        assert np.isfinite(lab.batch_gain), est.estimator_id
        assert est.cost_tier in ("cheap", "medium", "expensive")


def test_one_step_utility_equals_leave_one_out_batch_gain(task, store):
    """Both are U(Update(theta, B)) - U(theta) with one step, so they must
    agree exactly. A mismatch means one of them mutates the parameters."""
    t, _ = task
    osu = OneStepUtilityEstimator(lr=SMALL_LR)
    loo = LeaveOneOutEstimator(lr=SMALL_LR, n_steps=1)
    for ids in _batches(store, n=4):
        assert osu.batch_gain(t, ids) == pytest.approx(loo.batch_gain(t, ids), abs=1e-9)


def test_gradient_alignment_gain_has_the_same_sign_as_measured_gain(task, store):
    """Sign convention: a step along -g_B changes utility by +lr <g_B, g_val>,
    so aligned gradients must give a POSITIVE gain."""
    t, _ = task
    ga = GradientAlignmentEstimator("dot", lr=SMALL_LR)
    loo = LeaveOneOutEstimator(lr=SMALL_LR, n_steps=1)
    pred, real = [], []
    for ids in _batches(store, n=12):
        pred.append(ga.batch_gain(t, ids))
        real.append(loo.batch_gain(t, ids))
    pred, real = np.array(pred), np.array(real)
    assert spearmanr(pred, real)[0] > 0.9
    assert np.corrcoef(pred, real)[0, 1] > 0.9
    # and they should agree in sign on most batches
    assert np.mean(np.sign(pred) == np.sign(real)) > 0.8


def test_influence_and_trak_agree_in_rank_with_gradient_alignment(task, store):
    """All first-order scores are monotone transforms of <g_i, g_val> under a
    damped-identity Hessian, so their ranks must line up."""
    t, _ = task
    ga = GradientAlignmentEstimator("dot", lr=SMALL_LR)
    infl = InfluenceEstimator("damping")
    ids = _batches(store, n=1, size=10)[0]
    a = ga.sample_effect(t, ids)
    b = infl.sample_effect(t, ids)
    assert spearmanr(a, b)[0] > 0.99


def test_parameters_are_restored_after_labelling(task, store):
    """Estimators that evaluate hypothetical updates must not leave the policy
    perturbed; every later label would otherwise be computed at a drifted theta."""
    t, _ = task
    before = np.concatenate([p.detach().cpu().numpy().ravel() for p in t.parameters()])
    ids = _batches(store, n=1)[0]
    for est in (
        OneStepUtilityEstimator(lr=0.5),
        LeaveOneOutEstimator(lr=0.5, n_steps=3),
        InfluenceEstimator("lissa", n_iter=4),
    ):
        est.label(t, ids)
        after = np.concatenate([p.detach().cpu().numpy().ravel() for p in t.parameters()])
        assert np.allclose(before, after, atol=1e-9), est.estimator_id


def test_leave_one_out_is_genuinely_contextual(task, store):
    """The same sample in different batches must get different effects, and the
    in-batch marginal must differ from the sample's solo effect. If these were
    equal, a scalar per sample would suffice and LDVA would have no premise."""
    t, _ = task
    loo = LeaveOneOutEstimator(lr=0.1, n_steps=2)
    osu = OneStepUtilityEstimator(lr=0.1)
    rng = np.random.default_rng(0)
    target = 3
    marginals, solos = [], []
    for _ in range(8):
        others = rng.choice([i for i in range(len(store)) if i != target], size=5, replace=False)
        ids = np.concatenate([[target], others])
        marginals.append(loo.sample_effect(t, ids)[0])
        solos.append(osu.sample_effect(t, np.array([target]))[0])
    marginals = np.array(marginals)
    assert marginals.std() > 0, "effect of one sample never varies with its company"
    assert np.std(solos) == pytest.approx(0.0, abs=1e-12), "solo effect should be fixed"
    assert not np.allclose(marginals, solos[0], atol=1e-6)


def test_leave_one_out_definition_matches_its_formula(task, store):
    """Delta_i = U(Update(theta, B)) - U(Update(theta, B \\ {i}))."""
    t, _ = task
    loo = LeaveOneOutEstimator(lr=0.1, n_steps=1)
    ids = _batches(store, n=1, size=4)[0]
    effects = loo.sample_effect(t, ids)
    u_full = loo._updated_utility(t, ids)
    for k in range(len(ids)):
        without = np.delete(ids, k)
        assert effects[k] == pytest.approx(u_full - loo._updated_utility(t, without), abs=1e-9)


def test_lissa_falls_back_instead_of_diverging(task, store):
    """A too-small `scale` makes the Neumann series diverge; the estimator must
    detect that and fall back rather than emit infinities."""
    t, _ = task
    est = InfluenceEstimator("lissa", damping=0.01, scale=1e-6, n_iter=30)
    ids = _batches(store, n=1)[0]
    eff = est.sample_effect(t, ids)
    assert np.isfinite(eff).all()


def test_grad_norm_is_context_free_and_non_negative(task, store):
    t, _ = task
    est = GradientAlignmentEstimator("grad_norm")
    ids = _batches(store, n=1, size=5)[0]
    a = est.sample_effect(t, ids)
    b = est.sample_effect(t, ids[::-1])
    assert (a >= 0).all()
    assert np.allclose(a, b[::-1], atol=1e-6)


def test_context_generator_meets_its_coverage_target(supervision):
    """The generator promises `contexts_per_sample`; it must deliver it, not
    just run a fixed number of batches."""
    _, report = supervision
    assert report.target_met, (
        f"only {report.min_contexts_per_sample} contexts for the least-covered sample"
    )
    assert report.min_contexts_per_sample >= 10
