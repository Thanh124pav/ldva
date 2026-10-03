"""Rollout evaluation and the P0 fixes of PLAN.md 15.

These guard the four things that changed what a number in the report *means*:
real rollout utility, an environment-agnostic loop, baselines that do not
borrow LDVA predictions, and a policy context that cannot disagree with itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldva.envs.base import get_adapter
from ldva.envs.rollout import EvalConditions, PolicyActor, RolloutMetrics
from ldva.policy.bc import MLPPolicy
from ldva.policy.checkpoints import Checkpoint, PolicyContextRef

dm_control = pytest.importorskip("dm_control")


# ---- EvalConditions ------------------------------------------------------


def test_eval_conditions_fingerprint_is_stable_and_discriminating():
    """PLAN.md 15 requires the SAME conditions across methods and rounds.

    The fingerprint is what makes a silent re-draw visible in the report, so it
    must be identical for identical conditions and differ for any change.
    """
    m = np.array([[0.1, 0.2], [0.3, 0.4]])
    a = EvalConditions(m, seed=0, env_name="dmc")
    b = EvalConditions(m.copy(), seed=0, env_name="dmc")
    assert a.fingerprint() == b.fingerprint()

    assert EvalConditions(m + 1e-3, 0, "dmc").fingerprint() != a.fingerprint()
    # the environment is part of the identity: the same numbers mean different
    # conditions in different simulators
    assert EvalConditions(m, 0, "metaworld").fingerprint() != a.fingerprint()


def test_eval_conditions_subset_is_a_prefix_not_a_resample():
    """A cheaper evaluation must be a subset of the expensive one, or the two
    are not comparable."""
    m = np.arange(20, dtype=np.float64).reshape(10, 2)
    c = EvalConditions(m, seed=0, env_name="dmc")
    s = c.subset(4)
    assert len(s) == 4
    np.testing.assert_array_equal(s.metadata, m[:4])


def test_rollout_metrics_keep_per_episode_values():
    """Aggregating across seeds needs the episodes, not an average of
    averages."""
    r = RolloutMetrics.from_episodes([1.0, 3.0], [True, False], [10, 20])
    assert r.mean_return == pytest.approx(2.0)
    assert r.success_rate == pytest.approx(0.5)
    assert r.returns == [1.0, 3.0]
    assert r.n_episodes == 2
    assert r.standard_error == pytest.approx(1.0 / np.sqrt(2))


def test_rollout_metrics_tolerate_no_episodes():
    r = RolloutMetrics.from_episodes([], [], [])
    assert r.n_episodes == 0
    assert np.isnan(r.mean_return)


# ---- the adapter contract ------------------------------------------------


def test_adapters_without_a_simulator_refuse_to_fake_a_rollout():
    """PLAN.md 18 does not accept the BC proxy as a robotics outcome, so an
    adapter that cannot roll out must raise rather than return a proxy that
    looks like one."""
    a = get_adapter("synthetic", seed=0)
    assert a.supports_rollout_eval is False
    policy = MLPPolicy(4, 2)
    with pytest.raises(NotImplementedError, match="rollout"):
        a.evaluate_policy(policy, EvalConditions(np.zeros((2, 3)), 0, "synthetic"))


def test_dmc_rollout_separates_an_expert_from_an_untrained_policy():
    """The headline metric has to have dynamic range, or no acquisition curve
    measured on it can be read."""
    ad = get_adapter("dmc", task="reacher-easy", seed=0, max_steps=60)
    assert ad.supports_rollout_eval is True
    cond = ad.eval_conditions(4, np.random.default_rng(0))

    untrained = ad.evaluate_policy(
        MLPPolicy(ad._obs_dim, ad._act_dim, hidden=(32,)), cond)
    expert = ad.expert_reference(cond)

    assert expert.mean_return > untrained.mean_return
    assert expert.success_rate >= untrained.success_rate
    assert untrained.n_episodes == expert.n_episodes == 4
    assert untrained.info["metric"] == "dmc_return"


def test_dmc_rollout_is_deterministic_for_fixed_conditions():
    """Two evaluations of the same policy on the same conditions must agree, or
    a round-to-round difference cannot be attributed to the data."""
    ad = get_adapter("dmc", task="reacher-easy", seed=0, max_steps=40)
    cond = ad.eval_conditions(3, np.random.default_rng(1))
    pol = MLPPolicy(ad._obs_dim, ad._act_dim, hidden=(16,))
    a = ad.evaluate_policy(pol, cond)
    b = ad.evaluate_policy(pol, cond)
    assert a.returns == pytest.approx(b.returns)


def test_policy_actor_restores_training_mode():
    pol = MLPPolicy(4, 2)
    pol.train()
    with PolicyActor(pol) as actor:
        assert pol.training is False
        assert actor(np.zeros(4)).shape == (2,)
    assert pol.training is True


# ---- P0.4: the policy context ------------------------------------------


def test_policy_context_ref_drops_the_ckpt_id_by_default():
    """PLAN.md 4.2: the main representation is the continuous features alone."""
    c = Checkpoint(ckpt_id="r0_s50", flat_params=np.zeros(3), step=50,
                   features=np.array([0.5, 0.1, 0.2, 0.3]))
    ref = PolicyContextRef.from_checkpoint(c, ["r0_s0", "r0_s50"])
    assert ref.ckpt_index is None and ref.is_seen is False
    abl = PolicyContextRef.from_checkpoint(c, ["r0_s0", "r0_s50"], use_ckpt_id=True)
    assert abl.ckpt_index == 1 and abl.is_seen is True


def test_policy_context_ref_has_no_index_for_an_unseen_checkpoint():
    """An unseen future policy has no vocabulary index; claiming one would be
    the memorization PLAN.md 4.2 warns about."""
    c = Checkpoint(ckpt_id="r9_s999", flat_params=np.zeros(3), step=999,
                   features=np.zeros(4))
    ref = PolicyContextRef.from_checkpoint(c, ["r0_s0"], use_ckpt_id=True)
    assert ref.ckpt_index is None


def test_objective_refuses_two_sources_of_policy_conditioning():
    """The P0.4 bug was features from one checkpoint combined with the ID
    embedding of another. Two sources for one thing is how that happened, so
    passing both is refused."""
    from ldva.acquisition.objective import AllocationObjective

    ref = PolicyContextRef(features=np.zeros(4, np.float32), ckpt_id="c")
    with pytest.raises(ValueError, match="not both"):
        AllocationObjective(
            model=None, sampler=None, directions=[object()], budget=None,
            policy_features=np.zeros(4), policy_context=ref)


def test_objective_does_not_default_the_checkpoint_index_to_zero():
    """A silent `ckpt_index=0` is exactly the P0.4 bug: it conditions on
    checkpoint 0 no matter whose features were passed."""
    from ldva.acquisition.objective import AllocationObjective

    obj = AllocationObjective(
        model=None, sampler=None, directions=[object()], budget=None,
        policy_features=np.zeros(4))
    assert obj.ckpt_index is None


def test_policy_context_encoder_handles_a_missing_index():
    """With an embedding configured, `in_dim` is fixed - so an unseen
    checkpoint would crash the ablated model unless the slot is zero-filled.
    Held-out-checkpoint evaluation is the whole point of the ablation, so it
    has to be possible to run."""
    from ldva.models.datamodel import PolicyContextConfig, PolicyContextEncoder

    enc = PolicyContextEncoder(PolicyContextConfig(
        policy_feat_dim=4, n_checkpoints=5, embed_dim=8, out_dim=16))
    pf = torch.zeros(3, 4)
    with_idx = enc(pf, torch.zeros(3, dtype=torch.long))
    without = enc(pf, None)
    assert with_idx.shape == without.shape == (3, 16)
    assert enc.uses_ckpt_id is True

    plain = PolicyContextEncoder(PolicyContextConfig(
        policy_feat_dim=4, n_checkpoints=0, out_dim=16))
    assert plain.uses_ckpt_id is False
    assert plain(pf, None).shape == (3, 16)


def test_context_dataset_can_hold_out_whole_checkpoints(context_dataset):
    """PLAN.md 4.2: "hold out entire policy checkpoints", not only unseen batch
    compositions."""
    ds = context_dataset
    tr, va = ds.split(0.3, seed=0, by="checkpoint")
    assert set(tr.checkpoint_ids).isdisjoint(set(va.checkpoint_ids))
    assert len(va.records) > 0
