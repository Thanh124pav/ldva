"""The environment seam (SETUP.md 37).

Stages 1-6 are not wired up. What these tests guarantee is that the seam itself
is correct: the synthetic adapter satisfies the contract, and the unimplemented
stages fail loudly at the integration point with instructions rather than
silently producing wrong data deeper in the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldva.envs import available_adapters, get_adapter
from ldva.envs.base import EnvAdapter, NotImplementedAdapter


def test_registry_lists_every_stage():
    av = available_adapters()
    assert set(av) == {"synthetic", "dmc", "pusht", "metaworld", "maniskill"}
    # implemented stages
    assert av["synthetic"] is True
    assert av["dmc"] is True
    assert av["metaworld"] is True
    # not wired up yet, and honestly reported as such
    assert not any(av[k] for k in ("pusht", "maniskill"))


def test_unknown_adapter_raises():
    with pytest.raises(KeyError, match="unknown env"):
        get_adapter("does_not_exist")


@pytest.mark.parametrize("name", ["pusht", "maniskill"])
def test_unimplemented_adapters_explain_themselves(name):
    a = get_adapter(name)
    assert isinstance(a, NotImplementedAdapter)
    with pytest.raises(NotImplementedError) as e:
        a.metadata_spec
    msg = str(e.value)
    assert "SETUP.md section" in msg
    assert "To implement this adapter" in msg
    assert a.policy_defaults()["kind"] in ("mlp_bc", "sac", "ppo")


def test_synthetic_adapter_satisfies_the_contract():
    a = get_adapter("synthetic", seed=0)
    assert isinstance(a, EnvAdapter)
    rng = np.random.default_rng(0)
    spec = a.metadata_spec
    assert len(spec) >= 1

    d0 = a.initial_dataset(30, rng)
    assert len(d0) == 30
    assert (d0.round_id == 0).all()
    assert spec.is_feasible(d0.metadata).all()

    # one acquisition request per row, one distinct trajectory per request
    req = spec.sample(7, rng)
    new = a.collect(req, rng, round_id=2)
    assert len(new) >= 7
    assert len(np.unique(new.trajectory_id)) == 7
    assert (new.round_id == 2).all()
    # the synthetic world honours requests exactly, so realized == requested
    assert np.allclose(new.metadata, req)

    vo, va = a.evaluation_set(25, rng)
    assert vo.shape[0] == va.shape[0] == 25
    assert vo.shape[1] == d0.chunk_len

    assert a.acquisition_cost(req).shape == (7,)


def test_synthetic_initial_dataset_has_incomplete_coverage():
    """If D_0 already spans the metadata box, directional acquisition has
    nowhere to expand and the experiment is vacuous."""
    a = get_adapter("synthetic", seed=0)
    rng = np.random.default_rng(0)
    d0 = a.initial_dataset(200, rng)
    mn = a.metadata_spec.normalize(d0.metadata)
    span = mn.max(0) - mn.min(0)
    assert np.any(span < 0.7)


def test_collected_data_is_usable_by_the_rest_of_the_pipeline():
    """A store from collect() must drop straight into a supervision task."""
    from ldva.policy.bc import MLPPolicy
    from ldva.supervision.bc_task import BCSupervisionTask
    from ldva.supervision.leave_one_out import LeaveOneOutEstimator

    a = get_adapter("synthetic", seed=0)
    rng = np.random.default_rng(0)
    store = a.initial_dataset(20, rng)
    vo, va = a.evaluation_set(20, rng)
    policy = MLPPolicy(store.obs_dim, store.act_dim)
    task = BCSupervisionTask(policy, store, vo, va)
    labels = LeaveOneOutEstimator(lr=0.1, n_steps=1).label(task, np.arange(5))
    assert labels.per_sample_effects.shape == (5,)
    assert np.isfinite(labels.batch_gain)


def test_store_concat_models_an_acquisition_round():
    a = get_adapter("synthetic", seed=0)
    rng = np.random.default_rng(0)
    d0 = a.initial_dataset(20, rng)
    d1 = d0.concat(a.collect(a.metadata_spec.sample(6, rng), rng, round_id=1))
    assert len(d1) == 26
    assert (d1.round_id[20:] == 1).all()
    assert d1.metadata_spec.names == d0.metadata_spec.names
