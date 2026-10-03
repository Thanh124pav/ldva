"""DMC and MetaWorld adapters (PLAN.md 15, 14).

These run against real simulators, so they are kept small and are skipped
cleanly when the package is absent - the LDVA core must stay testable without
any simulator installed.

The property that matters most is the one that broke directional acquisition
when it was violated: **the declared metadata box must be the environment's
actual feasible set**, so a request inside the box comes back unchanged. If the
adapter silently clips, the local linear metadata map acquires a kink that no
Jacobian can represent and the planner's requests land somewhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

dm_control = pytest.importorskip("dm_control", reason="dm_control not installed")


@pytest.fixture(scope="module")
def dmc():
    from ldva.envs.dmc.adapter import DMCAdapter, DMCConfig

    return DMCAdapter(DMCConfig(task="reacher-easy", chunk_len=8, max_steps=60,
                                max_chunks_per_episode=2, seed=0))


def test_dmc_requests_are_realized_exactly(dmc):
    """No clipping: realized metadata equals the request, to the bit."""
    rng = np.random.default_rng(0)
    req = dmc.metadata_spec.sample(12, rng)
    store = dmc.collect(req, rng)
    realized = np.array(
        [store.metadata[store.trajectory_id == i][0] for i in range(len(req))])
    assert np.abs(realized - req).max() < 1e-9


def test_dmc_metadata_box_avoids_unreachable_and_seam(dmc):
    """Radius inside the arm's reach, angle short of a full turn."""
    lo, hi = dmc.metadata_spec.low, dmc.metadata_spec.high
    assert dmc.metadata_spec.names == ["target_radius", "target_angle"]
    assert lo[0] > 0.0 and hi[0] < 0.24, "radius must stay inside L1 + L2"
    assert hi[1] < np.pi and lo[1] > -np.pi, "angle must not include the +/-pi seam"


def test_dmc_collect_contract(dmc):
    rng = np.random.default_rng(0)
    req = dmc.metadata_spec.sample(5, rng)
    store = dmc.collect(req, rng, round_id=3)
    assert len(np.unique(store.trajectory_id)) == 5
    assert (store.round_id == 3).all()
    assert store.obs.shape[1] == dmc.cfg.chunk_len
    assert np.isfinite(store.obs).all() and np.isfinite(store.act).all()


def test_dmc_scripted_expert_actually_solves_the_task(dmc):
    """BC on random actions would be meaningless, so the expert must succeed."""
    rng = np.random.default_rng(0)
    store = dmc.collect(dmc.metadata_spec.sample(6, rng), rng)
    assert store.success.mean() > 0.5
    assert store.reward.mean() > 0.0


def test_dmc_initial_dataset_is_a_corner_of_the_box(dmc):
    rng = np.random.default_rng(0)
    d0 = dmc.initial_dataset(12, rng)
    mn = dmc.metadata_spec.normalize(d0.metadata)
    assert mn.max() <= dmc.cfg.initial_corner_frac + 1e-6
    # the evaluation distribution spans the whole box, so coverage is incomplete
    assert mn.max(0).min() < 0.5


def test_dmc_unregistered_task_refuses_with_instructions():
    from ldva.envs.dmc.adapter import DMCAdapter, DMCConfig

    with pytest.raises(NotImplementedError) as e:
        DMCAdapter(DMCConfig(task="cheetah-run"))
    msg = str(e.value)
    assert "no state setter or expert controller" in msg
    assert "setter(" in msg and "expert(" in msg


def test_dmc_data_trains_a_bc_policy_without_diverging(dmc):
    """Real observations are badly scaled; the policy must normalize them.

    Unnormalized, an MLP[128,128] on DMC reacher diverges to NaN under the same
    SGD settings that work on the synthetic world.
    """
    import torch

    from ldva.policy.train import BCTrainConfig, train_bc

    rng = np.random.default_rng(0)
    store = dmc.initial_dataset(14, rng)
    vo, va = dmc.evaluation_set(30, rng)
    policy, ckpts = train_bc(
        store, vo, va,
        BCTrainConfig(steps=80, snapshot_every=40, n_restarts=2),
        hidden=(128, 128), seed=0)
    with torch.no_grad():
        loss = float(policy.bc_loss(vo, va).item())
    assert np.isfinite(loss), "BC diverged on real observations"
    assert policy.normalize_obs
    assert (policy.obs_scale > 0).all()
    assert len(ckpts) >= 2


def test_obs_normalizer_is_not_part_of_the_parameter_vector(dmc):
    """Normalization lives in buffers, so effect estimators see clean params."""
    from ldva.policy.bc import MLPPolicy

    p = MLPPolicy(6, 2, hidden=(16,))
    n_before = p.flat_params().numel()
    p.fit_obs_normalizer(np.random.default_rng(0).normal(size=(100, 6)) * 5 + 3)
    assert p.flat_params().numel() == n_before
    assert not np.allclose(p.obs_mean.numpy(), 0.0)
    # and a round trip through flat params leaves the normalizer untouched
    mean_before = p.obs_mean.clone()
    p.load_flat_params(p.flat_params())
    assert np.allclose(p.obs_mean.numpy(), mean_before.numpy())


# ---- MetaWorld -----------------------------------------------------------


@pytest.fixture(scope="module")
def mw():
    pytest.importorskip("metaworld", reason="metaworld not installed")
    from ldva.envs.metaworld.adapter import MetaWorldAdapter, MetaWorldConfig

    return MetaWorldAdapter(MetaWorldConfig(
        tasks=("reach-v3", "push-v3"), chunk_len=8, max_steps=80,
        max_chunks_per_episode=2, seed=0))


def test_metaworld_uses_v3_tasks_and_declares_the_deviation(mw):
    """metaworld 3.1.1 has no v2 environments, so PLAN.md 14's names shift."""
    rep = mw.report()
    assert all(t.endswith("-v3") for t in rep["tasks"])
    assert "v3" in rep["note"]


def test_metaworld_metadata_includes_task_and_poses(mw):
    names = mw.metadata_spec.names
    assert names[0] == "task_id"
    assert "object_x" in names and "goal_x" in names
    # a coordinate no task randomizes must be pinned, not pretend-controllable
    mask = mw.metadata_spec.controllable_mask
    assert mask[names.index("object_x")]
    assert not mask[names.index("object_z")]


def test_metaworld_collect_contract_and_expert_success(mw):
    rng = np.random.default_rng(0)
    req = mw.metadata_spec.sample(4, rng)
    store = mw.collect(req, rng, round_id=2)
    assert len(np.unique(store.trajectory_id)) == 4
    assert (store.round_id == 2).all()
    assert store.obs.shape[2] == 39 and store.act.shape[2] == 4
    assert store.success.mean() > 0.5, "scripted experts should solve these tasks"
    assert mw.metadata_spec.is_feasible(store.metadata).all()


def test_metaworld_realized_metadata_is_clipped_per_task(mw):
    """Tasks have different reset boxes, so a union-box request is clipped to
    the task's own set - and what gets *recorded* must be the clipped value."""
    rng = np.random.default_rng(0)
    # ask for the extreme corner of the union box for every task
    rows = []
    for i in range(len(mw.cfg.tasks)):
        row = mw.metadata_spec.high.copy()
        row[0] = i
        rows.append(row)
    store = mw.collect(np.array(rows), rng)
    for i in range(len(mw.cfg.tasks)):
        got = store.metadata[store.trajectory_id == i][0]
        assert mw.metadata_spec.is_feasible(got)
        t = mw.cfg.tasks[int(round(got[0]))]
        lo, hi = mw._task_box[t]
        n = mw._rand_dim[t]
        assert np.all(got[1 : 1 + n] <= hi[:n] + 1e-9)
        assert np.all(got[1 : 1 + n] >= lo[:n] - 1e-9)


def test_metaworld_initial_dataset_has_incomplete_coverage(mw):
    rng = np.random.default_rng(0)
    d0 = mw.initial_dataset(10, rng)
    mn = mw.metadata_spec.normalize(d0.metadata)
    ctrl = mw.metadata_spec.controllable_mask.copy()
    ctrl[0] = False  # task id is meant to span all tasks
    assert mn[:, ctrl].max() < 0.75, "D_0 should not already span the box"


# ---- configs must describe the adapters ----------------------------------


def test_dmc_config_matches_the_adapter(dmc):
    """A config that drifts from its adapter silently misdescribes the
    experiment, so the declared metadata must match field for field."""
    from ldva.data.metadata import MetadataSpec
    from ldva.utils import load_config

    cfg = MetadataSpec.from_config(load_config("configs/env/dmc.yaml")["metadata"])
    assert cfg.names == dmc.metadata_spec.names
    assert np.allclose(cfg.low, dmc.metadata_spec.low, atol=1e-6)
    assert np.allclose(cfg.high, dmc.metadata_spec.high, atol=1e-6)


def test_metaworld_config_matches_the_adapter():
    pytest.importorskip("metaworld", reason="metaworld not installed")
    from ldva.data.metadata import MetadataSpec
    from ldva.envs.metaworld.adapter import MetaWorldAdapter, MetaWorldConfig
    from ldva.utils import load_config

    file_cfg = load_config("configs/env/metaworld.yaml")
    a = MetaWorldAdapter(MetaWorldConfig(tasks=tuple(file_cfg["tasks"]), seed=0))
    spec = MetadataSpec.from_config(file_cfg["metadata"])
    assert spec.names == a.metadata_spec.names
    assert np.allclose(spec.low, a.metadata_spec.low, atol=1e-6)
    assert np.allclose(spec.high, a.metadata_spec.high, atol=1e-6)
    assert spec.controllable_mask.tolist() == a.metadata_spec.controllable_mask.tolist()
