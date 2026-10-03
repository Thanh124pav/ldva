"""Shared fixtures. Everything here is sized to run on CPU in seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldva.data.context_dataset import ContextDataset  # noqa: E402
from ldva.data.effect_profiles import EffectProfileTable  # noqa: E402
from ldva.envs.synthetic.generator import SyntheticConfig, SyntheticWorld  # noqa: E402
from ldva.models.datamodel import LDVAConfig, LDVADataModel  # noqa: E402
from ldva.policy.train import BCTrainConfig, train_bc  # noqa: E402
from ldva.supervision.bc_task import BCSupervisionTask  # noqa: E402
from ldva.supervision.generate_context_records import (  # noqa: E402
    ContextGenConfig,
    generate_context_records,
)
from ldva.supervision.leave_one_out import LeaveOneOutEstimator  # noqa: E402


@pytest.fixture(scope="session")
def world() -> SyntheticWorld:
    return SyntheticWorld(SyntheticConfig(seed=0))


@pytest.fixture(scope="session")
def eval_set(world):
    m = world.evaluation_metadata(120, np.random.default_rng(555))
    obs, act, _ = world.generate_chunks(m, np.random.default_rng(556))
    return torch.from_numpy(obs), torch.from_numpy(act)


@pytest.fixture(scope="session")
def store(world):
    return world.build_store(
        60, np.random.default_rng(1), regions=world.default_initial_regions()
    )


@pytest.fixture(scope="session")
def trained_policy(store, eval_set):
    vo, va = eval_set
    return train_bc(
        store, vo, va,
        BCTrainConfig(steps=60, snapshot_every=30, n_restarts=2),
        seed=0,
    )


@pytest.fixture(scope="session")
def supervision(store, eval_set, trained_policy):
    """A small but genuinely multi-context record set."""
    policy, ckpts = trained_policy
    vo, va = eval_set
    task = BCSupervisionTask(policy, store, vo, va)
    records, report = generate_context_records(
        task, ckpts, LeaveOneOutEstimator(lr=0.1, n_steps=2), len(store),
        ContextGenConfig(contexts_per_sample=10, batch_size=6, seed=0),
    )
    return records, report


@pytest.fixture(scope="session")
def context_dataset(store, supervision):
    records, _ = supervision
    return ContextDataset(store, records)


@pytest.fixture(scope="session")
def effect_table(store, supervision):
    records, _ = supervision
    return EffectProfileTable(records, len(store), min_shared=2)


@pytest.fixture(scope="session")
def datamodel(store, context_dataset):
    ds = context_dataset
    model = LDVADataModel(LDVAConfig.build(
        obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
        meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
        n_checkpoints=ds.n_checkpoints, latent_dim=8, hidden=(32, 32)))
    model.set_dataset_context(np.zeros((4, 8)))
    return model


@pytest.fixture
def latent_blobs():
    """Anisotropic, well-separated latent clusters."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=4.0, size=(3, 5))
    pts = []
    for c in centers:
        ax = rng.normal(size=5)
        ax /= np.linalg.norm(ax)
        pts.append(c + 1.0 * np.outer(rng.normal(size=80), ax)
                   + rng.normal(scale=0.12, size=(80, 5)))
    return np.concatenate(pts), centers
