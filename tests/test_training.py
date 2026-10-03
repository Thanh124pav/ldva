"""Losses, trainer plumbing and baselines (PLAN.md 5; SETUP.md 20, 22)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ldva.training import losses as L
from ldva.training.metrics import regression_metrics
from ldva.training.train_datamodel import DataModelTrainer, TrainConfig


def test_masked_mse_ignores_padded_slots():
    pred = torch.tensor([[1.0, 99.0], [2.0, 99.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, False], [True, False]])
    # only the first column counts: errors 0 and 2 -> mean of (0, 4) = 2
    assert L.masked_mse(pred, target, mask).item() == pytest.approx(2.0)


def test_effect_loss_is_zero_on_perfect_predictions():
    t = torch.randn(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert L.effect_loss(t.clone(), t, mask).item() == pytest.approx(0.0)


def test_metric_loss_matches_effect_distances_and_counts_pairs():
    """L_metric regresses ||z_i - z_j|| onto the normalized effect distance."""
    z = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]])  # distances 5, 0, 5
    ids = torch.tensor([[0, 1, 2]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    def lookup(pairs):
        # claim every pair is at distance 5 -> loss is |5-5|, |0-5|, |5-5|
        return np.full(len(pairs), 5.0), np.ones(len(pairs), dtype=bool)

    loss, n = L.metric_loss(z, ids, mask, lookup, rng=np.random.default_rng(0))
    assert n == 3
    assert loss.item() == pytest.approx(5.0 / 3.0, abs=1e-5)


def test_metric_loss_skips_cleanly_when_no_pair_is_known():
    z = torch.randn(2, 4, 3)
    ids = torch.arange(8).reshape(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss, n = L.metric_loss(
        z, ids, mask,
        lambda pairs: (np.zeros(len(pairs)), np.zeros(len(pairs), dtype=bool)),
        rng=np.random.default_rng(0))
    assert n == 0 and loss.item() == 0.0


def test_metric_loss_is_differentiable():
    z = torch.randn(1, 4, 3, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    loss, n = L.metric_loss(
        z, ids, mask,
        lambda p: (np.ones(len(p)), np.ones(len(p), dtype=bool)),
        rng=np.random.default_rng(0))
    loss.backward()
    assert n > 0 and z.grad is not None and torch.isfinite(z.grad).all()


def test_smoothness_loss_penalizes_only_shared_members():
    a = torch.tensor([[1.0, 5.0]])
    b = torch.tensor([[1.5, 99.0]])
    shared = torch.tensor([[True, False]])
    assert L.smoothness_loss(a, b, shared).item() == pytest.approx(0.5)


def test_metadata_direction_loss_is_zero_for_a_perfect_map():
    from ldva.models.metadata_direction import (
        MetadataDirectionConfig,
        MetadataDirectionModel,
    )

    m = MetadataDirectionModel(MetadataDirectionConfig(latent_dim=3, meta_dim=2,
                                                       hidden=(8,)))
    z_i = torch.randn(4, 3)
    m_i = torch.randn(4, 2)
    dm = torch.randn(4, 2) * 0.01
    # construct z_j so the model's own prediction is exact
    with torch.no_grad():
        z_j = z_i + m(z_i, m_i, dm)
    assert L.metadata_direction_loss(m, z_i, z_j, m_i, m_i + dm).item() == pytest.approx(
        0.0, abs=1e-10)


def test_loss_weights_roundtrip():
    w = L.LossWeights(1.0, 0.5, 0.1, 0.01, 0.0)
    assert w.as_dict()["batch"] == 0.5
    assert set(w.as_dict()) == {"effect", "batch", "metric", "smooth", "meta"}


def test_regression_metrics_r2_is_negative_for_a_bad_model():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    worse_than_mean = np.array([4.0, 3.0, 2.0, 1.0])
    m = regression_metrics(worse_than_mean, target)
    assert m["r2"] < 0
    assert m["spearman"] == pytest.approx(-1.0)


def test_scalar_baseline_is_fitted_out_of_sample_in_the_trainer(context_dataset, datamodel):
    """The per-sample scalar baseline must be estimated on train contexts, or it
    memorizes the very targets it is scored against."""
    tr, va = context_dataset.split(0.3, seed=0, by="context")
    trainer = DataModelTrainer(datamodel, tr, va, TrainConfig(epochs=1, eval_every=1))
    out = trainer.evaluate(va)
    assert out["effect_scalar_out_of_sample"] is True
    assert out["effect_scalar_hindsight_out_of_sample"] is False
    # hindsight must look better; that is exactly why it is unfair
    assert out["effect_scalar_hindsight_mse"] <= out["effect_scalar_mse"] + 1e-9


def test_trainer_reports_within_checkpoint_gain_metrics(context_dataset, datamodel):
    tr, va = context_dataset.split(0.3, seed=0, by="context")
    trainer = DataModelTrainer(datamodel, tr, va, TrainConfig(epochs=1, eval_every=1))
    out = trainer.evaluate(va)
    for k in ("gain_within_mse", "gain_within_r2", "gain_var_within_group_share"):
        assert k in out, k
    assert 0.0 <= out["gain_var_within_group_share"] <= 1.0


def test_training_reduces_the_loss(context_dataset, store):
    """A short run must actually learn something, else the plumbing is broken."""
    from ldva.models.datamodel import LDVAConfig, LDVADataModel

    ds = context_dataset
    tr, va = ds.split(0.3, seed=0, by="context")
    model = LDVADataModel(LDVAConfig.build(
        obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
        meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
        n_checkpoints=ds.n_checkpoints, latent_dim=8, hidden=(32, 32)))
    model.set_dataset_context(np.zeros((4, 8)))
    trainer = DataModelTrainer(model, tr, va,
                               TrainConfig(epochs=12, eval_every=12, lr=3e-3, seed=0))
    hist = trainer.fit()
    assert hist[-1]["train/loss"] < hist[0]["train/loss"]


def test_baseline_suite_produces_valid_allocations(store, datamodel, trained_policy, world):
    """SETUP.md 20: every baseline must spend the same budget over the same
    candidate directions, so the comparison isolates the allocation rule."""
    from ldva.acquisition.baselines import BaselineSuite
    from ldva.acquisition.clustering import ClusteringConfig, LatentClustering
    from ldva.acquisition.directions import DirectionConfig, DirectionGenerator
    from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig
    from ldva.acquisition.objective import (
        AllocationObjective,
        BudgetSpec,
        ObjectiveConfig,
    )

    _, ckpts = trained_policy
    ref = ckpts[1]
    z = datamodel.encode_store(store, ref.features)
    clusters = LatentClustering(ClusteringConfig(n_clusters=3, seed=0)).fit(z, store.metadata)
    dirs = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    sampler = LatentSampler(clusters, LatentSamplerConfig(seed=0))
    budget = BudgetSpec.from_directions(dirs, budget=5)
    obj = AllocationObjective(datamodel, sampler, dirs, budget,
                              policy_features=ref.features,
                              cfg=ObjectiveConfig(n_mc=3, seed=0))
    results = BaselineSuite(z_support=z, n_draw=6, seed=0).run(obj, datamodel)
    assert len(results) == 9
    for name, res in results.items():
        assert res.best_allocation.sum() == 5, name
        assert (res.best_allocation >= 0).all(), name
        assert np.isfinite(res.best_value), name
