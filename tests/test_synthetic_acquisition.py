"""Phase 0 deliverable (PLAN.md 20): the synthetic acquisition pipeline runs
end to end and recovers useful acquisition directions.

These are the structural guarantees. The quantitative Stage 0 gate lives in
`experiments/synthetic/run_stage0.py`, which is too slow for a test suite.
"""

from __future__ import annotations

import numpy as np

from ldva.acquisition.clustering import ClusteringConfig, LatentClustering
from ldva.acquisition.directions import DirectionConfig, DirectionGenerator
from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig
from ldva.acquisition.metadata_mapper import (
    ActionabilityConfig,
    MetadataMapper,
    MetadataMapperConfig,
    filter_actionable_directions,
)
from ldva.acquisition.objective import AllocationObjective, BudgetSpec, ObjectiveConfig
from ldva.acquisition.greedy import greedy_search


def test_metadata_to_latent_map_is_smooth_and_differentiable(world):
    """The analytic Jacobian of f(m) matches finite differences (A2)."""
    m0 = world.metadata_spec.sample(1, np.random.default_rng(0))[0]
    J = world.latent_jacobian(m0)
    fd = np.zeros_like(J)
    for k in range(len(m0)):
        e = np.zeros_like(m0)
        e[k] = 1e-5
        fd[:, k] = (world.latent_from_metadata(m0 + e)[0]
                    - world.latent_from_metadata(m0 - e)[0]) / 2e-5
    assert np.abs(J - fd).max() < 1e-6


def test_initial_support_is_incomplete(world, store):
    """Acquisition must have somewhere to go: the initial modes cover only part
    of the metadata box that the evaluation distribution spans."""
    mn = world.metadata_spec.normalize(store.metadata)
    covered = mn.max(0) - mn.min(0)
    assert np.any(covered < 0.7), "initial dataset already spans the metadata box"


def test_pipeline_runs_end_to_end(world, store, datamodel, trained_policy):
    """Latents -> clusters -> directions -> allocation, with no empty stage."""
    _, ckpts = trained_policy
    ref = ckpts[len(ckpts) // 2]
    z = datamodel.encode_store(store, ref.features)
    assert z.shape == (len(store), datamodel.latent_dim)
    assert np.isfinite(z).all()

    clusters = LatentClustering(ClusteringConfig(n_clusters=3, seed=0)).fit(z, store.metadata)
    assert len(clusters) == 3
    assert all(c.size >= 2 for c in clusters)

    directions = DirectionGenerator(DirectionConfig(r_max=2, seed=0)).generate(clusters, z)
    assert directions, "no candidate directions survived the filters"

    mapper = MetadataMapper(world.metadata_spec, MetadataMapperConfig(seed=0))
    mapper.fit(z, store.metadata, clusters)
    assert mapper.jacobians, "no local Jacobian was fitted"

    kept, report = filter_actionable_directions(
        directions, mapper, store.metadata, ActionabilityConfig(min_achievable_cosine=0.5))
    assert kept
    assert 0.0 <= report["survival_rate"] <= 1.0

    sampler = LatentSampler(clusters, LatentSamplerConfig(seed=0))
    budget = BudgetSpec.from_directions(kept, budget=4)
    obj = AllocationObjective(datamodel, sampler, kept, budget,
                              policy_features=ref.features,
                              cfg=ObjectiveConfig(n_mc=4, seed=0))
    res = greedy_search(obj)
    assert res.best_allocation.sum() == 4
    assert np.isfinite(res.best_value)


def test_acquisition_plan_is_feasible_and_collectable(world, store, datamodel, trained_policy):
    """A plan must produce metadata inside the declared feasible box, and the
    environment must actually be able to collect at it."""
    _, ckpts = trained_policy
    ref = ckpts[len(ckpts) // 2]
    z = datamodel.encode_store(store, ref.features)
    clusters = LatentClustering(ClusteringConfig(n_clusters=3, seed=0)).fit(z, store.metadata)
    directions = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    mapper = MetadataMapper(world.metadata_spec, MetadataMapperConfig(seed=0))
    mapper.fit(z, store.metadata, clusters)

    plans = mapper.plan_allocation(
        directions, np.array([2] + [0] * (len(directions) - 1)), store.metadata,
        rng=np.random.default_rng(0))
    assert plans
    for p in plans:
        assert world.metadata_spec.is_feasible(p.metadata).all()
        assert len(p.metadata) == len(p.anchor_sample_ids)
        new = world.build_store(0, np.random.default_rng(0), metadata=p.metadata)
        assert len(new) == len(p.metadata)
        assert np.isfinite(new.obs).all() and np.isfinite(new.act).all()


def test_dataset_grows_by_exactly_the_budget(world, store):
    """Round bookkeeping: |D_{t+1}| = |D_t| + B."""
    rng = np.random.default_rng(0)
    new_meta = world.metadata_spec.sample(7, rng)
    grown = store.concat(world.build_store(0, rng, metadata=new_meta, round_id=1))
    assert len(grown) == len(store) + 7
    assert (grown.round_id[len(store):] == 1).all()
    assert (grown.round_id[: len(store)] == 0).all()
