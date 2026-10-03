"""Latent direction -> metadata inversion (PLAN.md 12, 24; SETUP.md 17).

The mapper is tested against a world whose metadata-to-latent map is known
analytically, so "did it recover the right perturbation" is checkable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ldva.acquisition.clustering import ClusteringConfig, LatentClustering
from ldva.acquisition.directions import DirectionConfig, DirectionGenerator
from ldva.acquisition.metadata_mapper import (
    ActionabilityConfig,
    LocalJacobian,
    MetadataMapper,
    MetadataMapperConfig,
    evaluate_realized_direction,
    filter_actionable_directions,
)
from ldva.data.metadata import MetadataField, MetadataSpec


def test_reachability_cosine_measures_subspace_overlap():
    """A latent direction outside range(J) cannot be produced by any delta_m."""
    J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    jac = LocalJacobian(0, J, np.zeros(2), np.zeros(3), 100)
    assert jac.reachability_cosine(np.array([1.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert jac.reachability_cosine(np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0)
    assert jac.reachability_cosine(np.array([1.0, 0.0, 1.0])) == pytest.approx(
        np.sqrt(0.5), abs=1e-6)
    assert jac.reachability_cosine(np.zeros(3)) == 0.0


def test_mapper_recovers_a_known_linear_jacobian():
    """With z exactly linear in m, the fitted J must match the truth."""
    rng = np.random.default_rng(0)
    spec = MetadataSpec([MetadataField("a", 0.0, 1.0), MetadataField("b", 0.0, 1.0)])
    J_true = np.array([[2.0, -1.0], [0.5, 1.5], [0.0, 3.0]])
    m = spec.sample(300, rng)
    z = spec.normalize(m) @ J_true.T

    class _C:
        cluster_id = 0
        member_ids = np.arange(300)
        boundary_ids = np.arange(10)

    mp = MetadataMapper(spec, MetadataMapperConfig(ridge=1e-8, seed=0))
    mp.fit(z, m, [_C()])
    assert np.abs(mp.jacobians[0].J - J_true).max() < 1e-4
    assert mp.jacobians[0].r2_heldout > 0.99


def test_solve_delta_m_hits_a_reachable_target():
    rng = np.random.default_rng(0)
    spec = MetadataSpec([MetadataField("a", 0.0, 1.0), MetadataField("b", 0.0, 1.0)])
    J_true = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    m = spec.sample(300, rng)
    z = spec.normalize(m) @ J_true.T

    class _C:
        cluster_id = 0
        member_ids = np.arange(300)
        boundary_ids = np.arange(10)

    mp = MetadataMapper(spec, MetadataMapperConfig(ridge=1e-8, max_step_norm=1.0, seed=0))
    mp.fit(z, m, [_C()])
    v = np.array([1.0, 0.0, 0.0])
    dm, resid, ok = mp.solve_delta_m(0, v, 0.2, np.array([0.5, 0.5]))
    assert ok and resid < 1e-6
    assert dm[0] == pytest.approx(0.2, abs=1e-4)
    assert abs(dm[1]) < 1e-4


def test_solve_respects_box_and_step_constraints():
    """A perturbation must stay inside the feasible metadata box and inside the
    metadata-space trust region."""
    rng = np.random.default_rng(0)
    spec = MetadataSpec([MetadataField("a", 0.0, 1.0), MetadataField("b", 0.0, 1.0)])
    J_true = np.eye(2)
    m = spec.sample(200, rng)
    z = spec.normalize(m) @ J_true.T

    class _C:
        cluster_id = 0
        member_ids = np.arange(200)
        boundary_ids = np.arange(10)

    mp = MetadataMapper(spec, MetadataMapperConfig(ridge=1e-8, max_step_norm=0.1, seed=0))
    mp.fit(z, m, [_C()])
    # ask for a huge move from a corner of the box
    dm, _, _ = mp.solve_delta_m(0, np.array([1.0, 0.0]), 10.0, np.array([0.98, 0.5]))
    assert dm[0] <= 0.02 + 1e-9, "left the feasible box"
    assert np.all(np.abs(dm) <= 0.1 + 1e-9), "exceeded the step cap"


def test_uncontrollable_fields_are_never_moved():
    rng = np.random.default_rng(0)
    spec = MetadataSpec([
        MetadataField("free", 0.0, 1.0),
        MetadataField("fixed", 0.0, 1.0, controllable=False),
    ])
    m = spec.sample(200, rng)
    z = spec.normalize(m) @ np.eye(2).T

    class _C:
        cluster_id = 0
        member_ids = np.arange(200)
        boundary_ids = np.arange(10)

    mp = MetadataMapper(spec, MetadataMapperConfig(ridge=1e-8, seed=0))
    mp.fit(z, m, [_C()])
    dm, _, _ = mp.solve_delta_m(0, np.array([0.0, 1.0]), 0.5, np.array([0.5, 0.5]))
    assert abs(dm[1]) < 1e-9, "moved a field the acquisition interface cannot control"


def test_random_pairs_fit_better_than_nearest_neighbour_pairs(world, store, datamodel,
                                                              trained_policy):
    """Nearest-neighbour pair differences minimize signal-to-noise: the signal
    is J*delta_m, which vanishes as delta_m -> 0, while encoder noise does not."""
    _, ckpts = trained_policy
    z = datamodel.encode_store(store, ckpts[1].features)
    clusters = LatentClustering(ClusteringConfig(n_clusters=2, seed=0)).fit(z, store.metadata)
    r2 = {}
    for mode in ("knn", "random"):
        mp = MetadataMapper(
            world.metadata_spec, MetadataMapperConfig(pair_mode=mode, seed=0))
        mp.fit(z, store.metadata, clusters)
        r2[mode] = np.mean([j.r2_heldout for j in mp.jacobians.values()])
    assert r2["random"] >= r2["knn"]


def test_plans_are_feasible_and_anchor_matched(world, store, datamodel, trained_policy):
    _, ckpts = trained_policy
    z = datamodel.encode_store(store, ckpts[1].features)
    clusters = LatentClustering(ClusteringConfig(n_clusters=3, seed=0)).fit(z, store.metadata)
    dirs = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    mp = MetadataMapper(world.metadata_spec, MetadataMapperConfig(seed=0))
    mp.fit(z, store.metadata, clusters)
    plan = mp.plan_direction(dirs[0], 5, store.metadata, rng=np.random.default_rng(0))
    assert plan.metadata.shape == (5, store.meta_dim)
    assert world.metadata_spec.is_feasible(plan.metadata).all()
    assert plan.anchor_sample_ids.shape == (5,)
    # every row's anchor must be one of the direction's own anchors
    assert set(plan.anchor_sample_ids.tolist()) <= set(np.asarray(dirs[0].anchor_ids).tolist())
    assert np.allclose(plan.anchor_metadata, store.metadata[plan.anchor_sample_ids])


def test_actionability_filter_drops_unachievable_directions(world, store, datamodel,
                                                            trained_policy):
    """SETUP.md 16's fourth filter. A strict threshold must keep a subset of a
    loose one, and the kept directions must beat the threshold."""
    _, ckpts = trained_policy
    z = datamodel.encode_store(store, ckpts[1].features)
    clusters = LatentClustering(ClusteringConfig(n_clusters=3, seed=0)).fit(z, store.metadata)
    dirs = DirectionGenerator(DirectionConfig(r_max=2, seed=0)).generate(clusters, z)
    mp = MetadataMapper(world.metadata_spec, MetadataMapperConfig(seed=0))
    mp.fit(z, store.metadata, clusters)

    loose, r_loose = filter_actionable_directions(
        list(dirs), mp, store.metadata, ActionabilityConfig(min_achievable_cosine=0.0))
    strict, r_strict = filter_actionable_directions(
        list(dirs), mp, store.metadata, ActionabilityConfig(min_achievable_cosine=0.999,
                                                           keep_at_least=0))
    assert len(loose) >= len(strict)
    assert r_loose["survival_rate"] >= r_strict["survival_rate"]
    for d in strict:
        assert d.extra["achievable_cosine"] >= 0.999 - 1e-9
    assert "reachability_cosine" in loose[0].extra


def test_actionability_keeps_a_fallback_when_nothing_qualifies(world, store, datamodel,
                                                              trained_policy):
    """The planner should still get candidates (flagged) rather than nothing."""
    _, ckpts = trained_policy
    z = datamodel.encode_store(store, ckpts[1].features)
    clusters = LatentClustering(ClusteringConfig(n_clusters=2, seed=0)).fit(z, store.metadata)
    dirs = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    mp = MetadataMapper(world.metadata_spec, MetadataMapperConfig(seed=0))
    mp.fit(z, store.metadata, clusters)
    kept, rep = filter_actionable_directions(
        dirs, mp, store.metadata,
        ActionabilityConfig(min_achievable_cosine=1.1, keep_at_least=2))
    assert len(kept) == 2
    assert rep["fallback_used"]
    assert all(d.extra.get("actionability_fallback") for d in kept)


def test_evaluate_realized_direction_signs():
    """Perfect alignment -> cosine 1; opposite -> -1; orthogonal -> 0."""
    v = np.array([1.0, 0.0, 0.0])
    base = np.zeros((1, 3))
    assert evaluate_realized_direction(base, np.array([[2.0, 0, 0]]), v)[
        "direction_cosine"] == pytest.approx(1.0)
    assert evaluate_realized_direction(base, np.array([[-2.0, 0, 0]]), v)[
        "direction_cosine"] == pytest.approx(-1.0)
    orth = evaluate_realized_direction(base, np.array([[0.0, 2.0, 0]]), v)
    assert orth["direction_cosine"] == pytest.approx(0.0)
    assert orth["orthogonal_error"] == pytest.approx(2.0)
    good = evaluate_realized_direction(base, np.array([[3.0, 0, 0]]), v)
    assert good["displacement_along_direction"] == pytest.approx(3.0)
    assert good["frac_samples_positive_cosine"] == pytest.approx(1.0)


def test_world_jacobian_is_locally_predictive(world):
    """Assumption A2: a small metadata change produces a predictable latent
    change in the *true* generative map."""
    rng = np.random.default_rng(0)
    m0 = world.metadata_spec.sample(1, rng)[0]
    J = world.latent_jacobian(m0)
    cos = []
    for _ in range(50):
        dm = rng.normal(size=len(m0)) * 0.01
        actual = (world.latent_from_metadata(m0 + dm)
                  - world.latent_from_metadata(m0))[0]
        pred = J @ dm
        cos.append(float(pred @ actual / (np.linalg.norm(pred) * np.linalg.norm(actual))))
    assert np.mean(cos) > 0.99
