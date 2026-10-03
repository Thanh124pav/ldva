"""Clustering and candidate-direction generation (PLAN.md 7-8; PLAN.md 6-16)."""

from __future__ import annotations

import numpy as np

from ldva.acquisition.clustering import (
    ClusteringConfig,
    LatentClustering,
    distance_to_support,
    support_density,
)
from ldva.acquisition.directions import DirectionConfig, DirectionGenerator


def _clusters(z, n=3, metadata=None):
    return LatentClustering(ClusteringConfig(n_clusters=n, seed=0)).fit(z, metadata)


def test_cluster_state_exposes_required_fields(latent_blobs):
    """PLAN.md 7 lists exactly what a ClusterState must carry."""
    z, _ = latent_blobs
    for c in _clusters(z):
        assert c.member_ids.size >= 2
        assert c.centroid.shape == (z.shape[1],)
        assert c.covariance.shape == (z.shape[1], z.shape[1])
        assert c.pca_basis.shape[1] == z.shape[1]
        assert c.pca_eigenvalues.shape[0] == z.shape[1]
        assert c.boundary_ids.size >= 1
        assert c.radius > 0
        # eigenvalues sorted descending and non-negative
        assert np.all(np.diff(c.pca_eigenvalues) <= 1e-9)
        assert np.all(c.pca_eigenvalues >= -1e-12)


def test_boundary_points_are_further_out_than_the_average_member(latent_blobs):
    z, _ = latent_blobs
    for c in _clusters(z):
        d_all = np.linalg.norm(z[c.member_ids] - c.centroid, axis=1)
        d_bnd = np.linalg.norm(z[c.boundary_ids] - c.centroid, axis=1)
        assert d_bnd.mean() > d_all.mean()


def test_intrinsic_rank_respects_rho_and_r_max(latent_blobs):
    z, _ = latent_blobs
    for c in _clusters(z):
        for rho in (0.5, 0.9, 0.99):
            for r_max in (1, 2, 5):
                r = c.intrinsic_rank(rho, r_max)
                assert 1 <= r <= r_max
                cum = np.cumsum(c.explained_variance_ratio())
                # either rho is met, or the cap stopped us short
                assert cum[r - 1] >= rho - 1e-9 or r == r_max


def test_elongated_cluster_has_rank_one(latent_blobs):
    """A nearly 1-D domain must be detected as such, otherwise the direction
    set is padded with noise components."""
    z, _ = latent_blobs
    for c in _clusters(z):
        assert c.intrinsic_rank(0.90, 5) == 1
        assert c.explained_variance_ratio()[0] > 0.85


def test_directions_are_unit_vectors_with_both_signs(latent_blobs):
    z, _ = latent_blobs
    clusters = _clusters(z)
    dirs = DirectionGenerator(DirectionConfig(r_max=2, seed=0)).generate(clusters, z)
    assert dirs
    for d in dirs:
        assert abs(np.linalg.norm(d.vector) - 1.0) < 1e-8
    # both signs of a retained component should appear for some cluster
    pairs = {(d.cluster_id, d.component, d.sign) for d in dirs}
    assert any((c, k, 1) in pairs and (c, k, -1) in pairs for c, k, _ in pairs)


def test_anchors_are_extreme_along_their_own_direction(latent_blobs):
    """Per-direction anchors (PLAN.md 8.2). Choosing anchors once per cluster
    straddles both ends of an elongated domain, which makes +v and -v each
    average to zero outward movement."""
    z, _ = latent_blobs
    clusters = _clusters(z)
    dirs = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    by_cluster = {}
    for d in dirs:
        by_cluster.setdefault(d.cluster_id, []).append(d)
    for cid, ds in by_cluster.items():
        c = [x for x in clusters if x.cluster_id == cid][0]
        for d in ds:
            proj_anchor = ((z[d.anchor_ids] - c.centroid) @ d.vector).mean()
            proj_all = ((z[c.member_ids] - c.centroid) @ d.vector).mean()
            assert proj_anchor > proj_all
        # opposite signs must use different anchors
        if len(ds) == 2:
            assert set(ds[0].anchor_ids) != set(ds[1].anchor_ids)


def test_all_kept_directions_are_outward(latent_blobs):
    z, _ = latent_blobs
    dirs = DirectionGenerator(DirectionConfig(r_max=2, seed=0)).generate(_clusters(z), z)
    assert dirs
    assert all(d.outward_score > 0 for d in dirs)


def test_all_kept_directions_decrease_support_density(latent_blobs):
    """Expansion, not thickening of what we already own."""
    z, _ = latent_blobs
    dirs = DirectionGenerator(DirectionConfig(r_max=2, seed=0)).generate(_clusters(z), z)
    assert all(d.density_ratio < 1.0 for d in dirs)


def test_trust_region_blocks_long_range_extrapolation(latent_blobs):
    """PLAN.md 7: uncontrolled long-range extrapolation must be refused."""
    z, _ = latent_blobs
    clusters = _clusters(z)
    kept_small = DirectionGenerator(
        DirectionConfig(r_max=1, delta_scale=0.5, seed=0)).generate(clusters, z)
    gen_big = DirectionGenerator(DirectionConfig(r_max=1, delta_scale=12.0, seed=0))
    kept_big = gen_big.generate(clusters, z)
    assert len(kept_small) > 0
    assert len(kept_big) == 0
    assert "outside_trust_region_absolute" in gen_big.report()["rejection_reasons"]


def test_filters_can_be_disabled_for_ablation(latent_blobs):
    """Ablation 6-7 of PLAN.md 18 need the unfiltered candidate set."""
    z, _ = latent_blobs
    clusters = _clusters(z)
    cfg = DirectionConfig(r_max=1, require_outward=False, require_density_decrease=False,
                          require_trust_region=False, seed=0)
    dirs = DirectionGenerator(cfg).generate(clusters, z)
    assert len(dirs) == 2 * sum(c.intrinsic_rank(0.90, 1) for c in clusters)


def test_random_directions_ablation(latent_blobs):
    z, _ = latent_blobs
    cfg = DirectionConfig(use_random_directions=True, n_random_per_cluster=3,
                          require_outward=False, require_density_decrease=False,
                          require_trust_region=False, seed=0)
    dirs = DirectionGenerator(cfg).generate(_clusters(z), z)
    assert all(d.source == "random" and d.component == -1 for d in dirs)
    assert len(dirs) == 3 * 3


def test_support_density_and_distance_behave_monotonically(latent_blobs):
    z, _ = latent_blobs
    inside = z[:1]
    far = z[:1] + 100.0
    assert support_density(inside, z)[0] > support_density(far, z)[0]
    assert distance_to_support(far, z)[0] > distance_to_support(inside, z)[0]


def test_hdbscan_noise_points_are_not_domains():
    """Label -1 is HDBSCAN's noise class and must never become a cluster."""
    rng = np.random.default_rng(0)
    dense = np.concatenate([rng.normal(scale=0.1, size=(60, 3)) + c
                            for c in ([0, 0, 0], [5, 5, 5])])
    outliers = rng.uniform(-20, 20, size=(10, 3))
    z = np.concatenate([dense, outliers])
    clu = LatentClustering(ClusteringConfig(method="hdbscan", min_cluster_size=10, seed=0))
    clusters = clu.fit(z)
    assert all(c.cluster_id >= 0 for c in clusters)
    assert clu.report()["n_noise"] >= 0


def test_latent_sampler_is_stochastic_and_centred_on_the_step(latent_blobs):
    """PLAN.md 9: a direction is a distribution, not one latent point."""
    from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig

    z, _ = latent_blobs
    clusters = _clusters(z)
    dirs = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    s = LatentSampler(clusters, LatentSamplerConfig(sigma=0.3, seed=0))
    d = dirs[0]
    draws = s.sample(d, 400)
    assert draws.shape == (400, z.shape[1])
    assert draws.std(0).sum() > 0, "sampler is deterministic"
    # the mean should sit near anchor_mean + delta * v
    expected = d.anchors.mean(0) + d.delta * d.vector
    assert np.linalg.norm(draws.mean(0) - expected) < 0.5 * d.delta + 0.5


def test_latent_sampler_allocation_sizes_match(latent_blobs):
    from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig

    z, _ = latent_blobs
    clusters = _clusters(z)
    dirs = DirectionGenerator(DirectionConfig(r_max=1, seed=0)).generate(clusters, z)
    s = LatentSampler(clusters, LatentSamplerConfig(seed=0))
    alloc = np.zeros(len(dirs), dtype=np.int64)
    alloc[0], alloc[-1] = 3, 2
    batch = s.sample_allocation(dirs, alloc)
    assert batch.shape == (5, z.shape[1])
    assert s.sample_allocation(dirs, np.zeros(len(dirs), dtype=np.int64)).shape[0] == 0
