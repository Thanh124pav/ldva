"""Latent-geometry diagnostics and the Phase 3 gate (PLAN.md 19; SETUP.md 14, 22).

Each diagnostic is tested on data where the right answer is known by
construction, because a gate that cannot distinguish good geometry from noise
is worse than no gate.
"""

from __future__ import annotations

import numpy as np

from ldva.analysis.latent_geometry import (
    GeometryGate,
    cluster_stability,
    label_noise_report,
    latent_geometry_report,
    latent_vs_effect_distance_correlation,
    neighbor_effect_consistency,
    pca_spectrum,
)
from ldva.data.context_dataset import ContextRecord
from ldva.data.effect_profiles import EffectProfileTable


def test_pca_spectrum_detects_intrinsic_dimension():
    rng = np.random.default_rng(0)
    #: 2 informative directions embedded in 8 dimensions
    latent = rng.normal(size=(500, 2))
    basis = np.linalg.qr(rng.normal(size=(8, 8)))[0][:, :2]
    z = latent @ basis.T + 1e-3 * rng.normal(size=(500, 8))
    rep = pca_spectrum(z)
    assert rep["n_dims_for_90pct"] == 2
    assert 1.5 < rep["participation_ratio"] < 2.5
    assert len(rep["eigenvalues"]) == 8
    assert np.all(np.diff(rep["explained_variance_ratio"]) <= 1e-9)


def _records_with_effect_structure(n_samples=40, n_ctx=300, latent=None, seed=0):
    """Build records whose effects are a known function of a hidden factor."""
    rng = np.random.default_rng(seed)
    factor = latent if latent is not None else rng.normal(size=n_samples)
    recs = []
    for c in range(n_ctx):
        ids = rng.choice(n_samples, size=6, replace=False)
        eff = factor[ids] + 0.05 * rng.normal(size=len(ids))
        recs.append(ContextRecord(c, f"ck{c % 4}", ids, eff, float(eff.sum())))
    return recs, factor


def test_neighbor_consistency_detects_a_good_and_a_bad_latent():
    """A latent aligned with the effect factor must score well below 1; a random
    latent must score about 1."""
    recs, factor = _records_with_effect_structure()
    table = EffectProfileTable(recs, 40, min_shared=1)
    rng = np.random.default_rng(1)

    good = np.stack([factor, 0.01 * rng.normal(size=40)], axis=1)
    bad = rng.normal(size=(40, 2))
    r_good = neighbor_effect_consistency(good, table, k=5, seed=0)
    r_bad = neighbor_effect_consistency(bad, table, k=5, seed=0)
    assert r_good["neighbor_consistency_ratio"] < 0.6
    assert r_good["frac_probes_consistent"] > 0.8
    assert r_bad["neighbor_consistency_ratio"] > r_good["neighbor_consistency_ratio"]


def test_latent_vs_effect_distance_correlation():
    recs, factor = _records_with_effect_structure()
    table = EffectProfileTable(recs, 40, min_shared=1)
    good = factor.reshape(-1, 1)
    bad = np.random.default_rng(2).normal(size=(40, 1))
    assert latent_vs_effect_distance_correlation(good, table, seed=0)[
        "latent_effect_spearman"] > 0.8
    r_bad = latent_vs_effect_distance_correlation(bad, table, seed=0)
    assert r_bad["latent_effect_spearman"] < 0.5


def test_neighbor_consistency_reports_when_no_pairs_exist():
    """Pairs only exist where samples co-occur. With disjoint batches the
    diagnostic must say so instead of returning a misleading number."""
    recs = [ContextRecord(i, "ck0", np.array([2 * i, 2 * i + 1]),
                          np.array([0.1, 0.2]), 0.0) for i in range(10)]
    table = EffectProfileTable(recs, 20, min_shared=5)
    rep = neighbor_effect_consistency(np.random.default_rng(0).normal(size=(20, 2)),
                                      table, k=3, seed=0)
    assert np.isnan(rep["neighbor_consistency_ratio"])
    assert rep["n_probes"] == 0


def test_cluster_stability_separates_real_clusters_from_noise():
    rng = np.random.default_rng(0)
    clustered = np.concatenate([c + 0.15 * rng.normal(size=(80, 3))
                                for c in rng.normal(scale=6.0, size=(4, 3))])
    unstructured = rng.normal(size=(320, 3))
    s_good = cluster_stability(clustered, n_clusters=4, seed=0)
    s_bad = cluster_stability(unstructured, n_clusters=4, seed=0)
    assert s_good["cluster_ari_mean"] > 0.9
    assert s_bad["cluster_ari_mean"] < s_good["cluster_ari_mean"]


def test_label_noise_report_separates_signal_from_noise():
    """F3: with a strong per-sample factor the between-sample share is high;
    with pure noise labels it collapses."""
    recs_signal, _ = _records_with_effect_structure(seed=0)
    table_s = EffectProfileTable(recs_signal, 40, min_shared=1)
    rep_s = label_noise_report(recs_signal, table_s)
    assert rep_s["between_sample_var_share"] > 0.9

    rng = np.random.default_rng(3)
    recs_noise = [
        ContextRecord(c, "ck0", rng.choice(40, size=6, replace=False),
                      rng.normal(size=6), 0.0)
        for c in range(300)
    ]
    table_n = EffectProfileTable(recs_noise, 40, min_shared=1)
    rep_n = label_noise_report(recs_noise, table_n)
    assert rep_n["between_sample_var_share"] < 0.3


def test_gain_signal_report_flags_weak_composition_signal():
    """If batch gain is driven by the checkpoint, set-level metrics computed
    over all contexts are confounded, and the report must say so."""
    from ldva.analysis.latent_geometry import gain_signal_report

    rng = np.random.default_rng(0)
    weak = [ContextRecord(c, f"ck{c % 5}", rng.choice(20, 4, replace=False),
                          rng.normal(size=4), float((c % 5) * 10 + rng.normal() * 0.1))
            for c in range(200)]
    strong = [ContextRecord(c, f"ck{c % 5}", rng.choice(20, 4, replace=False),
                            rng.normal(size=4), float(rng.normal() * 10))
              for c in range(200)]
    r_weak = gain_signal_report(weak)
    r_strong = gain_signal_report(strong)
    assert r_weak["gain_within_group_share"] < 0.1
    assert r_weak["composition_signal_is_weak"] is True
    assert r_strong["gain_within_group_share"] > 0.5
    assert r_strong["composition_signal_is_weak"] is False


def test_geometry_gate_passes_on_good_geometry_and_flags_bad():
    recs, factor = _records_with_effect_structure(n_samples=40, n_ctx=400)
    table = EffectProfileTable(recs, 40, min_shared=1)
    rng = np.random.default_rng(4)
    good = np.stack([factor, 0.02 * rng.normal(size=40)], axis=1)
    rep = latent_geometry_report(good, recs, table, n_clusters=4, k_neighbors=5,
                                 gate=GeometryGate(), seed=0)
    assert set(rep) >= {"pca", "neighbor", "distance_corr", "clusters", "labels",
                        "additivity", "gain_signal", "checks", "gate_passed"}
    assert rep["checks"]["F1_neighbor_effect_consistency"] is True
    assert rep["checks"]["F3_label_signal"] is True

    bad = rng.normal(size=(40, 2))
    rep_bad = latent_geometry_report(bad, recs, table, n_clusters=4, k_neighbors=5, seed=0)
    assert rep_bad["checks"]["F1_latent_effect_distance_corr"] is not True


def test_gate_treats_undetermined_as_not_passed():
    """A NaN diagnostic must not count as a pass."""
    from ldva.analysis.latent_geometry import _ge, _le

    assert _ge(float("nan"), 0.5) is None
    assert _le(float("nan"), 0.5) is None
    assert _ge(0.6, 0.5) is True
    assert _le(0.4, 0.5) is True
    assert _ge(0.4, 0.5) is False
