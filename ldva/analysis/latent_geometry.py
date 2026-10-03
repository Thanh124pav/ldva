"""Latent-geometry diagnostics (PLAN.md Phase 3, 17.1, 19; PLAN.md 6, 18).

This module exists to answer one question before any acquisition code runs:
**is the latent geometry meaningful enough to justify clustering and directional
expansion?** The checks map one-to-one onto the failure modes of PLAN.md 19:

- `neighbor_effect_consistency` -> F1 (no stable effect geometry)
- `additivity_check`            -> F2 (batch utility is nearly additive)
- `label_noise_report`          -> F3 (multi-context labels are mostly noise)
- `cluster_stability`           -> whether clusters are an artifact of the seed

`latent_geometry_report` bundles them and returns an explicit `gate_passed`
verdict rather than leaving the reader to interpret a wall of numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from ldva.data.context_dataset import ContextRecord
from ldva.data.effect_profiles import EffectProfileTable
from ldva.training.metrics import (
    additivity_report,
    group_variance_decomposition,
    pearson,
    spearman,
)


def pca_spectrum(z: np.ndarray) -> dict:
    """Eigen-spectrum of the latent covariance, plus participation ratio.

    The participation ratio is a smooth stand-in for "how many directions are
    actually used"; a value far below the latent dimension means the sweep in
    PLAN.md 6 has already saturated.
    """
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(0, keepdims=True)
    cov = np.cov(zc, rowvar=False)
    cov = np.atleast_2d(cov)
    evals = np.linalg.eigvalsh(cov)[::-1]
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    ratio = evals / total if total > 0 else np.zeros_like(evals)
    cum = np.cumsum(ratio)
    return {
        "eigenvalues": evals.tolist(),
        "explained_variance_ratio": ratio.tolist(),
        "n_dims_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "participation_ratio": float(total**2 / np.sum(evals**2)) if total > 0 else 0.0,
        "latent_dim": int(z.shape[1]),
        "latent_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
        "latent_norm_std": float(np.linalg.norm(z, axis=1).std()),
    }


def neighbor_effect_consistency(
    z: np.ndarray,
    table: EffectProfileTable,
    k: int = 10,
    n_probe: int = 400,
    seed: int = 0,
) -> dict:
    """F1: do latent neighbours have similar contextual effect profiles?

    For each probe sample we compare the mean effect distance to its k latent
    nearest neighbours against the mean distance to k random samples. The ratio
    is the headline: < 1 means latent proximity predicts effect similarity, and
    1.0 means the geometry carries no effect information at all.
    """
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    rng = np.random.default_rng(seed)
    k = min(k, n - 1)
    if k < 1:
        return {"neighbor_consistency_ratio": float("nan"), "n_probes": 0}

    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    probe = rng.choice(n, size=min(n_probe, n), replace=False)
    _, idx = nn.kneighbors(z[probe])

    near_d, rand_d = [], []
    for row, i in enumerate(probe):
        neigh = [j for j in idx[row] if j != i][:k]
        dn = [table.distance(int(i), int(j)) for j in neigh]
        dn = [d for d in dn if d is not None]
        rnd = rng.choice(n, size=k, replace=False)
        dr = [table.distance(int(i), int(j)) for j in rnd if j != i]
        dr = [d for d in dr if d is not None]
        if dn and dr:
            near_d.append(np.mean(dn))
            rand_d.append(np.mean(dr))

    if not near_d:
        # pairs only exist where samples co-occur, so a sparse context set can
        # leave no comparable pairs; report that rather than a misleading number
        return {
            "neighbor_consistency_ratio": float("nan"),
            "n_probes": 0,
            "note": "no co-occurring pairs among latent neighbours",
        }
    near_d, rand_d = np.array(near_d), np.array(rand_d)
    return {
        "neighbor_effect_distance": float(near_d.mean()),
        "random_effect_distance": float(rand_d.mean()),
        "neighbor_consistency_ratio": float(near_d.mean() / max(rand_d.mean(), 1e-12)),
        "frac_probes_consistent": float((near_d < rand_d).mean()),
        "n_probes": int(len(near_d)),
        "k": int(k),
    }


def latent_vs_effect_distance_correlation(
    z: np.ndarray, table: EffectProfileTable, n_pairs: int = 5000, seed: int = 0
) -> dict:
    """Correlation between ||z_i - z_j|| and d_effect(i, j) (PLAN.md 17.1)."""
    rng = np.random.default_rng(seed)
    if len(table) == 0:
        return {"latent_effect_spearman": float("nan"), "n_pairs": 0}
    pairs, target = table.sample_pairs(min(n_pairs, len(table) * 4), rng)
    d = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1)
    return {
        "latent_effect_spearman": spearman(d, target),
        "latent_effect_pearson": pearson(d, target),
        "n_pairs": int(len(pairs)),
    }


def cluster_stability(
    z: np.ndarray, n_clusters: int = 8, n_trials: int = 5, subsample: float = 0.8, seed: int = 0
) -> dict:
    """Adjusted Rand index of KMeans labels across seeds and subsamples.

    Unstable clusters mean "local effect domain" is not a well-defined object on
    this latent space, and the direction generation built on top of it would be
    reporting noise.
    """
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n < n_clusters * 2:
        return {"cluster_ari_mean": float("nan"), "n_clusters": n_clusters}
    rng = np.random.default_rng(seed)
    base = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(z)
    aris = []
    for t in range(n_trials):
        sub = rng.choice(n, size=int(subsample * n), replace=False)
        lab = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed + 1 + t).fit_predict(z[sub])
        aris.append(adjusted_rand_score(base[sub], lab))
    return {
        "cluster_ari_mean": float(np.mean(aris)),
        "cluster_ari_min": float(np.min(aris)),
        "cluster_ari_std": float(np.std(aris)),
        "n_clusters": int(n_clusters),
        "n_trials": int(n_trials),
    }


def label_noise_report(records: list[ContextRecord], table: EffectProfileTable) -> dict:
    """F3: is the per-sample effect signal above the estimator noise floor?

    Decomposes the variance of effect labels into a between-sample component
    (real differences between samples) and a within-sample component (the same
    sample varying across contexts). The within-sample part mixes genuine
    contextuality with estimator noise, so this is an upper bound on noise, not
    a clean separation - but a between-sample share near zero means the labels
    carry almost no sample-level signal at all.
    """
    all_eff, all_ids = [], []
    for r in records:
        all_eff.append(r.per_sample_effects)
        all_ids.append(r.batch_sample_ids)
    eff = np.concatenate(all_eff)
    ids = np.concatenate(all_ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    means = np.bincount(inv, weights=eff, minlength=len(uniq)) / np.maximum(
        np.bincount(inv, minlength=len(uniq)), 1
    )
    within = eff - means[inv]
    total_var = float(np.var(eff))
    within_var = float(np.var(within))
    return {
        "effect_total_var": total_var,
        "effect_within_sample_var": within_var,
        "effect_between_sample_var": max(total_var - within_var, 0.0),
        "between_sample_var_share": float(
            max(total_var - within_var, 0.0) / total_var if total_var > 1e-20 else 0.0
        ),
        "n_labels": int(len(eff)),
        "effect_distance_median": table.scale,
    }


def additivity_check(records: list[ContextRecord], n_samples: int, seed: int = 0) -> dict:
    """F2 wrapper that uses *all* records, so the fit is not underdetermined."""
    gains = np.array([r.batch_gain for r in records], dtype=np.float64)
    comps = [r.batch_sample_ids for r in records]
    return additivity_report(gains, comps, n_samples, seed=seed)


def gain_signal_report(records: list[ContextRecord]) -> dict:
    """How much of the batch-gain signal is about *composition* at all?

    `U(Update(theta, B)) - U(theta)` moves mostly with the checkpoint. If the
    within-checkpoint share is tiny, batch-gain metrics computed over all
    contexts mostly measure checkpoint identification, and any claim about
    set-level utility has to be made on the within-checkpoint residual instead.
    """
    gains = np.array([r.batch_gain for r in records], dtype=np.float64)
    ckpts = np.array([r.policy_ckpt_id for r in records])
    dec = group_variance_decomposition(gains, ckpts)
    return {
        **{f"gain_{k}": v for k, v in dec.items()},
        "composition_signal_is_weak": bool(dec["within_group_share"] < 0.1),
    }


@dataclass
class GeometryGate:
    """Thresholds for the Phase 3 go / no-go decision."""

    max_neighbor_consistency_ratio: float = 0.95
    min_latent_effect_spearman: float = 0.1
    min_cluster_ari: float = 0.5
    max_additive_r2_heldout: float = 0.98
    min_between_sample_var_share: float = 0.05


def latent_geometry_report(
    z: np.ndarray,
    records: list[ContextRecord],
    table: EffectProfileTable,
    n_clusters: int = 8,
    k_neighbors: int = 10,
    gate: GeometryGate | None = None,
    seed: int = 0,
) -> dict:
    """Full Phase 3 report with an explicit pass/fail verdict."""
    gate = gate or GeometryGate()
    rep: dict = {}
    rep["pca"] = pca_spectrum(z)
    rep["neighbor"] = neighbor_effect_consistency(z, table, k=k_neighbors, seed=seed)
    rep["distance_corr"] = latent_vs_effect_distance_correlation(z, table, seed=seed)
    rep["clusters"] = cluster_stability(z, n_clusters=n_clusters, seed=seed)
    rep["labels"] = label_noise_report(records, table)
    rep["additivity"] = additivity_check(records, len(z), seed=seed)
    rep["gain_signal"] = gain_signal_report(records)

    checks = {
        "F1_neighbor_effect_consistency": _le(
            rep["neighbor"].get("neighbor_consistency_ratio"),
            gate.max_neighbor_consistency_ratio,
        ),
        "F1_latent_effect_distance_corr": _ge(
            rep["distance_corr"].get("latent_effect_spearman"),
            gate.min_latent_effect_spearman,
        ),
        "cluster_stability": _ge(rep["clusters"].get("cluster_ari_mean"), gate.min_cluster_ari),
        "F2_non_additive_utility": _le(
            rep["additivity"].get("additive_r2_heldout"), gate.max_additive_r2_heldout
        ),
        "F3_label_signal": _ge(
            rep["labels"].get("between_sample_var_share"), gate.min_between_sample_var_share
        ),
    }
    rep["checks"] = checks
    rep["gate_passed"] = all(v is True for v in checks.values())
    rep["gate_thresholds"] = gate.__dict__
    return rep


def _ge(value, threshold):
    if value is None or not np.isfinite(value):
        return None  # undetermined, not a pass
    return bool(value >= threshold)


def _le(value, threshold):
    if value is None or not np.isfinite(value):
        return None
    return bool(value <= threshold)
