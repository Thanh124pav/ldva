"""Prediction-quality metrics (PLAN.md 18, 17.1).

Rank correlation is reported next to MSE everywhere because the acquisition
planner only ever *compares* candidate batches - PLAN.md 17.2 is explicit that
"can the model rank future acquisition batches" is the core test, not whether
absolute effect values are right.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    v = fn(a, b)[0]
    return float(v) if np.isfinite(v) else float("nan")


def spearman(a, b) -> float:
    return _safe_corr(spearmanr, a, b)


def pearson(a, b) -> float:
    return _safe_corr(pearsonr, a, b)


def regression_metrics(pred: np.ndarray, target: np.ndarray, prefix: str = "") -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    resid = pred - target
    var = float(np.var(target))
    out = {
        f"{prefix}mse": float(np.mean(resid**2)),
        f"{prefix}mae": float(np.mean(np.abs(resid))),
        f"{prefix}spearman": spearman(pred, target),
        f"{prefix}pearson": pearson(pred, target),
        # variance explained relative to predicting the mean; negative means
        # the model is worse than the constant baseline
        f"{prefix}r2": float(1.0 - np.mean(resid**2) / var) if var > 1e-12 else float("nan"),
    }
    return out


def constant_baseline_metrics(target: np.ndarray, prefix: str = "const_") -> dict:
    """The baseline a contextual model must beat (success criterion 1)."""
    target = np.asarray(target, dtype=np.float64).ravel()
    return {
        f"{prefix}mse": float(np.var(target)),
        f"{prefix}mae": float(np.mean(np.abs(target - target.mean()))),
    }


def fit_per_sample_scalar(
    sample_ids: np.ndarray, target: np.ndarray
) -> tuple[dict[int, float], float]:
    """Fit the "one value per sample" hypothesis on training data.

    Returns the per-sample mean effect and a global fallback for samples the
    split never showed.
    """
    sample_ids = np.asarray(sample_ids).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    uniq, inv = np.unique(sample_ids, return_inverse=True)
    sums = np.bincount(inv, weights=target, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    means = sums / np.maximum(counts, 1)
    return {int(u): float(m) for u, m in zip(uniq, means)}, float(target.mean())


def per_sample_scalar_baseline_metrics(
    sample_ids: np.ndarray,
    target: np.ndarray,
    prefix: str = "scalar_",
    fit: tuple[dict[int, float], float] | None = None,
) -> dict:
    """The strongest form of "a sample has one fixed value".

    With `fit` supplied (per-sample means estimated on the *training* contexts)
    this is a genuine out-of-sample baseline, directly comparable to the model's
    validation numbers. That is the comparison that belongs in success criterion
    1 of PLAN.md 14 and ablation 1 of PLAN.md 18.

    Without `fit` it falls back to fitting on the evaluation data itself. That
    hindsight variant is reported too, as an optimistic reference, but it is not
    a fair comparison: with few contexts per sample it is close to memorizing
    the targets it is being scored on.
    """
    sample_ids = np.asarray(sample_ids).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if fit is None:
        table, default = fit_per_sample_scalar(sample_ids, target)
        out_of_sample = False
    else:
        table, default = fit
        out_of_sample = True
    pred = np.array([table.get(int(i), default) for i in sample_ids], dtype=np.float64)
    out = regression_metrics(pred, target, prefix=prefix)
    out[f"{prefix}n_unique_samples"] = int(len(np.unique(sample_ids)))
    out[f"{prefix}out_of_sample"] = out_of_sample
    out[f"{prefix}frac_unseen_samples"] = float(
        np.mean([int(i) not in table for i in sample_ids])
    )
    return out


def _fit_additive(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def within_group_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    prefix: str = "within_",
    min_group_size: int = 2,
) -> dict:
    """Prediction quality *after* removing each group's mean.

    Needed for batch-gain evaluation. `U(Update(theta, B)) - U(theta)` depends
    far more on which checkpoint `theta` is than on which samples are in `B`:
    on the Stage 0 world 97.9% of the gain variance is between-checkpoint and
    only 2.1% is within. A plain MSE over all contexts therefore scores
    "can you identify the checkpoint", and an additive and a set-level utility
    head look identical because both nail that part.

    Centring per checkpoint isolates the composition-dependent signal, which is
    the only part where set-level structure can possibly help - so this is the
    comparison behind ablation 3 of PLAN.md 18 and success criterion 2 of
    PLAN.md 14.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    keep = counts[inv] >= min_group_size
    if keep.sum() < 3:
        return {f"{prefix}n": int(keep.sum())}

    p_c = pred[keep].copy()
    t_c = target[keep].copy()
    inv_k = inv[keep]
    for g in np.unique(inv_k):
        m = inv_k == g
        p_c[m] -= p_c[m].mean()
        t_c[m] -= t_c[m].mean()
    out = regression_metrics(p_c, t_c, prefix=prefix)
    out[f"{prefix}n"] = int(keep.sum())
    out[f"{prefix}n_groups"] = int(len(np.unique(inv_k)))
    out[f"{prefix}target_var"] = float(np.var(t_c))
    return out


def group_variance_decomposition(values: np.ndarray, groups: np.ndarray) -> dict:
    """Split the variance of `values` into between- and within-group parts."""
    values = np.asarray(values, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    uniq, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, weights=values, minlength=len(uniq)) / np.maximum(
        np.bincount(inv, minlength=len(uniq)), 1
    )
    within = values - means[inv]
    total = float(np.var(values))
    wv = float(np.var(within))
    return {
        "total_var": total,
        "within_group_var": wv,
        "between_group_var": max(total - wv, 0.0),
        "within_group_share": float(wv / total) if total > 1e-20 else 0.0,
        "n_groups": int(len(uniq)),
    }


def additivity_report(
    gains: np.ndarray,
    batch_sample_ids: list[np.ndarray],
    n_samples: int,
    ridge: float = 1.0,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Failure mode F2: how additive is the true batch gain?

    Fits `V(B) = sum_{i in B} v_i + c` to the observed gains and reports how
    much variance it explains. The headline number is **held out**: the additive
    model has one free parameter per sample, so an in-sample fit is perfect
    whenever `n_contexts <= n_samples` and would always "prove" additivity.
    We therefore cross-validate over contexts and report the in-sample value
    only as a reference.

    A high `additive_r2_heldout` means a per-sample scalar already explains
    batch utility, and the set-level motivation of LDVA is unsupported on this
    data. `contexts_per_param` says whether the test had the power to tell.
    """
    gains = np.asarray(gains, dtype=np.float64).ravel()
    n_ctx = len(gains)
    base = {
        "n_contexts_additivity": int(n_ctx),
        "contexts_per_param": float(n_ctx / max(n_samples + 1, 1)),
    }
    if n_ctx < 3:
        return {**base, "additive_r2_heldout": float("nan"), "additive_r2_insample": float("nan")}

    # composition indicator matrix plus an intercept, so batch-size effects are
    # not charged to additivity
    X = np.zeros((n_ctx, n_samples + 1))
    for r, ids in enumerate(batch_sample_ids):
        X[r, np.asarray(ids, dtype=np.int64)] = 1.0
    X[:, -1] = 1.0

    w_all = _fit_additive(X, gains, ridge)
    resid_in = gains - X @ w_all
    var = float(np.var(gains))

    rng = np.random.default_rng(seed)
    folds = rng.permutation(n_ctx) % max(min(n_folds, n_ctx), 2)
    pred_oof = np.zeros(n_ctx)
    for f in range(folds.max() + 1):
        te = folds == f
        tr = ~te
        if tr.sum() < 2 or te.sum() == 0:
            pred_oof[te] = gains[tr].mean() if tr.any() else gains.mean()
            continue
        pred_oof[te] = X[te] @ _fit_additive(X[tr], gains[tr], ridge)
    resid_oof = gains - pred_oof

    r2 = lambda res: float(1.0 - np.mean(res**2) / var) if var > 1e-12 else float("nan")  # noqa: E731
    return {
        **base,
        "additive_r2_heldout": r2(resid_oof),
        "additive_r2_insample": r2(resid_in),
        "nonadditive_residual_frac": (
            float(1.0 - r2(resid_oof)) if np.isfinite(r2(resid_oof)) else float("nan")
        ),
        "additive_fit_spearman_heldout": spearman(pred_oof, gains),
        "gain_var": var,
    }
