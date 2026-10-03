"""Figures (PLAN.md 15, 18). Matplotlib with the Agg backend, no interactivity.

The headline figure for the paper is `plot_performance_vs_cost` - PLAN.md 18
names "real-robot success rate vs monetary acquisition cost" as the main plot,
and the simulation stages produce the same shape with cost standing in for
number of trajectories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latent_pca(
    z: np.ndarray,
    labels: np.ndarray | None = None,
    path: str | Path = "latent_pca.png",
    title: str = "Effect latent space (PCA)",
) -> Path:
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    proj = zc @ vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    if labels is None:
        ax[0].scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.6)
    else:
        for c in np.unique(labels):
            m = labels == c
            ax[0].scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.7, label=f"cluster {c}")
        ax[0].legend(fontsize=7, markerscale=1.5)
    ax[0].set(xlabel="PC1", ylabel="PC2", title=title)

    evr = (s**2) / np.sum(s**2)
    ax[1].plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "o-")
    ax[1].axhline(0.90, ls="--", c="gray", lw=1, label="rho = 0.90")
    ax[1].set(xlabel="component", ylabel="cumulative explained variance",
              title="PCA spectrum", ylim=(0, 1.02))
    ax[1].legend(fontsize=8)
    return _save(fig, path)


def plot_effect_predictions(
    pred: np.ndarray,
    target: np.ndarray,
    path: str | Path = "effect_pred.png",
    title: str = "Contextual effect prediction",
) -> Path:
    pred, target = np.ravel(pred), np.ravel(target)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target, pred, s=8, alpha=0.4)
    lim = [min(target.min(), pred.min()), max(target.max(), pred.max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set(xlabel="target effect", ylabel="predicted effect", title=title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_contexts_per_sample(
    counts: np.ndarray, min_target: int = 20, path: str | Path = "coverage.png"
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=min(30, max(5, len(np.unique(counts)))), color="steelblue")
    ax.axvline(min_target, ls="--", c="crimson", label=f"target = {min_target}")
    ax.set(xlabel="contexts per sample", ylabel="samples",
           title="Multi-context supervision coverage")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_allocation(
    allocations: dict[str, np.ndarray],
    path: str | Path = "allocations.png",
    direction_labels: list[str] | None = None,
) -> Path:
    """Grouped bars of how each planner split the same budget."""
    names = list(allocations)
    A = len(next(iter(allocations.values())))
    x = np.arange(A)
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(max(7, A * 0.7), 4))
    for i, n in enumerate(names):
        ax.bar(x + i * w - 0.4 + w / 2, allocations[n], width=w, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(direction_labels or [f"a{j}" for j in range(A)], rotation=45, ha="right")
    ax.set(xlabel="candidate direction", ylabel="allocated samples",
           title="Budget allocation across acquisition directions")
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_solver_comparison(
    values: dict[str, float],
    path: str | Path = "solvers.png",
    reference: str | None = "exact",
) -> Path:
    names = list(values)
    vals = [values[n] for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    colors = ["crimson" if n == reference else "steelblue" for n in names]
    ax.bar(names, vals, color=colors)
    if reference in values:
        ax.axhline(values[reference], ls="--", c="crimson", lw=1, label=f"{reference}")
        ax.legend(fontsize=8)
    ax.set(ylabel="predicted utility V_hat", title="Planner comparison")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def plot_predicted_vs_realized(
    predicted: np.ndarray,
    realized: np.ndarray,
    labels: list[str] | None = None,
    path: str | Path = "calibration.png",
) -> Path:
    predicted, realized = np.ravel(predicted), np.ravel(realized)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(predicted, realized, s=40)
    if labels is not None:
        for p, r, t in zip(predicted, realized, labels):
            ax.annotate(t, (p, r), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if np.std(predicted) > 1e-12:
        sl, bi = np.polyfit(predicted, realized, 1)
        xs = np.linspace(predicted.min(), predicted.max(), 50)
        ax.plot(xs, sl * xs + bi, "r--", lw=1, label=f"fit slope={sl:.2f}")
        ax.legend(fontsize=8)
    ax.set(xlabel="predicted gain V_hat", ylabel="realized gain",
           title="Acquisition calibration")
    return _save(fig, path)


def plot_performance_vs_cost(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path = "performance_vs_cost.png",
    xlabel: str = "acquisition cost",
    ylabel: str = "policy performance",
    target: float | None = None,
) -> Path:
    """The headline acquisition figure (PLAN.md 18)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (x, y) in curves.items():
        ax.plot(np.asarray(x), np.asarray(y), "o-", label=name)
    if target is not None:
        ax.axhline(target, ls=":", c="gray", lw=1, label=f"target = {target:g}")
    ax.set(xlabel=xlabel, ylabel=ylabel, title="Performance vs acquisition budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_direction_alignment(
    desired_cos: np.ndarray, path: str | Path = "direction_alignment.png"
) -> Path:
    """Histogram of cosine(desired latent direction, realized movement)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.ravel(desired_cos), bins=20, range=(-1, 1), color="seagreen")
    ax.axvline(0, ls="--", c="gray", lw=1)
    ax.axvline(float(np.mean(desired_cos)), ls="-", c="crimson", lw=1.5,
               label=f"mean = {np.mean(desired_cos):.2f}")
    ax.set(xlabel="cosine(desired, realized)", ylabel="count",
           title="Metadata-to-latent direction control")
    ax.legend(fontsize=8)
    return _save(fig, path)
