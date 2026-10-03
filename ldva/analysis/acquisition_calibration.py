"""Predicted vs realized acquisition gain (PLAN.md 17.2; SETUP.md 22).

The central question is *not* whether absolute predicted gains are right. It is
whether candidate acquisition compositions are ranked correctly, because that is
all the planner uses. So `calibration_report` leads with rank correlation and
top-k overlap, and reports absolute error as a secondary slope/bias fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ldva.training.metrics import pearson, spearman


@dataclass
class CalibrationRecord:
    """One candidate composition, predicted ahead of time and realized later."""

    allocation: np.ndarray
    predicted: float
    realized: float
    cost: float = 0.0
    label: str = ""
    extra: dict = field(default_factory=dict)


def calibration_report(records: list[CalibrationRecord], top_k: int = 3) -> dict:
    """Rank and absolute agreement between predicted and realized gain."""
    if len(records) < 2:
        return {"n_candidates": len(records)}
    pred = np.array([r.predicted for r in records], dtype=np.float64)
    real = np.array([r.realized for r in records], dtype=np.float64)

    k = min(top_k, len(records))
    pred_top = set(np.argsort(-pred)[:k].tolist())
    real_top = set(np.argsort(-real)[:k].tolist())
    best_pred_idx = int(np.argmax(pred))
    best_real_idx = int(np.argmax(real))

    # how much of the achievable realized gain the predicted-best choice gets
    span = float(real.max() - real.min())
    regret = float(real.max() - real[best_pred_idx])

    out = {
        "n_candidates": len(records),
        "spearman": spearman(pred, real),
        "pearson": pearson(pred, real),
        f"top{k}_overlap": len(pred_top & real_top) / k,
        "picked_the_best": bool(best_pred_idx == best_real_idx),
        "regret_absolute": regret,
        "regret_normalized": float(regret / span) if span > 1e-12 else 0.0,
        "realized_of_best_predicted": float(real[best_pred_idx]),
        "realized_best": float(real.max()),
        "realized_mean": float(real.mean()),
    }
    if np.std(pred) > 1e-12:
        slope, bias = np.polyfit(pred, real, 1)
        out.update({"calibration_slope": float(slope), "calibration_bias": float(bias)})
    return out


def cost_efficiency_report(records: list[CalibrationRecord]) -> dict:
    """Realized gain per unit cost (SETUP.md 31, 35)."""
    out = {}
    for r in records:
        if r.cost > 0:
            out[r.label or f"alloc_{np.sum(r.allocation)}"] = {
                "realized": r.realized,
                "cost": r.cost,
                "gain_per_cost": r.realized / r.cost,
            }
    return out


def cost_to_target(
    costs: np.ndarray, performance: np.ndarray, target: float
) -> float:
    """Cost at which performance first reaches `target` (SETUP.md 35).

    Linearly interpolates between the two bracketing points, and returns `inf`
    when the target is never reached so the caller cannot mistake a truncated
    curve for a cheap success.
    """
    costs = np.asarray(costs, dtype=np.float64)
    performance = np.asarray(performance, dtype=np.float64)
    order = np.argsort(costs)
    costs, performance = costs[order], performance[order]
    hit = np.nonzero(performance >= target)[0]
    if len(hit) == 0:
        return float("inf")
    i = int(hit[0])
    if i == 0:
        return float(costs[0])
    p0, p1 = performance[i - 1], performance[i]
    if abs(p1 - p0) < 1e-12:
        return float(costs[i])
    frac = (target - p0) / (p1 - p0)
    return float(costs[i - 1] + frac * (costs[i] - costs[i - 1]))


def acquisition_curve_report(
    budgets: np.ndarray,
    performance: np.ndarray,
    costs: np.ndarray | None = None,
    targets: tuple[float, ...] = (),
) -> dict:
    """Summarize a performance-vs-budget curve (PLAN.md 17.3)."""
    budgets = np.asarray(budgets, dtype=np.float64)
    performance = np.asarray(performance, dtype=np.float64)
    out = {
        "budgets": budgets.tolist(),
        "performance": performance.tolist(),
        "final_performance": float(performance[-1]),
        "total_improvement": float(performance[-1] - performance[0]),
        # area under the curve, normalized by budget span: rewards getting
        # good early rather than only ending well
        "auc_normalized": float(
            np.trapezoid(performance, budgets) / max(budgets[-1] - budgets[0], 1e-12)
        )
        if len(budgets) > 1
        else float(performance[0]),
    }
    if costs is not None:
        out["costs"] = np.asarray(costs, dtype=np.float64).tolist()
    for t in targets:
        src = np.asarray(costs if costs is not None else budgets, dtype=np.float64)
        out[f"cost_to_target_{t}"] = cost_to_target(src, performance, t)
    return out
