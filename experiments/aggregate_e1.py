"""Merge per-method acquisition reports into one comparison (PLAN.md 17, 18).

`run_e1_dmc.sh` runs one method per invocation so a crash cannot lose the rest,
which leaves one report per method. This joins them and applies the same
resolution test the single-process loop applies, so a ranking read off the
merged table is held to the same standard: a difference counts only if it
exceeds twice the standard error of a method's mean, estimated from the spread
across independently trained evaluation policies.

It also reports the two things PLAN.md 17 E1 asks for besides the curve:
predicted-vs-realized gain, and metadata-direction control.

    python experiments/aggregate_e1.py --root runs/E1_dmc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldva.utils import load_json, run_provenance, save_json  # noqa: E402


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency at import time."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a[ok], b[ok]).statistic)


def collect(root: Path) -> tuple[dict, dict, str]:
    """Read every `acquisition_loop_report.json` under `root`."""
    summary, runs, primary = {}, {}, "utility"
    # per-method subdirectories (how `run_e1_dmc.sh` lays them out), plus a
    # report written directly into `root` by a single-process run
    paths = sorted(root.glob("*/acquisition_loop_report.json"))
    direct = root / "acquisition_loop_report.json"
    if direct.exists():
        paths.append(direct)
    for path in paths:
        rep = load_json(path)
        primary = rep.get("primary_metric", primary)
        for m, s in rep.get("summary", {}).items():
            summary[m] = s
        for r in rep.get("runs", []):
            runs.setdefault(r["method"], []).append(r)
    return summary, runs, primary


def calibration(runs: dict) -> dict:
    """Predicted allocation utility vs realized change in the primary metric.

    PLAN.md 17 E1 lists "predicted vs realized gain" as a measurement in its own
    right, and PLAN.md 14's Stage 1 exit criterion 2 requires that the
    acquisition ranking predict realized gain. Pooled over every
    (method, seed, round) that recorded both, because within a single method the
    predicted values barely vary.
    """
    pred, realized, tags = [], [], []
    for method, rs in runs.items():
        for r in rs:
            h = r["history"]
            for i in range(len(h) - 1):
                p = h[i].get("predicted_utility")
                a = h[i].get("rollout_return", h[i].get("utility"))
                b = h[i + 1].get("rollout_return", h[i + 1].get("utility"))
                if p is None or a is None or b is None:
                    continue
                pred.append(float(p))
                realized.append(float(b) - float(a))
                tags.append(f"{method}/s{r['seed']}/r{i}")
    rho = _spearman(np.array(pred), np.array(realized))
    return {
        "n_pairs": len(pred),
        "spearman_predicted_vs_realized": rho,
        "rule": "PLAN.md 14 Stage 1 exit criterion 2: acquisition ranking "
                "predicts realized gain (spearman > 0.3)",
        "passed": bool(np.isfinite(rho) and rho > 0.3),
        "predicted": pred,
        "realized": realized,
        "tags": tags,
    }


def direction_control(runs: dict) -> dict:
    """Did collected data move the latents the way the plan asked?

    Only LDVA plans a latent direction, so this is reported per method but is
    meaningful only for the LDVA variants; the others are recorded as NaN
    rather than silently omitted.
    """
    out = {}
    for method, rs in runs.items():
        cos = [h.get("direction_control_cosine") for r in rs for h in r["history"]]
        cos = [c for c in cos if c is not None and np.isfinite(c)]
        out[method] = {
            "mean_cosine": float(np.mean(cos)) if cos else float("nan"),
            "frac_positive": float(np.mean(np.array(cos) > 0)) if cos else float("nan"),
            "n_rounds_measured": len(cos),
        }
    return out


def resolution(summary: dict, primary: str) -> dict:
    present = list(summary)
    if len(present) < 2:
        # keep every key the printer reads, so a partial run (one method
        # finished, the rest still going) still prints instead of crashing
        return {
            "metric": primary,
            "resolvable": False,
            "reason": "fewer than two methods",
            "method_spread": float("nan"),
            "standard_error_per_method": float("nan"),
            "n_seeds": max((summary[m]["n_seeds"] for m in present), default=0),
        }
    finals = np.array([summary[m]["final_mean"] for m in present], dtype=np.float64)
    noise = np.array([summary[m]["final_std_across_policies"] for m in present],
                     dtype=np.float64)
    n_seeds = max(summary[m]["n_seeds"] for m in present)
    spread = float(np.nanmax(finals) - np.nanmin(finals))
    sem = float(np.nanmean(noise) / np.sqrt(max(n_seeds, 1)))
    return {
        "metric": primary,
        "resolvable": bool(np.isfinite(spread) and spread > 2.0 * sem),
        "method_spread": spread,
        "standard_error_per_method": sem,
        "n_seeds": int(n_seeds),
        "rule": "spread between methods > 2 x standard error of a method's mean",
    }


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs/E1_dmc")
    args = ap.parse_args()
    root = Path(args.root)

    summary, runs, primary = collect(root)
    if not summary:
        print(f"no reports found under {root}")
        return {}

    res = resolution(summary, primary)
    cal = calibration(runs)
    ctrl = direction_control(runs)
    report = {
        "root": str(root),
        "primary_metric": primary,
        "summary": summary,
        "resolution": res,
        "calibration": cal,
        "direction_control": ctrl,
        "provenance": run_provenance({"experiment": "E1_dmc"}),
    }
    save_json(report, root / "E1_report.json")

    w = 108
    print("\n" + "=" * w)
    print("E1 - DMC reacher-easy closed-loop acquisition (PLAN.md 17)")
    print(f"primary metric: {primary}   seeds: {res.get('n_seeds')}")
    print("=" * w)
    print(f"  {'method':<28s} {'class':<14s} {'final':>11s} {'+/- seeds':>11s} "
          f"{'+/- policies':>13s} {'improvement':>12s} {'dir cos':>9s}")
    order = sorted(summary, key=lambda m: -summary[m]["final_mean"])
    for m in order:
        s = summary[m]
        c = ctrl.get(m, {}).get("mean_cosine", float("nan"))
        print(f"  {m:<28s} {s['method_class']:<14s} {s['final_mean']:>+11.3f} "
              f"{s['final_std_across_seeds']:>11.3f} "
              f"{s['final_std_across_policies']:>13.3f} "
              f"{s['total_improvement_mean']:>+12.3f} {c:>+9.3f}")
    print("-" * w)
    if res["resolvable"]:
        print(f"  RESOLVABLE: spread {res['method_spread']:.3f} > "
              f"2 x s.e. {2 * res['standard_error_per_method']:.3f}")
    else:
        print(f"  NOT RESOLVABLE: spread {res['method_spread']:.3f} vs "
              f"2 x s.e. {2 * res['standard_error_per_method']:.3f}")
        print("  The ranking above must NOT be read as a result.")
    print(f"  predicted vs realized gain: spearman {cal['spearman_predicted_vs_realized']:+.3f} "
          f"over {cal['n_pairs']} (method, seed, round) pairs -> "
          f"{'PASS' if cal['passed'] else 'FAIL'} (PLAN.md 14 exit criterion 2)")
    print(f"  report: {root / 'E1_report.json'}")
    print("=" * w + "\n")
    return report


if __name__ == "__main__":
    main()
