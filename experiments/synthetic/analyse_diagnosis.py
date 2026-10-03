"""Which conditions produce a positive, null, or negative criterion 5?

`diagnose_criteria.py` sweeps seeds x step lengths and records, for every cell,
the two criteria plus every quantity that could plausibly explain them. This
script answers the question that matters: *under what shared conditions* does
the predicted-vs-realized correlation come out positive, null, or negative.

A mean per bucket is not enough on its own - with a handful of cells per bucket
any variable will differ somewhat - so each candidate explanation is reported
three ways and should only be believed when they agree:

1. its mean within each outcome bucket,
2. its rank correlation with the criterion across all cells,
3. whether the buckets' ranges overlap.

Runs on a partial sweep, so a long run can be read while it is still going.

    python experiments/synthetic/analyse_diagnosis.py --report runs/E0_diag/diagnose_criteria.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ldva.utils import load_json, save_json  # noqa: E402

#: candidate explanations, as (label, extractor). Each is a quantity measured
#: *before* the criteria are evaluated, so a correlation with a criterion is a
#: usable lead rather than a restatement of the outcome.
FEATURES: list[tuple[str, callable]] = [
    ("delta_scale (step length)", lambda c: c["delta_scale"]),
    ("overshoot (realized/planned)", lambda c: c["criterion6"]["overshoot"]),
    ("realized |displacement|", lambda c: c["criterion6"]["realized_displacement_mean"]),
    ("planned step", lambda c: c["criterion6"]["planned_step_mean"]),
    ("noise/signal", lambda c: c["criterion5"].get("noise_over_signal", np.nan)),
    ("frac draws diverged", lambda c: c["criterion5"].get("frac_draws_diverged", np.nan)),
    ("predicted spread", lambda c: c["criterion5"].get("predicted_spread", np.nan)),
    ("n distinct allocations", lambda c: c["criterion5"].get("n_candidates", np.nan)),
    ("n directions", lambda c: c["n_directions"]),
    ("latent norm mean", lambda c: c.get("latent_norm_mean", np.nan)),
    ("effect spearman (model fit)", lambda c: c.get("effect_spearman", np.nan)),
    ("gain within-ckpt r2", lambda c: c.get("gain_within_r2", np.nan)),
]


def _bucket(v: float, pos: float, neg: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if v > pos:
        return "positive"
    if v < neg:
        return "negative"
    return "null"


def _rho(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() >= 3 else float("nan")


def analyse(cells: list[dict], pos: float, neg: float) -> dict:
    ok = [c for c in cells if "error" not in c and "criterion5" in c
          and np.isfinite(c["criterion5"].get("spearman_mean", np.nan))]
    if len(ok) < 2:
        return {"n_cells": len(ok), "note": "not enough cells yet"}

    c5 = np.array([c["criterion5"]["spearman_mean"] for c in ok])
    c5_one = np.array([c["criterion5"].get("spearman_single_draw", np.nan) for c in ok])
    c6 = np.array([c["criterion6"]["direction_cosine_mean"] for c in ok])
    buckets = np.array([_bucket(v, pos, neg) for v in c5])

    feat_rows = []
    for label, fn in FEATURES:
        vals = np.array([fn(c) for c in ok], dtype=np.float64)
        row = {
            "feature": label,
            "rho_vs_c5": _rho(vals, c5),
            "rho_vs_c6": _rho(vals, c6),
        }
        for b in ("positive", "null", "negative"):
            sel = vals[buckets == b]
            sel = sel[np.isfinite(sel)]
            row[f"mean_{b}"] = float(sel.mean()) if sel.size else float("nan")
            row[f"range_{b}"] = (
                [float(sel.min()), float(sel.max())] if sel.size else [float("nan")] * 2)
            row[f"n_{b}"] = int(sel.size)
        feat_rows.append(row)

    # rank the leads by how strongly they track criterion 5
    feat_rows.sort(key=lambda r: -abs(r["rho_vs_c5"]) if np.isfinite(r["rho_vs_c5"]) else 0)

    by_delta = {}
    for c in ok:
        d = c["delta_scale"]
        by_delta.setdefault(d, []).append(c)
    delta_rows = []
    for d in sorted(by_delta):
        g = by_delta[d]
        v5 = np.array([x["criterion5"]["spearman_mean"] for x in g])
        v6 = np.array([x["criterion6"]["direction_cosine_mean"] for x in g])
        ov = np.array([x["criterion6"]["overshoot"] for x in g])
        delta_rows.append({
            "delta_scale": d,
            "n_seeds": len(g),
            "c5_mean": float(np.nanmean(v5)),
            "c5_min": float(np.nanmin(v5)),
            "c5_max": float(np.nanmax(v5)),
            "c5_n_positive": int((v5 > pos).sum()),
            "c5_n_negative": int((v5 < neg).sum()),
            "c6_mean": float(np.nanmean(v6)),
            "c6_n_pass": int((v6 > 0.3).sum()),
            "overshoot_mean": float(np.nanmean(ov)),
        })

    return {
        "n_cells": len(ok),
        "thresholds": {"positive_above": pos, "negative_below": neg},
        "counts": {b: int((buckets == b).sum()) for b in ("positive", "null", "negative")},
        "c5_mean": float(np.nanmean(c5)),
        "c5_range": [float(np.nanmin(c5)), float(np.nanmax(c5))],
        # is criterion 5 even a stable statistic? if averaging over draws does
        # not systematically help, the statistic itself is too noisy to read
        # from a single run, which is a conclusion in its own right
        "averaging_effect_mean": float(np.nanmean(c5 - c5_one)),
        "averaging_helped_in_n_cells": int(np.nansum(c5 > c5_one)),
        "features": feat_rows,
        "by_delta_scale": delta_rows,
    }


#: the one line `diagnose_criteria.py` prints per cell. Parsing it lets a sweep
#: be read while it is still running, and recovers a run whose JSON is only
#: written at the end - which is how the first 24-cell sweep was launched.
_CELL_RE = re.compile(
    r"seed\s+(?P<seed>\d+)\s+delta=(?P<delta>[\d.]+):\s+"
    r"C5 spearman\(mean\)=(?P<c5>[+-][\d.]+)\s+"
    r"\(single-draw\s+(?P<c5one>[+-][\d.]+),\s+"
    r"noise/signal\s+(?P<noise>[\d.]+)x\)\s+\|\s+"
    r"C6 cos=(?P<c6>[+-][\d.]+)\s+overshoot=(?P<over>[\d.]+)x"
)


def cells_from_log(path: Path) -> list[dict]:
    """Rebuild the cells a sweep has finished so far from its stdout log.

    Only the quantities the progress line carries are recovered, so the feature
    table is narrower than from the JSON. The step length, overshoot and
    noise/signal - the three leads that matter - are all present.
    """
    cells = []
    for line in Path(path).read_text().splitlines():
        m = _CELL_RE.search(line)
        if not m:
            continue
        g = m.groupdict()
        cells.append({
            "seed": int(g["seed"]),
            "delta_scale": float(g["delta"]),
            "n_directions": float("nan"),
            "criterion5": {
                "spearman_mean": float(g["c5"]),
                "spearman_single_draw": float(g["c5one"]),
                "noise_over_signal": float(g["noise"]),
            },
            "criterion6": {
                "direction_cosine_mean": float(g["c6"]),
                "overshoot": float(g["over"]),
                "realized_displacement_mean": float("nan"),
                "planned_step_mean": float("nan"),
            },
            "from_log": True,
        })
    return cells


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=str, default="runs/E0_diag/diagnose_criteria.json")
    ap.add_argument("--log", type=str, default=None,
                    help="read cells from a sweep's stdout log instead of its JSON, "
                         "so a run in progress can be analysed")
    ap.add_argument("--positive-above", type=float, default=0.3,
                    help="PLAN.md 14 criterion 5 threshold")
    ap.add_argument("--negative-below", type=float, default=-0.1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.log:
        cells = cells_from_log(Path(args.log))
        print(f"read {len(cells)} finished cell(s) from {args.log}")
    else:
        cells = load_json(args.report).get("cells", [])
    a = analyse(cells, args.positive_above, args.negative_below)
    if a.get("n_cells", 0) < 2:
        print(f"only {a.get('n_cells', 0)} usable cell(s) so far; nothing to pool yet")
        return a

    w = 112
    print("\n" + "=" * w)
    print(f"CONDITIONS FOR A POSITIVE / NULL / NEGATIVE CRITERION 5   ({a['n_cells']} cells)")
    print("=" * w)
    c = a["counts"]
    print(f"  outcome buckets: {c['positive']} positive (>{args.positive_above}), "
          f"{c['null']} null, {c['negative']} negative (<{args.negative_below})")
    print(f"  criterion 5 across cells: mean {a['c5_mean']:+.3f}, "
          f"range [{a['c5_range'][0]:+.3f}, {a['c5_range'][1]:+.3f}]")
    print(f"  averaging over draws helped in {a['averaging_helped_in_n_cells']}/"
          f"{a['n_cells']} cells (mean effect {a['averaging_effect_mean']:+.3f})")

    print("\n  -- per step length ------------------------------------------------")
    print(f"  {'delta':>7s} {'n':>3s} {'C5 mean':>9s} {'C5 range':>18s} {'+/-':>7s} "
          f"{'C6 mean':>9s} {'C6 pass':>8s} {'overshoot':>10s}")
    for r in a["by_delta_scale"]:
        print(f"  {r['delta_scale']:>7.2f} {r['n_seeds']:>3d} {r['c5_mean']:>+9.3f} "
              f"[{r['c5_min']:>+7.3f},{r['c5_max']:>+7.3f}] "
              f"{str(r['c5_n_positive']) + '/' + str(r['c5_n_negative']):>7s} "
              f"{r['c6_mean']:>+9.3f} {str(r['c6_n_pass']) + '/' + str(r['n_seeds']):>8s} "
              f"{r['overshoot_mean']:>9.2f}x")

    print("\n  -- candidate explanations, ranked by |rho| with criterion 5 -------")
    print(f"  {'feature':<30s} {'rho C5':>8s} {'rho C6':>8s} "
          f"{'mean(pos)':>11s} {'mean(null)':>11s} {'mean(neg)':>11s}")
    for r in a["features"]:
        print(f"  {r['feature']:<30s} {r['rho_vs_c5']:>+8.3f} {r['rho_vs_c6']:>+8.3f} "
              f"{r['mean_positive']:>+11.3f} {r['mean_null']:>+11.3f} "
              f"{r['mean_negative']:>+11.3f}")
    print("=" * w + "\n")

    if args.out:
        save_json(a, args.out)
        print(f"  written: {args.out}")
    return a


if __name__ == "__main__":
    main()
