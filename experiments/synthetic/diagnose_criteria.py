"""Why criteria 5 and 6 fail, and which knob controls them (PLAN.md 14, 19).

E0 on three seeds gave:

    seed        criterion 5 (predicted -> realized)     criterion 6 (direction)
    0           +0.034                                  +0.314  PASS
    1           -0.076                                  +0.459  PASS
    2           -0.569                                  +0.012  FAIL

Three hypotheses were on the table for criterion 5: bad clustering, too few
compositions (15), or something else. Reading the raw records settles it, and
this script turns that reading into a measurement.

**What the E0 records already show.** The 15 "compositions" are not 15
independent samples - they are the allocations that 6 solvers and 9 baselines
*chose*, so they are near-optimal by construction and many are identical. On
seed 2, nine of the fifteen picked the same allocation `000000800000`. Their
predicted value is therefore identical (+2.8381) while their realized gains
range from -26.6 to +2.6. That spread is pure measurement noise: same
allocation, same prediction. Ranking it against a constant is meaningless, and
the noise-to-signal ratio ordered the seeds exactly as their Spearman did
(0.1x, 3.9x, 7.8x -> +0.03, -0.08, -0.57).

The noise has a specific source. `predicted` is an expectation: the objective
Monte-Carlo averages over the *distribution* of latents a direction could
yield. `realized` was a single draw from that distribution - one random anchor
choice in `plan_direction`, one collection, one SGD update at lr=0.3 that can
and does diverge. Comparing an expectation with one sample of a heavy-tailed
variable is the bug, not the method.

So this script measures `realized` the way `predicted` is defined:

1. **de-duplicate** allocations, so one allocation contributes one point;
2. average realized gain over `--repeats` *independent plan draws* (new anchors,
   new collection, new update each time), which is what makes it an expectation;
3. **widen the candidate set** with random allocations spanning the simplex, so
   the correlation is not computed inside the narrow band the optimizers picked;
4. report Spearman as a function of the number of repeats and of the number of
   candidates, which separates hypothesis 2 (too few points) from the noise
   explanation;
5. flag divergent updates instead of letting them silently set the rank order.

**Criterion 6 has a different and simpler cause: overshoot.** The planner asks
for a latent step of length ~0.2; the data that actually arrives moves the
latents 1.2x / 2.4x / 3.5x that far on seeds 0 / 1 / 2, and the direction
cosine collapses exactly where the overshoot is largest. The local linear
Jacobian is fitted locally and is not valid over a step three times longer than
requested - note that seed 2 has the *best* held-out Jacobian R2 (0.429) and
the worst control (+0.012), so the fit is not the problem, the step length is.
`--delta-scales` sweeps that knob to test whether shrinking the step restores
control, which is the "can the seed-2 change be reversed" question.

    python experiments/synthetic/diagnose_criteria.py --seeds 0,1,2 --repeats 6
    python experiments/synthetic/diagnose_criteria.py --seeds 0,1,2,3,4,5,6,7 \
        --delta-scales 0.1,0.2,0.4 --repeats 6
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ldva.acquisition.clustering import ClusteringConfig, LatentClustering  # noqa: E402
from ldva.acquisition.directions import DirectionConfig, DirectionGenerator  # noqa: E402
from ldva.acquisition.greedy import greedy_search  # noqa: E402
from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig  # noqa: E402
from ldva.acquisition.metadata_mapper import (  # noqa: E402
    ActionabilityConfig,
    MetadataMapper,
    MetadataMapperConfig,
    filter_actionable_directions,
)
from ldva.acquisition.objective import (  # noqa: E402
    AllocationObjective,
    BudgetSpec,
    ObjectiveConfig,
)
from ldva.analysis.wandb_logger import make_logger  # noqa: E402
from ldva.envs.synthetic.oracle import (  # noqa: E402
    OracleConfig,
    SyntheticAcquisitionOracle,
    measure_realized_latent_movement,
)
from ldva.policy.checkpoints import PolicyContextRef  # noqa: E402
from ldva.utils import run_provenance, save_json  # noqa: E402


def _load_fixture_module():
    """Reuse `run_ablations.Fixture` rather than rebuilding the pipeline.

    It already constructs exactly the world, dataset, supervision set and data
    model this diagnosis needs, and reusing it guarantees the diagnosis runs on
    the same setup the ablations do.
    """
    path = ROOT / "experiments" / "synthetic" / "run_ablations.py"
    spec = importlib.util.spec_from_file_location("ldva_abl_fixture", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ldva_abl_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- candidate allocations ------------------------------------------------


def _random_allocations(
    n_directions: int, budget: int, n: int, rng: np.random.Generator
) -> list[tuple[int, ...]]:
    """Allocations spread over the simplex, to defeat range restriction.

    The solver-chosen allocations all sit in a narrow near-optimal band, and a
    rank correlation computed inside that band is mostly noise even when the
    model is good. Random allocations reach the bad part of the space, where a
    useful predictor should be clearly right.
    """
    out = set()
    tries = 0
    while len(out) < n and tries < 40 * n:
        tries += 1
        # a Dirichlet-like split gives both concentrated and spread allocations
        w = rng.dirichlet(np.full(n_directions, rng.choice([0.3, 1.0, 3.0])))
        alloc = np.floor(w * budget).astype(np.int64)
        short = budget - int(alloc.sum())
        for _ in range(short):
            alloc[rng.integers(n_directions)] += 1
        out.add(tuple(int(x) for x in alloc))
    return sorted(out)


def _solver_allocations(objective, budget: int, n_directions: int) -> dict:
    """The allocations the planners actually pick, for reference."""
    out = {}
    try:
        out["greedy"] = tuple(int(x) for x in greedy_search(objective).best_allocation)
    except Exception:
        pass
    for name, fn in (("uniform", None),):
        if fn is None:
            a = np.zeros(n_directions, dtype=np.int64)
            for i in range(budget):
                a[i % n_directions] += 1
            out[name] = tuple(int(x) for x in a)
    return out


# ---- the two criteria -----------------------------------------------------


def criterion6(fx, model, z_all, clusters, mapper, directions, delta_scale: float) -> dict:
    """Realized direction control, plus the overshoot that explains it."""
    rng = fx.seeds.rng("acquisition")
    cos, disp, planned = [], [], []
    for d in directions:
        plan = mapper.plan_direction(d, 4, fx.store.metadata, rng=rng)
        m = measure_realized_latent_movement(
            model, fx.world, plan, d, fx.ref.features, rng, n_per_anchor=24)
        cos.append(m["direction_cosine"])
        disp.append(m.get("displacement_norm", np.nan))
        planned.append(float(d.delta))
    cos = np.asarray(cos, dtype=np.float64)
    disp = np.asarray(disp, dtype=np.float64)
    planned = np.asarray(planned, dtype=np.float64)
    overshoot = float(np.nanmean(disp) / max(np.nanmean(planned), 1e-9))
    return {
        "delta_scale": delta_scale,
        "n_directions": len(directions),
        "direction_cosine_mean": float(np.nanmean(cos)),
        "direction_cosine_median": float(np.nanmedian(cos)),
        "frac_positive": float(np.mean(cos > 0)),
        "planned_step_mean": float(np.nanmean(planned)),
        "realized_displacement_mean": float(np.nanmean(disp)),
        "overshoot": overshoot,
        "passed": bool(np.nanmean(cos) > 0.3),
        "per_direction_cosine": np.round(cos, 4).tolist(),
    }


def criterion5(
    fx, model, objective, mapper, directions, budget_spec,
    n_random: int, repeats: int, divergence_threshold: float,
) -> dict:
    """Predicted vs realized gain, measured as an expectation on both sides."""
    rng = fx.seeds.rng("acquisition")
    n_dir = len(directions)
    budget = budget_spec.budget

    cands = dict(_solver_allocations(objective, budget, n_dir))
    for i, a in enumerate(_random_allocations(n_dir, budget, n_random, rng)):
        cands[f"random_{i}"] = a
    # de-duplicate: one allocation contributes exactly one point, or identical
    # predictions get several ranks and the correlation is decided by noise
    uniq: dict[tuple, str] = {}
    for name, a in cands.items():
        uniq.setdefault(a, name)

    oracle = SyntheticAcquisitionOracle(
        fx.world, fx.policy, fx.vo, fx.va,
        OracleConfig(lr=fx.cfg["label_lr"], n_steps=fx.cfg["label_steps"],
                     n_repeats=1, seed=fx.seeds["acquisition"]),
    )

    rows = []
    for alloc, name in uniq.items():
        a = np.asarray(alloc, dtype=np.int64)
        pred = float(objective.value(a))
        draws = []
        for _ in range(repeats):
            # a fresh plan per draw: new anchors, new collection, new update. This
            # is the distribution `predicted` is an expectation over.
            plans = mapper.plan_allocation(directions, a, fx.store.metadata, rng=rng)
            if not plans:
                continue
            g = oracle.realized_allocation_gain(
                plans, fx.ref.flat_params, fx.ref.features, rng)["realized_gain"]
            draws.append(float(g))
        if not draws:
            continue
        draws = np.asarray(draws, dtype=np.float64)
        diverged = np.abs(draws) > divergence_threshold
        rows.append({
            "label": name,
            "allocation": [int(x) for x in a],
            "predicted": pred,
            "realized_mean": float(draws.mean()),
            "realized_median": float(np.median(draws)),
            "realized_std": float(draws.std()),
            "realized_draws": np.round(draws, 5).tolist(),
            "n_diverged": int(diverged.sum()),
            "realized_mean_clean": (
                float(draws[~diverged].mean()) if (~diverged).any() else float("nan")),
        })

    if len(rows) < 3:
        return {"n_candidates": len(rows), "spearman_mean": float("nan"),
                "passed": False, "reason": "too few candidates"}

    pred = np.array([r["predicted"] for r in rows])
    out = {
        "n_candidates": len(rows),
        "n_distinct_allocations": len(uniq),
        "repeats": repeats,
        "predicted_spread": float(pred.max() - pred.min()),
    }

    def rho(y):
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(pred) & np.isfinite(y)
        return float(spearmanr(pred[ok], y[ok]).statistic) if ok.sum() >= 3 else float("nan")

    # the headline: realized as an expectation over draws
    out["spearman_mean"] = rho([r["realized_mean"] for r in rows])
    # the median is robust to a single divergent update
    out["spearman_median"] = rho([r["realized_median"] for r in rows])
    # divergent draws excluded entirely
    out["spearman_clean"] = rho([r["realized_mean_clean"] for r in rows])
    # ONE draw per allocation: reproduces the original, broken measurement, so
    # the difference between this and `spearman_mean` is the cost of the bug
    out["spearman_single_draw"] = rho([r["realized_draws"][0] for r in rows])
    # how Spearman improves as the expectation is estimated better; if it is
    # flat, averaging is not the issue and the model really is uninformative
    for k in (1, 2, 3):
        if k <= repeats:
            out[f"spearman_at_{k}_repeats"] = rho(
                [float(np.mean(r["realized_draws"][:k])) for r in rows])

    within = np.array([r["realized_std"] for r in rows], dtype=np.float64)
    out["within_allocation_noise_mean"] = float(np.nanmean(within))
    out["noise_over_signal"] = float(
        np.nanmean(within) / max(out["predicted_spread"], 1e-9))
    out["n_allocations_with_divergence"] = int(sum(r["n_diverged"] > 0 for r in rows))
    out["frac_draws_diverged"] = float(
        sum(r["n_diverged"] for r in rows) / max(sum(len(r["realized_draws"]) for r in rows), 1))
    out["passed"] = bool(np.isfinite(out["spearman_mean"]) and out["spearman_mean"] > 0.3)
    out["records"] = rows
    return out


# ---- one (seed, delta_scale) cell ----------------------------------------


def run_cell(FixtureCls, cfg: dict, seed: int, delta_scale: float, args, say) -> dict:
    cfg = dict(cfg)
    cfg["delta_scale"] = delta_scale
    fx = FixtureCls(cfg, seed)
    model, final = fx.train()

    pctx = PolicyContextRef.from_checkpoint(fx.ref, fx.train_ds.checkpoint_ids)
    z_all = model.encode_store(fx.store, pctx.features, ckpt_index=pctx.ckpt_index)
    model.set_dataset_context(z_all)
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=cfg["n_clusters"], seed=fx.seeds["latent"])
    ).fit(z_all, fx.store.metadata)
    directions = DirectionGenerator(
        DirectionConfig(r_max=cfg["r_max"], delta_scale=delta_scale,
                        seed=fx.seeds["acquisition"])
    ).generate(clusters, z_all)
    if not directions:
        return {"seed": seed, "delta_scale": delta_scale, "error": "no directions"}

    mapper = MetadataMapper(
        fx.world.metadata_spec, MetadataMapperConfig(seed=fx.seeds["acquisition"]))
    mapper.fit(z_all, fx.store.metadata, clusters)
    directions, act = filter_actionable_directions(
        directions, mapper, fx.store.metadata, ActionabilityConfig())
    if not directions:
        return {"seed": seed, "delta_scale": delta_scale, "error": "no actionable directions"}

    sampler = LatentSampler(
        clusters, LatentSamplerConfig(sigma=0.3, seed=fx.seeds["acquisition"]))
    budget = BudgetSpec.from_directions(directions, budget=cfg["budget"])
    objective = AllocationObjective(
        model, sampler, directions, budget, policy_context=pctx,
        cfg=ObjectiveConfig(n_mc=cfg["n_mc"], seed=fx.seeds["acquisition"]))

    c6 = criterion6(fx, model, z_all, clusters, mapper, directions, delta_scale)
    c5 = criterion5(fx, model, objective, mapper, directions, budget,
                    args.n_random, args.repeats, args.divergence_threshold)

    say(f"  seed {seed} delta={delta_scale:.2f}: "
        f"C5 spearman(mean)={c5.get('spearman_mean', float('nan')):+.3f} "
        f"(single-draw {c5.get('spearman_single_draw', float('nan')):+.3f}, "
        f"noise/signal {c5.get('noise_over_signal', float('nan')):.1f}x) | "
        f"C6 cos={c6['direction_cosine_mean']:+.3f} overshoot={c6['overshoot']:.2f}x")

    return {
        "seed": seed,
        "delta_scale": delta_scale,
        "criterion5": c5,
        "criterion6": c6,
        "actionability": act,
        "n_directions": len(directions),
        "latent_norm_mean": final.get("latent_norm_mean", float("nan")),
        "effect_spearman": final.get("effect_spearman", float("nan")),
        "gain_within_r2": final.get("gain_within_r2", float("nan")),
    }


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--delta-scales", type=str, default="0.4",
                    help="sweep the latent step length; 0.4 is the E0 default")
    ap.add_argument("--repeats", type=int, default=6,
                    help="independent plan draws per allocation (makes realized "
                         "an expectation, as predicted already is)")
    ap.add_argument("--n-random", type=int, default=14,
                    help="random allocations added to defeat range restriction")
    ap.add_argument("--divergence-threshold", type=float, default=5.0,
                    help="|realized gain| above this is a diverged update")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="runs/diagnose_criteria")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--experiment", type=str, default="diag")
    args = ap.parse_args()

    mod = _load_fixture_module()
    cfg = dict(mod.DEFAULTS)
    if args.quick:
        cfg.update(mod.QUICK)
    # E0 labels at lr=0.3; the ablation defaults use 0.1. Match E0, since that
    # is the run being diagnosed.
    cfg["label_lr"] = 0.3

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    scales = [float(x) for x in args.delta_scales.split(",") if x.strip()]
    say = lambda m: print(f"[diag] {m}", flush=True)  # noqa: E731
    say(f"{len(seeds)} seeds x {len(scales)} delta_scales, "
        f"{args.repeats} repeats, +{args.n_random} random allocations")

    log = make_logger(
        enabled=args.wandb, project=args.wandb_project,
        name=f"{args.experiment}-criteria",
        group=f"{args.experiment}-criteria", job_type="diagnosis",
        config={"seeds": seeds, "delta_scales": scales, "repeats": args.repeats,
                "n_random": args.n_random, **{f"cfg/{k}": (list(v) if isinstance(v, tuple) else v)
                                              for k, v in cfg.items()}},
        tags=("synthetic", "diagnosis", args.experiment),
    )

    out = Path(args.out)
    cells = []

    def _checkpoint():
        """Persist after every cell, so a crash costs one cell, not the sweep."""
        save_json({
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "args": vars(args),
            "cells": cells,
            "analysis": analyse(cells),
            "partial": True,
            "provenance": run_provenance({"experiment": args.experiment}),
        }, out / "diagnose_criteria.json")

    for ds in scales:
        for s in seeds:
            try:
                cells.append(run_cell(mod.Fixture, cfg, s, ds, args, say))
            except Exception as e:  # one cell must not lose the sweep
                import traceback

                tb = traceback.format_exc()
                say(f"  seed {s} delta={ds}: FAILED {type(e).__name__}: {e}")
                say("    " + tb.strip().splitlines()[-3].strip())
                cells.append({"seed": s, "delta_scale": ds,
                              "error": f"{type(e).__name__}: {e}",
                              "traceback": tb})
            _checkpoint()

    report = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "args": vars(args),
        "cells": cells,
        "analysis": analyse(cells),
        "partial": False,
        "provenance": run_provenance({"experiment": args.experiment}),
    }
    save_json(report, out / "diagnose_criteria.json")
    _print(report, out)
    if log.active:
        ok = [c for c in cells if "error" not in c]
        log.table(
            "cells",
            ["seed", "delta_scale", "c5_spearman_mean", "c5_spearman_single_draw",
             "c5_noise_over_signal", "c5_frac_diverged", "c6_cosine", "c6_overshoot"],
            [[c["seed"], c["delta_scale"],
              c["criterion5"].get("spearman_mean", float("nan")),
              c["criterion5"].get("spearman_single_draw", float("nan")),
              c["criterion5"].get("noise_over_signal", float("nan")),
              c["criterion5"].get("frac_draws_diverged", float("nan")),
              c["criterion6"]["direction_cosine_mean"],
              c["criterion6"]["overshoot"]] for c in ok],
        )
        log.summary({f"analysis/{k}": v for k, v in report["analysis"].items()
                     if isinstance(v, (int, float, bool, str))})
        log.finish()
    return report


def analyse(cells: list[dict]) -> dict:
    """Pool the cells and ask which quantity actually predicts the failures."""
    ok = [c for c in cells if "error" not in c and "criterion5" in c]
    if not ok:
        return {"n_cells": 0}

    def arr(fn):
        return np.array([fn(c) for c in ok], dtype=np.float64)

    s_mean = arr(lambda c: c["criterion5"].get("spearman_mean", np.nan))
    s_one = arr(lambda c: c["criterion5"].get("spearman_single_draw", np.nan))
    noise = arr(lambda c: c["criterion5"].get("noise_over_signal", np.nan))
    div = arr(lambda c: c["criterion5"].get("frac_draws_diverged", np.nan))
    c6 = arr(lambda c: c["criterion6"]["direction_cosine_mean"])
    over = arr(lambda c: c["criterion6"]["overshoot"])

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(a[m], b[m]).statistic) if m.sum() >= 3 else float("nan")

    pos = s_mean > 0.3
    neg = s_mean < -0.1
    null = ~pos & ~neg
    return {
        "n_cells": len(ok),
        # does averaging over draws help at all? (hypothesis: the measurement)
        "spearman_mean_avg": float(np.nanmean(s_mean)),
        "spearman_single_draw_avg": float(np.nanmean(s_one)),
        "improvement_from_averaging": float(np.nanmean(s_mean - s_one)),
        "n_passed_c5": int(np.nansum(pos)),
        "n_negative_c5": int(np.nansum(neg)),
        "n_null_c5": int(np.nansum(null)),
        # which measured quantity explains the spread in criterion 5?
        "corr_c5_vs_noise_over_signal": corr(s_mean, noise),
        "corr_c5_vs_frac_diverged": corr(s_mean, div),
        "corr_c5_vs_overshoot": corr(s_mean, over),
        # and in criterion 6?
        "corr_c6_vs_overshoot": corr(c6, over),
        "corr_c6_vs_noise_over_signal": corr(c6, noise),
        "n_passed_c6": int(np.nansum(c6 > 0.3)),
        # conditions under which criterion 5 comes out positive vs negative
        "overshoot_when_c5_positive": float(np.nanmean(over[pos])) if pos.any() else float("nan"),
        "overshoot_when_c5_negative": float(np.nanmean(over[neg])) if neg.any() else float("nan"),
        "diverged_when_c5_positive": float(np.nanmean(div[pos])) if pos.any() else float("nan"),
        "diverged_when_c5_negative": float(np.nanmean(div[neg])) if neg.any() else float("nan"),
    }


def _print(report: dict, out: Path) -> None:
    cells = [c for c in report["cells"] if "error" not in c]
    a = report["analysis"]
    if not cells:
        print("\nno usable cells; errors were:")
        for c in report["cells"]:
            print(f"  seed {c.get('seed')} delta={c.get('delta_scale')}: {c.get('error')}")
        print(f"  report: {out / 'diagnose_criteria.json'}\n")
        return
    w = 118
    print("\n" + "=" * w)
    print("CRITERION 5 / 6 DIAGNOSIS")
    print("=" * w)
    print(f"  {'seed':>5s} {'delta':>6s} | {'C5 1-draw':>10s} {'C5 avg':>9s} "
          f"{'noise/sig':>10s} {'%diverged':>10s} | {'C6 cos':>8s} {'overshoot':>10s} "
          f"{'C5':>4s} {'C6':>4s}")
    print("-" * w)
    for c in sorted(cells, key=lambda x: (x["delta_scale"], x["seed"])):
        c5, c6 = c["criterion5"], c["criterion6"]
        print(f"  {c['seed']:>5d} {c['delta_scale']:>6.2f} | "
              f"{c5.get('spearman_single_draw', float('nan')):>+10.3f} "
              f"{c5.get('spearman_mean', float('nan')):>+9.3f} "
              f"{c5.get('noise_over_signal', float('nan')):>9.1f}x "
              f"{100 * c5.get('frac_draws_diverged', float('nan')):>9.1f}% | "
              f"{c6['direction_cosine_mean']:>+8.3f} {c6['overshoot']:>9.2f}x "
              f"{'PASS' if c5.get('passed') else 'FAIL':>4s} "
              f"{'PASS' if c6['passed'] else 'FAIL':>4s}")
    print("-" * w)
    print(f"  cells: {a['n_cells']}   criterion 5: {a['n_passed_c5']} positive, "
          f"{a['n_null_c5']} null, {a['n_negative_c5']} negative   "
          f"criterion 6: {a['n_passed_c6']} pass")
    print(f"  averaging over draws moved criterion 5 by "
          f"{a['improvement_from_averaging']:+.3f} "
          f"({a['spearman_single_draw_avg']:+.3f} -> {a['spearman_mean_avg']:+.3f})")
    print(f"  criterion 5 vs noise/signal : rho {a['corr_c5_vs_noise_over_signal']:+.3f}")
    print(f"  criterion 5 vs %diverged    : rho {a['corr_c5_vs_frac_diverged']:+.3f}")
    print(f"  criterion 5 vs overshoot    : rho {a['corr_c5_vs_overshoot']:+.3f}")
    print(f"  criterion 6 vs overshoot    : rho {a['corr_c6_vs_overshoot']:+.3f}")
    print(f"  report: {out / 'diagnose_criteria.json'}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
