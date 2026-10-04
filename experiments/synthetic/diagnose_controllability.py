"""Does a better-fitting effect model cost metadata controllability?

The 24-cell sweep (docs/E0_diagnosis.md) found, across 8 seeds:

    C6 (realized direction cosine) vs effect_spearman   rho = -0.833  p = 0.010

with the two extremes being seed 7, whose data model essentially failed to
learn (effect Spearman 0.085) yet scored C6 +0.494, and seed 4, which fits well
(0.789) and scores C6 -0.075 at every step length. That is only an
**observational** correlation over 8 points: `effect_spearman` is a property of
a seed's trained model, so anything else that varies by seed is a candidate
confound.

This script tests it causally, by varying the fit *within a fixed seed* and
leaving everything else - world, dataset, supervision records, clustering seed -
identical. Two arms:

**Arm 1, the fit sweep.** Train the same model for different numbers of epochs,
which moves `effect_spearman` without touching the data. If C6 declines as the
fit improves within a seed, the tension is real and the cross-seed correlation
was not a confound. If C6 is flat while the fit climbs, the correlation was
driven by something seed-specific and the lead is dead.

**Arm 2, the remedy.** `L_smooth` (PLAN.md 5.4) perturbs the batch composition
and the policy context and penalises the change in the readout, so it is the
loss term most directly about local regularity of the latent space - which is
exactly what a local linear Jacobian needs. At a fixed high-fit setting the
smoothness weight is swept. If controllability recovers without losing effect
accuracy, the tension is a weighting choice rather than a structural limit;
PLAN.md 5.4 already warns against over-weighting this term, so the arm reports
both sides of the trade.

Everything else is held fixed on purpose: the same `Fixture` (so the same
supervision records), the same clustering and direction seeds, the same step
length. Only the model changes.

    python experiments/synthetic/diagnose_controllability.py --seeds 0,4,7
    python experiments/synthetic/diagnose_controllability.py --seeds 0,4,7 \
        --epochs 5,15,30,60 --smooth-weights 0.0,0.01,0.1,0.5
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
from ldva.acquisition.metadata_mapper import (  # noqa: E402
    ActionabilityConfig,
    MetadataMapper,
    MetadataMapperConfig,
    filter_actionable_directions,
)
from ldva.analysis.wandb_logger import make_logger  # noqa: E402
from ldva.envs.synthetic.oracle import measure_realized_latent_movement  # noqa: E402
from ldva.policy.checkpoints import PolicyContextRef  # noqa: E402
from ldva.training.losses import LossWeights  # noqa: E402
from ldva.utils import run_provenance, save_json  # noqa: E402


def _load_fixture_module():
    path = ROOT / "experiments" / "synthetic" / "run_ablations.py"
    spec = importlib.util.spec_from_file_location("ldva_abl_fixture", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ldva_abl_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


def measure_controllability(fx, model, cfg: dict, delta_scale: float) -> dict:
    """Realized direction control for one trained model.

    Clustering, direction generation and the metadata map are all re-derived
    from this model's latents, because that is the real pipeline: a different
    representation gives different domains and different directions. The
    *seeds* for those steps are fixed, so the only thing varying is the model.
    """
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
        return {"error": "no directions", "n_directions": 0}

    mapper = MetadataMapper(
        fx.world.metadata_spec, MetadataMapperConfig(seed=fx.seeds["acquisition"]))
    mapper.fit(z_all, fx.store.metadata, clusters)
    directions, act = filter_actionable_directions(
        directions, mapper, fx.store.metadata, ActionabilityConfig())
    if not directions:
        return {"error": "no actionable directions", "n_directions": 0}

    rng = fx.seeds.rng("acquisition")
    cos, disp, planned, jr2, reach = [], [], [], [], []
    for d in directions:
        plan = mapper.plan_direction(d, 4, fx.store.metadata, rng=rng)
        m = measure_realized_latent_movement(
            model, fx.world, plan, d, fx.ref.features, rng, n_per_anchor=24)
        cos.append(m["direction_cosine"])
        disp.append(m.get("displacement_norm", np.nan))
        planned.append(float(d.delta))
        jr2.append(m.get("jacobian_r2_heldout", np.nan))
        reach.append(m.get("reachability_cosine", np.nan))
    cos = np.asarray(cos, dtype=np.float64)
    return {
        "n_directions": len(directions),
        "direction_cosine_mean": float(np.nanmean(cos)),
        "direction_cosine_median": float(np.nanmedian(cos)),
        "frac_positive": float(np.mean(cos > 0)),
        "planned_step_mean": float(np.nanmean(planned)),
        "realized_displacement_mean": float(np.nanmean(disp)),
        "overshoot": float(np.nanmean(disp) / max(np.nanmean(planned), 1e-9)),
        "jacobian_r2_heldout_mean": float(np.nanmean(jr2)),
        "reachability_cosine_mean": float(np.nanmean(reach)),
        "actionable_survival": act["survival_rate"],
        "passed": bool(np.nanmean(cos) > 0.3),
    }


def run_arm(fx, cfg: dict, epochs: int, smooth: float, delta_scale: float,
            say, replicates: int = 1) -> dict:
    """Train one model variant on a fixed fixture and measure both quantities.

    `replicates` retrains the same configuration with a different *training*
    seed each time and reports the spread. Without it the experiment cannot
    tell an effect from run-to-run variation: an accidental duplicate of the
    (epochs=60, smooth=0.01) cell came out at C6 +0.471 in one arm and +0.622
    in another, and the effects under study are themselves only ~0.3. The noise
    floor therefore has to be measured, not assumed.
    """
    weights = LossWeights(1.0, 1.0, 0.1, smooth)
    local = dict(cfg)
    local["epochs"] = epochs
    fits, c6s, reps = [], [], []
    for k in range(max(replicates, 1)):
        fx.cfg = local  # Fixture.train reads epochs from cfg
        # the offset must go THROUGH Fixture.train: it re-seeds from
        # `seeds["latent"]` itself, so seeding here would simply be overwritten
        # and all replicates would come out byte-identical (they did, std=0.000)
        model, final = fx.train(weights=weights, seed_offset=7919 * k)
        ctrl = measure_controllability(fx, model, local, delta_scale)
        fits.append(final.get("effect_spearman", float("nan")))
        c6s.append(ctrl.get("direction_cosine_mean", float("nan")))
        reps.append({"replicate": k, "effect_spearman": fits[-1],
                     "gain_within_r2": final.get("gain_within_r2", float("nan")),
                     "control": ctrl})
    out = {
        "epochs": epochs,
        "smooth_weight": smooth,
        "replicates": len(reps),
        "effect_spearman": float(np.nanmean(fits)),
        "effect_spearman_std": float(np.nanstd(fits)),
        "gain_within_r2": float(np.nanmean(
            [r["gain_within_r2"] for r in reps])),
        "c6_mean": float(np.nanmean(c6s)),
        "c6_std": float(np.nanstd(c6s)),
        "per_replicate": reps,
        "control": reps[-1]["control"],
    }
    say(f"    epochs={epochs:<3d} smooth={smooth:<5.3g} -> "
        f"effect_spearman={out['effect_spearman']:+.3f}"
        f"+/-{out['effect_spearman_std']:.3f} "
        f"gain_r2={out['gain_within_r2']:+.3f} | "
        f"C6={out['c6_mean']:+.3f}+/-{out['c6_std']:.3f} "
        f"(n={out['replicates']})")
    return out


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=str, default="0,4,7",
                    help="0 is middling, 4 fits well with bad control, 7's model "
                         "failed yet controls well - the two extremes plus a middle")
    ap.add_argument("--epochs", type=str, default="5,15,30,60",
                    help="arm 1: fit quality, varied within a seed")
    ap.add_argument("--smooth-weights", type=str, default="0.0,0.01,0.1,0.5",
                    help="arm 2: L_smooth weight at the highest epoch count")
    ap.add_argument("--delta-scale", type=float, default=0.4)
    ap.add_argument("--replicates", type=int, default=3,
                    help="retrain each arm this many times with different training "
                         "seeds, so the noise floor is measured")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="runs/diagnose_controllability")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--experiment", type=str, default="ctrl")
    args = ap.parse_args()

    mod = _load_fixture_module()
    cfg = dict(mod.DEFAULTS)
    if args.quick:
        cfg.update(mod.QUICK)
    cfg["label_lr"] = 0.3  # match E0, which is what is being explained

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    smooths = [float(x) for x in args.smooth_weights.split(",") if x.strip()]
    base_smooth = 0.01  # the value E0 and the acquisition loop use

    say = lambda m: print(f"[ctrl] {m}", flush=True)  # noqa: E731
    say(f"{len(seeds)} seeds | arm 1: epochs {epochs} | arm 2: smooth {smooths} "
        f"at epochs={max(epochs)}")

    log = make_logger(
        enabled=args.wandb, project=args.wandb_project,
        name=f"{args.experiment}-controllability",
        group=f"{args.experiment}-controllability", job_type="diagnosis",
        config={"seeds": seeds, "epochs": epochs, "smooth_weights": smooths,
                "delta_scale": args.delta_scale},
        tags=("synthetic", "controllability", args.experiment),
    )

    out_dir = Path(args.out)
    results = {"fit_sweep": [], "smooth_sweep": []}

    def _save():
        save_json({
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "args": vars(args),
            "results": results,
            "analysis": analyse(results),
            "contrast": contrast_analysis(results),
            "provenance": run_provenance({"experiment": args.experiment}),
        }, out_dir / "diagnose_controllability.json")

    for s in seeds:
        say(f"  seed {s} - arm 1: fit sweep (everything else fixed)")
        fx = mod.Fixture(cfg, s)
        for e in epochs:
            try:
                r = run_arm(fx, cfg, e, base_smooth, args.delta_scale, say,
                            args.replicates)
                r["seed"] = s
                results["fit_sweep"].append(r)
            except Exception as exc:  # one arm must not lose the sweep
                say(f"    epochs={e}: FAILED {type(exc).__name__}: {exc}")
            _save()

        say(f"  seed {s} - arm 2: smoothness sweep at epochs={max(epochs)}")
        for w in smooths:
            try:
                r = run_arm(fx, cfg, max(epochs), w, args.delta_scale, say,
                            args.replicates)
                r["seed"] = s
                results["smooth_sweep"].append(r)
            except Exception as exc:
                say(f"    smooth={w}: FAILED {type(exc).__name__}: {exc}")
            _save()

    report_analysis = analyse(results)
    _save()
    _print(results, report_analysis, out_dir)
    if log.active:
        for arm in ("fit_sweep", "smooth_sweep"):
            rows = [[r["seed"], r["epochs"], r["smooth_weight"],
                     r["effect_spearman"], r["gain_within_r2"],
                     r["control"].get("direction_cosine_mean", float("nan")),
                     r["control"].get("overshoot", float("nan")),
                     r["control"].get("n_directions", 0)] for r in results[arm]]
            log.table(arm, ["seed", "epochs", "smooth", "effect_spearman",
                            "gain_r2", "c6_cosine", "overshoot", "n_directions"], rows)
        log.summary({f"analysis/{k}": v for k, v in report_analysis.items()
                     if isinstance(v, (int, float, bool, str))})
        log.finish()
    return results


def analyse(results: dict) -> dict:
    """Within-seed correlations, which is what makes the test causal.

    Pooling across seeds would reintroduce exactly the confound this experiment
    exists to remove, so each seed's sweep is correlated on its own and the
    per-seed values are reported rather than averaged into one number.
    """
    out: dict = {}
    for arm in ("fit_sweep", "smooth_sweep"):
        rows = results.get(arm, [])
        per_seed = {}
        for s in sorted({r["seed"] for r in rows}):
            g = [r for r in rows if r["seed"] == s]
            if len(g) < 3:
                continue
            fit = np.array([r["effect_spearman"] for r in g], dtype=np.float64)
            c6 = np.array([r.get("c6_mean", r["control"].get(
                "direction_cosine_mean", np.nan)) for r in g], dtype=np.float64)
            knob = np.array(
                [r["epochs"] if arm == "fit_sweep" else r["smooth_weight"] for r in g],
                dtype=np.float64)
            m = np.isfinite(fit) & np.isfinite(c6)
            per_seed[str(s)] = {
                "n": int(m.sum()),
                "rho_fit_vs_c6": (
                    float(spearmanr(fit[m], c6[m]).statistic) if m.sum() >= 3
                    else float("nan")),
                "rho_knob_vs_fit": (
                    float(spearmanr(knob[m], fit[m]).statistic) if m.sum() >= 3
                    else float("nan")),
                "rho_knob_vs_c6": (
                    float(spearmanr(knob[m], c6[m]).statistic) if m.sum() >= 3
                    else float("nan")),
                "fit_range": [float(np.nanmin(fit)), float(np.nanmax(fit))],
                "c6_range": [float(np.nanmin(c6)), float(np.nanmax(c6))],
            }
        rhos = [v["rho_fit_vs_c6"] for v in per_seed.values()
                if np.isfinite(v["rho_fit_vs_c6"])]
        out[arm] = {
            "per_seed": per_seed,
            "n_seeds": len(per_seed),
            "rho_fit_vs_c6_mean": float(np.mean(rhos)) if rhos else float("nan"),
            "n_seeds_negative": int(sum(1 for r in rhos if r < 0)),
            "n_seeds_positive": int(sum(1 for r in rhos if r > 0)),
        }
    a1 = out.get("fit_sweep", {})
    out["verdict"] = (
        "tension confirmed within seeds"
        if a1.get("n_seeds", 0) >= 2 and a1.get("n_seeds_negative", 0) == a1.get("n_seeds")
        else "not confirmed within seeds - the cross-seed correlation may be confounded"
        if a1.get("n_seeds", 0) >= 2
        else "insufficient data"
    )
    return out


def contrast_analysis(results: dict) -> dict:
    """Per-seed fit->C6 contrast, then group seeds by its sign.

    With two fit levels a within-seed rank correlation is meaningless, so each
    seed contributes one signed difference

        dC6 = C6(high fit) - C6(low fit)

    with a standard error from its replicates. Seeds are then split by sign and
    every seed-level property compared between the groups - which is the
    question "under what conditions does the trade-off happen" in a form the
    data can answer. A seed counts for one side only when |dC6| exceeds twice
    its own standard error; otherwise it is `unresolved` rather than being
    assigned a direction it cannot support.
    """
    rows = results.get("fit_sweep", [])
    by_seed: dict[int, list] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)

    per_seed = []
    for sd, g in sorted(by_seed.items()):
        g = sorted(g, key=lambda r: r["epochs"])
        if len(g) < 2:
            continue
        lo, hi = g[0], g[-1]
        d = hi.get("c6_mean", np.nan) - lo.get("c6_mean", np.nan)
        n_lo = max(lo.get("replicates", 1), 1)
        n_hi = max(hi.get("replicates", 1), 1)
        se = float(np.sqrt(lo.get("c6_std", 0.0) ** 2 / n_lo
                           + hi.get("c6_std", 0.0) ** 2 / n_hi))
        resolved = bool(np.isfinite(d) and se > 0 and abs(d) > 2 * se)
        per_seed.append({
            "seed": sd,
            "fit_low": lo.get("effect_spearman", np.nan),
            "fit_high": hi.get("effect_spearman", np.nan),
            "fit_gain": hi.get("effect_spearman", np.nan) - lo.get("effect_spearman", np.nan),
            "c6_low": lo.get("c6_mean", np.nan),
            "c6_high": hi.get("c6_mean", np.nan),
            "delta_c6": float(d),
            "se": se,
            "resolved": resolved,
            "sign": ("tradeoff" if resolved and d < 0
                     else "positive" if resolved and d > 0
                     else "unresolved"),
            "gain_r2_high": hi.get("gain_within_r2", np.nan),
            "overshoot_high": hi.get("control", {}).get("overshoot", np.nan),
            "n_directions_high": hi.get("control", {}).get("n_directions", np.nan),
        })

    groups = {k: [r for r in per_seed if r["sign"] == k]
              for k in ("tradeoff", "positive", "unresolved")}
    feats = ("fit_low", "fit_high", "fit_gain", "c6_low", "c6_high",
             "gain_r2_high", "overshoot_high", "n_directions_high")
    conditions = []
    for f in feats:
        row = {"feature": f}
        for k, g in groups.items():
            v = np.array([r[f] for r in g], dtype=np.float64)
            v = v[np.isfinite(v)]
            row[f"mean_{k}"] = float(v.mean()) if v.size else float("nan")
            row[f"n_{k}"] = int(v.size)
        conditions.append(row)

    deltas = np.array([r["delta_c6"] for r in per_seed], dtype=np.float64)
    return {
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "counts": {k: len(v) for k, v in groups.items()},
        "delta_c6_mean": float(np.nanmean(deltas)) if deltas.size else float("nan"),
        "delta_c6_sem": (float(np.nanstd(deltas) / np.sqrt(len(deltas)))
                         if deltas.size else float("nan")),
        "n_delta_negative": int(np.nansum(deltas < 0)),
        "n_delta_positive": int(np.nansum(deltas > 0)),
        "conditions": conditions,
        "verdict": _contrast_verdict(groups),
    }


def _contrast_verdict(groups: dict) -> str:
    n_t, n_p = len(groups["tradeoff"]), len(groups["positive"])
    if n_t == 0 and n_p == 0:
        return ("no seed resolves either way - the trade-off is smaller than "
                "run-to-run noise at this replicate count")
    if n_t > 0 and n_p == 0:
        return f"trade-off resolved in {n_t} seed(s), reversed in none"
    if n_p > 0 and n_t == 0:
        return f"POSITIVE in {n_p} seed(s), trade-off in none"
    return (f"both directions occur: trade-off in {n_t}, positive in {n_p} - "
            "see the conditions table for what separates them")


def _print(results: dict, a: dict, out_dir: Path) -> None:
    w = 104
    print("\n" + "=" * w)
    print("FIT vs METADATA CONTROLLABILITY")
    print("=" * w)
    for arm, title in (("fit_sweep", "ARM 1 - epochs sweep (fit varies, data fixed)"),
                       ("smooth_sweep", "ARM 2 - L_smooth sweep at max epochs")):
        rows = results.get(arm, [])
        if not rows:
            continue
        print(f"\n  {title}")
        knob = "epochs" if arm == "fit_sweep" else "smooth"
        print(f"  {'seed':>5s} {knob:>8s} {'eff.sprm':>10s} {'gain_r2':>9s} "
              f"{'C6 mean':>9s} {'C6 std':>8s} {'n':>3s}")
        for r in rows:
            k = r["epochs"] if arm == "fit_sweep" else r["smooth_weight"]
            print(f"  {r['seed']:>5d} {k:>8.3g} {r['effect_spearman']:>+10.3f} "
                  f"{r['gain_within_r2']:>+9.3f} "
                  f"{r.get('c6_mean', float('nan')):>+9.3f} "
                  f"{r.get('c6_std', float('nan')):>8.3f} "
                  f"{r.get('replicates', 1):>3d}")
        s = a.get(arm, {})
        print(f"    within-seed rho(fit, C6): mean {s.get('rho_fit_vs_c6_mean', float('nan')):+.3f}"
              f"   negative in {s.get('n_seeds_negative', 0)}/{s.get('n_seeds', 0)} seeds")
        for sd, v in s.get("per_seed", {}).items():
            print(f"      seed {sd}: rho(fit,C6)={v['rho_fit_vs_c6']:+.3f} "
                  f"fit {v['fit_range'][0]:+.3f}..{v['fit_range'][1]:+.3f} "
                  f"C6 {v['c6_range'][0]:+.3f}..{v['c6_range'][1]:+.3f}")
    ca = contrast_analysis(results)
    if ca.get("n_seeds"):
        print(f"\n  PER-SEED CONTRAST  C6(high fit) - C6(low fit)   ({ca['n_seeds']} seeds)")
        print(f"  {'seed':>5s} {'fit lo':>8s} {'fit hi':>8s} {'C6 lo':>8s} {'C6 hi':>8s} "
              f"{'dC6':>8s} {'SE':>7s} {'verdict':>11s}")
        for r in ca["per_seed"]:
            print(f"  {r['seed']:>5d} {r['fit_low']:>+8.3f} {r['fit_high']:>+8.3f} "
                  f"{r['c6_low']:>+8.3f} {r['c6_high']:>+8.3f} {r['delta_c6']:>+8.3f} "
                  f"{r['se']:>7.3f} {r['sign']:>11s}")
        c = ca["counts"]
        print(f"    resolved: {c['tradeoff']} trade-off, {c['positive']} positive, "
              f"{c['unresolved']} unresolved")
        print(f"    mean dC6 {ca['delta_c6_mean']:+.3f} +/- {ca['delta_c6_sem']:.3f} "
              f"(SEM over seeds)   sign split "
              f"{ca['n_delta_negative']}neg/{ca['n_delta_positive']}pos")
        print("\n  CONDITIONS (mean per group)")
        print(f"  {'feature':<20s} {'tradeoff':>10s} {'positive':>10s} {'unresolved':>12s}")
        for row in ca["conditions"]:
            print(f"  {row['feature']:<20s} {row['mean_tradeoff']:>+10.3f} "
                  f"{row['mean_positive']:>+10.3f} {row['mean_unresolved']:>+12.3f}")
        print(f"\n  CONTRAST VERDICT: {ca['verdict']}")
    print("\n" + "-" * w)
    print(f"  VERDICT: {a.get('verdict')}")
    print(f"  report: {out_dir / 'diagnose_controllability.json'}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
