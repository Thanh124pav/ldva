"""Why does the latent space collapse to ~1 effective dimension, and can it be
widened without losing effect accuracy?

Measured over 8 seeds, the encoder uses a participation ratio of 1.0-2.2 out of
32 latent dimensions. That is upstream of both failing criteria: with every
candidate direction pointing into the same narrow subspace, a realized
displacement aligns with any direction (criterion 6's raw cosine reads +0.91 at
only 1.3 sd above requesting a different direction), and the geometry the
planner reasons over is nearly one-dimensional. PLAN.md 19/F1 names this as a
failure mode: "no stable effect geometry".

**The leading hypothesis is that the collapse is rational.** The readout maps
`(z_i, h_{B\\i}, theta)` to a *scalar* effect, so one latent dimension is
sufficient to minimise the effect loss. Nothing in `L_effect` or `L_batch`
rewards keeping more. The term that should counteract it is `L_metric`
(PLAN.md 5.3), which pulls latent distance toward effect-*profile* distance -
and a profile is a vector over contexts, so matching it needs more than one
dimension. If that is right, the collapse is a weighting problem and raising
the metric weight should widen the space.

Three arms, each measuring effective dimensionality, direction specificity and
effect accuracy together, so a fix that widens the space while destroying
accuracy is visible as such:

1. **metric weight** 0 / 0.1 (default) / 1.0 / 5.0 — the hypothesised control.
2. **latent dimension** 8 / 32 — is the collapse absolute (always ~1 dimension)
   or proportional to the budget?
3. **contexts per sample** — a profile over more contexts is a richer target,
   so if the metric loss is the mechanism, more contexts should also help.

Direction specificity is the z-score of `cos(delta_i, v_i)` against the other
candidate directions, which is the only interpretable form when the space may
be collapsed; see `ldva.analysis.direction_validation.direction_specificity`.

    python experiments/synthetic/diagnose_latent_collapse.py --seeds 0,2,4
    python experiments/synthetic/diagnose_latent_collapse.py --seeds 0,1,2,3 \\
        --metric-weights 0,0.1,1.0,5.0 --latent-dims 8,32
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
from ldva.analysis.direction_validation import (  # noqa: E402
    direction_specificity,
    participation_ratio,
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


def measure(fx, model, cfg: dict, delta_scale: float, n_per_anchor: int) -> dict:
    """Effective dimensionality, direction specificity, and the direction count."""
    pctx = PolicyContextRef.from_checkpoint(fx.ref, fx.train_ds.checkpoint_ids)
    z_all = model.encode_store(fx.store, pctx.features, ckpt_index=pctx.ckpt_index)
    model.set_dataset_context(z_all)
    out = {
        "latent_dim": int(z_all.shape[1]),
        "participation_ratio": participation_ratio(z_all),
        "latent_norm_mean": float(np.linalg.norm(z_all, axis=1).mean()),
    }

    clusters = LatentClustering(
        ClusteringConfig(n_clusters=cfg["n_clusters"], seed=fx.seeds["latent"])
    ).fit(z_all, fx.store.metadata)
    directions = DirectionGenerator(
        DirectionConfig(r_max=cfg["r_max"], delta_scale=delta_scale,
                        seed=fx.seeds["acquisition"])
    ).generate(clusters, z_all)
    if not directions:
        out["error"] = "no directions"
        return out
    mapper = MetadataMapper(
        fx.world.metadata_spec, MetadataMapperConfig(seed=fx.seeds["acquisition"]))
    mapper.fit(z_all, fx.store.metadata, clusters)
    directions, _ = filter_actionable_directions(
        directions, mapper, fx.store.metadata, ActionabilityConfig())
    if not directions:
        out["error"] = "no actionable directions"
        return out

    rng = fx.seeds.rng("acquisition")
    deltas, vectors, cos = {}, {}, []
    for d in directions:
        plan = mapper.plan_direction(d, 4, fx.store.metadata, rng=rng)
        m = measure_realized_latent_movement(
            model, fx.world, plan, d, fx.ref.features, rng, n_per_anchor=n_per_anchor)
        cos.append(m["direction_cosine"])
        if "realized_delta" in m:
            deltas[d.direction_id] = np.asarray(m["realized_delta"], dtype=np.float64)
            vectors[d.direction_id] = np.asarray(d.vector, dtype=np.float64)
    spec = direction_specificity(deltas, vectors)
    out.update({
        "n_directions": len(directions),
        "cos_desired_mean": float(np.nanmean(cos)) if cos else float("nan"),
        "specificity_z": spec.get("z_score_mean", float("nan")),
        "null_abs_mean": spec.get("null_abs_mean", float("nan")),
        "frac_above_2sd": spec.get("frac_directions_above_2sd", float("nan")),
    })
    return out


def run_arm(fx, cfg: dict, label: str, say, replicates: int, args,
            metric_w: float, latent_dim: int | None, records=None) -> dict:
    pr, z, cosv, fits, nd = [], [], [], [], []
    for k in range(max(replicates, 1)):
        fx.cfg = cfg
        model, final = fx.train(
            latent_dim=latent_dim,
            weights=LossWeights(1.0, 1.0, metric_w, 0.01),
            records=records,
            seed_offset=7919 * k,
        )
        m = measure(fx, model, cfg, args.delta_scale, args.n_per_anchor)
        if "error" in m and "participation_ratio" not in m:
            continue
        pr.append(m.get("participation_ratio", np.nan))
        z.append(m.get("specificity_z", np.nan))
        cosv.append(m.get("cos_desired_mean", np.nan))
        nd.append(m.get("n_directions", np.nan))
        fits.append(final.get("effect_spearman", np.nan))
    if not pr:
        return {"label": label, "error": "no usable replicate"}
    out = {
        "label": label,
        "metric_weight": metric_w,
        "latent_dim_requested": latent_dim if latent_dim else cfg["latent_dim"],
        "replicates": len(pr),
        "participation_ratio": float(np.nanmean(pr)),
        "participation_ratio_std": float(np.nanstd(pr)),
        "specificity_z": float(np.nanmean(z)),
        "specificity_z_std": float(np.nanstd(z)),
        "cos_desired": float(np.nanmean(cosv)),
        "effect_spearman": float(np.nanmean(fits)),
        "n_directions": float(np.nanmean(nd)),
    }
    say(f"    {label:<22s} eff_dim={out['participation_ratio']:>5.2f}"
        f"+/-{out['participation_ratio_std']:.2f} "
        f"z={out['specificity_z']:+.2f}+/-{out['specificity_z_std']:.2f} "
        f"cos={out['cos_desired']:+.3f} fit={out['effect_spearman']:+.3f} "
        f"n_dir={out['n_directions']:.0f}")
    return out


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=str, default="0,1,2,3")
    ap.add_argument("--metric-weights", type=str, default="0,0.1,1.0,5.0")
    ap.add_argument("--latent-dims", type=str, default="8,32")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--delta-scale", type=float, default=0.4)
    ap.add_argument("--n-per-anchor", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="runs/diagnose_latent_collapse")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--experiment", type=str, default="collapse")
    args = ap.parse_args()

    mod = _load_fixture_module()
    cfg = dict(mod.DEFAULTS)
    if args.quick:
        cfg.update(mod.QUICK)
    cfg["label_lr"] = 0.3

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    mws = [float(x) for x in args.metric_weights.split(",") if x.strip()]
    lds = [int(x) for x in args.latent_dims.split(",") if x.strip()]
    say = lambda m: print(f"[collapse] {m}", flush=True)  # noqa: E731
    say(f"{len(seeds)} seeds | metric weights {mws} | latent dims {lds} | "
        f"{args.replicates} replicates")

    log = make_logger(
        enabled=args.wandb, project=args.wandb_project,
        name=f"{args.experiment}-latent-collapse",
        group=f"{args.experiment}-latent-collapse", job_type="diagnosis",
        config={"seeds": seeds, "metric_weights": mws, "latent_dims": lds,
                "replicates": args.replicates},
        tags=("synthetic", "collapse", args.experiment),
    )

    out_dir = Path(args.out)
    cells: list[dict] = []

    def _save():
        save_json({
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items()},
            "args": vars(args),
            "cells": cells,
            "analysis": analyse(cells),
            "provenance": run_provenance({"experiment": args.experiment}),
        }, out_dir / "diagnose_latent_collapse.json")

    base_ld = cfg["latent_dim"]
    for s in seeds:
        say(f"  seed {s}")
        try:
            fx = mod.Fixture(cfg, s)
        except Exception as e:
            say(f"    fixture FAILED {type(e).__name__}: {e}")
            continue

        for w in mws:
            try:
                r = run_arm(fx, cfg, f"metric_w={w:g}", say, args.replicates,
                            args, w, None)
                r.update({"seed": s, "arm": "metric_weight"})
                cells.append(r)
            except Exception as exc:
                say(f"    metric_w={w}: FAILED {type(exc).__name__}: {exc}")
            _save()

        for ld in lds:
            if ld == base_ld:
                continue
            try:
                r = run_arm(fx, cfg, f"latent_dim={ld}", say, args.replicates,
                            args, 0.1, ld)
                r.update({"seed": s, "arm": "latent_dim"})
                cells.append(r)
            except Exception as exc:
                say(f"    latent_dim={ld}: FAILED {type(exc).__name__}: {exc}")
            _save()

    a = analyse(cells)
    _save()
    _print(cells, a, out_dir)
    if log.active:
        log.table("cells",
                  ["seed", "arm", "label", "metric_weight", "latent_dim_requested",
                   "participation_ratio", "specificity_z", "cos_desired",
                   "effect_spearman", "n_directions"],
                  [[c.get("seed"), c.get("arm"), c.get("label"),
                    c.get("metric_weight"), c.get("latent_dim_requested"),
                    c.get("participation_ratio"), c.get("specificity_z"),
                    c.get("cos_desired"), c.get("effect_spearman"),
                    c.get("n_directions")] for c in cells if "error" not in c])
        log.summary({f"analysis/{k}": v for k, v in a.items()
                     if isinstance(v, (int, float, bool, str))})
        log.finish()
    return a


def analyse(cells: list[dict]) -> dict:
    ok = [c for c in cells if "error" not in c and "participation_ratio" in c]
    if len(ok) < 3:
        return {"n_cells": len(ok)}

    def grp(arm):
        return [c for c in ok if c.get("arm") == arm]

    out: dict = {"n_cells": len(ok)}

    mw = grp("metric_weight")
    if len(mw) >= 3:
        w = np.array([c["metric_weight"] for c in mw], dtype=np.float64)
        pr = np.array([c["participation_ratio"] for c in mw], dtype=np.float64)
        z = np.array([c["specificity_z"] for c in mw], dtype=np.float64)
        fit = np.array([c["effect_spearman"] for c in mw], dtype=np.float64)
        m = np.isfinite(w) & np.isfinite(pr)
        out["metric_weight"] = {
            "rho_weight_vs_eff_dim": (
                float(spearmanr(w[m], pr[m]).statistic) if m.sum() >= 3 else float("nan")),
            "rho_weight_vs_specificity": (
                float(spearmanr(w, z).statistic) if np.isfinite(z).sum() >= 3
                else float("nan")),
            "rho_weight_vs_fit": (
                float(spearmanr(w, fit).statistic) if np.isfinite(fit).sum() >= 3
                else float("nan")),
            "rho_eff_dim_vs_specificity": (
                float(spearmanr(pr, z).statistic) if np.isfinite(z).sum() >= 3
                else float("nan")),
            "by_weight": {
                f"{wv:g}": {
                    "eff_dim": float(np.nanmean([c["participation_ratio"]
                                                 for c in mw if c["metric_weight"] == wv])),
                    "specificity_z": float(np.nanmean([c["specificity_z"]
                                                       for c in mw if c["metric_weight"] == wv])),
                    "effect_spearman": float(np.nanmean([c["effect_spearman"]
                                                         for c in mw if c["metric_weight"] == wv])),
                } for wv in sorted(set(w.tolist()))
            },
        }

    ld = grp("latent_dim") + [c for c in mw if c.get("metric_weight") == 0.1]
    if ld:
        out["latent_dim"] = {
            f"{d:g}": {
                "eff_dim": float(np.nanmean([c["participation_ratio"]
                                             for c in ld
                                             if c["latent_dim_requested"] == d])),
                "specificity_z": float(np.nanmean([c["specificity_z"] for c in ld
                                                   if c["latent_dim_requested"] == d])),
            } for d in sorted({c["latent_dim_requested"] for c in ld})
        }

    pr_all = np.array([c["participation_ratio"] for c in ok], dtype=np.float64)
    z_all = np.array([c["specificity_z"] for c in ok], dtype=np.float64)
    m = np.isfinite(pr_all) & np.isfinite(z_all)
    out["rho_eff_dim_vs_specificity_all"] = (
        float(spearmanr(pr_all[m], z_all[m]).statistic) if m.sum() >= 3 else float("nan"))
    out["max_eff_dim"] = float(np.nanmax(pr_all))
    out["max_specificity_z"] = float(np.nanmax(z_all))
    out["any_cell_above_2sd"] = bool(np.nanmax(z_all) >= 2.0)
    mwa = out.get("metric_weight", {})
    out["verdict"] = (
        "metric weight widens the space AND improves specificity"
        if mwa.get("rho_weight_vs_eff_dim", 0) > 0.5
        and mwa.get("rho_weight_vs_specificity", 0) > 0.3 else
        "metric weight widens the space but specificity does not follow"
        if mwa.get("rho_weight_vs_eff_dim", 0) > 0.5 else
        "metric weight does not widen the space - the collapse is not a "
        "metric-loss weighting problem"
        if "metric_weight" in out else "insufficient data"
    )
    return out


def _print(cells: list[dict], a: dict, out_dir: Path) -> None:
    ok = [c for c in cells if "error" not in c and "participation_ratio" in c]
    w = 110
    print("\n" + "=" * w)
    print("LATENT COLLAPSE: what widens the space, and does specificity follow?")
    print("=" * w)
    print(f"  {'seed':>5s} {'arm':>14s} {'label':>18s} {'eff_dim':>9s} "
          f"{'spec z':>9s} {'cos':>8s} {'fit':>8s} {'n_dir':>6s}")
    for c in ok:
        print(f"  {c.get('seed', -1):>5d} {c.get('arm', ''):>14s} {c['label']:>18s} "
              f"{c['participation_ratio']:>9.2f} {c['specificity_z']:>+9.2f} "
              f"{c['cos_desired']:>+8.3f} {c['effect_spearman']:>+8.3f} "
              f"{c['n_directions']:>6.0f}")
    mwa = a.get("metric_weight")
    if mwa:
        print("-" * w)
        print("  metric weight sweep (mean over seeds)")
        print(f"  {'weight':>8s} {'eff_dim':>9s} {'spec z':>9s} {'fit':>9s}")
        for k, v in mwa["by_weight"].items():
            print(f"  {k:>8s} {v['eff_dim']:>9.2f} {v['specificity_z']:>+9.2f} "
                  f"{v['effect_spearman']:>+9.3f}")
        print(f"    rho(weight, eff_dim)      {mwa['rho_weight_vs_eff_dim']:+.3f}")
        print(f"    rho(weight, specificity)  {mwa['rho_weight_vs_specificity']:+.3f}")
        print(f"    rho(weight, effect fit)   {mwa['rho_weight_vs_fit']:+.3f}")
        print(f"    rho(eff_dim, specificity) {mwa['rho_eff_dim_vs_specificity']:+.3f}")
    if a.get("latent_dim"):
        print("-" * w)
        print("  latent dimension (is the collapse absolute or proportional?)")
        for k, v in a["latent_dim"].items():
            print(f"    latent_dim={k:>4s} -> eff_dim {v['eff_dim']:.2f}, "
                  f"spec z {v['specificity_z']:+.2f}")
    print("-" * w)
    print(f"  best eff_dim seen {a.get('max_eff_dim', float('nan')):.2f}, "
          f"best specificity z {a.get('max_specificity_z', float('nan')):+.2f}, "
          f"any cell >= 2sd: {a.get('any_cell_above_2sd')}")
    print(f"\n  VERDICT: {a.get('verdict')}")
    print(f"  report: {out_dir / 'diagnose_latent_collapse.json'}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
