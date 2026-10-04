"""Is criterion 6 measuring lost control, or just a bigger latent space?

Criterion 6 is the realized direction cosine: collect at a plan's metadata,
encode, and take `cos(z_new - z_anchor, v_desired)`. It fails on 2 of 5 seeds
and is the blocker for E1.

Before trying to fix the method, this checks the measurement - because that is
exactly where criterion 5's failure turned out to live, and criterion 6 shows
the same warning signs. Two specific suspicions:

**1. A raw cosine falls as dimensionality rises, for free.** Between a fixed
vector and a random one in d dimensions, E|cos| is about sqrt(2/(pi*d)): 0.45 at
d=4, 0.25 at d=12, 0.11 at d=64. So if training makes the encoder *use* more
latent directions, criterion 6 declines without any control being lost. That
would explain the central puzzle in docs/E0_criteria_resolution.md - that C6
converges toward 0.3-0.4 regardless of where a seed started - as an artefact of
the denominator rather than drift.

**2. The signal may be below the measurement noise.** The README already
records that one chunk's latent noise (0.50) exceeds a planned displacement
(0.22), which is the same signal-to-noise structure that made criterion 5
unreadable.

So every direction is scored three ways instead of one:

- `cos_desired`  the current criterion: alignment with the intended direction.
- `cos_random`   alignment with random unit vectors in the *same* latent space,
                 sampled per direction. This is the chance level, measured
                 rather than assumed, and it absorbs both the dimensionality
                 effect and any anisotropy of the space.
- `z_score`      (cos_desired - mean cos_random) / std cos_random: how far
                 above chance the realized movement actually is.

If `cos_desired` falls with training while `z_score` holds, control was never
lost and criterion 6 needs normalising. If both fall, the trade-off is real and
the remedy has to be a method change.

Effective dimensionality is reported alongside, as the participation ratio
`(sum lambda)^2 / sum lambda^2` of the latent covariance, so the mechanism can
be checked directly rather than inferred.

    python experiments/synthetic/diagnose_direction_control.py --seeds 0,2,4
    python experiments/synthetic/diagnose_direction_control.py \
        --seeds 0,1,2,3,4,5,6,7 --epochs 5,60 --replicates 2
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


def participation_ratio(z: np.ndarray) -> float:
    """Effective number of latent dimensions actually in use.

    `(sum lambda)^2 / sum lambda^2` over the covariance eigenvalues: equal to d
    when variance is spread evenly and to 1 when a single direction dominates.
    This is the quantity a raw cosine is sensitive to.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        return float("nan")
    c = np.cov(z - z.mean(0), rowvar=False)
    lam = np.linalg.eigvalsh(c)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def chance_cosines_isotropic(
    delta: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Cosines against unit directions drawn isotropically in the full space.

    Reported for reference only, because it is the **wrong null here** and
    measuring it showed why. The trained encoder collapses the latent space to
    roughly one effective dimension (participation ratio 1.18 out of 32), so
    both the realized displacement and every candidate direction live inside
    that same narrow subspace, while an isotropic draw almost never does.
    Comparing two vectors from a shared 1-d subspace against one vector from
    the full 32-d space inflates the z-score - it measured the collapse, not
    the control.
    """
    d = delta.shape[0]
    v = rng.normal(size=(n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    dn = delta / (np.linalg.norm(delta) + 1e-12)
    return v @ dn


def chance_cosines_permutation(
    delta: np.ndarray, own_id: int, all_vectors: list[tuple[int, np.ndarray]]
) -> np.ndarray:
    """Cosines against the *other candidate directions* - the right null.

    The question criterion 6 asks is whether collecting at a plan's metadata
    moved the latents along the direction that was **asked for**. The null that
    answers it is therefore "how well does this displacement align with a
    direction we might have asked for instead", i.e. the other candidates from
    the same generator. That controls for the effective dimensionality and for
    the structure of the direction set at once, where an isotropic draw
    controls for neither.

    If `cos(delta_i, v_i)` is no larger than `cos(delta_i, v_j)` for j != i,
    then collection moves the latents the same way whatever was requested, and
    the criterion is measuring a generic drift rather than control.
    """
    dn = delta / (np.linalg.norm(delta) + 1e-12)
    out = []
    for did, v in all_vectors:
        if did == own_id:
            continue
        vn = np.asarray(v, dtype=np.float64)
        vn = vn / (np.linalg.norm(vn) + 1e-12)
        out.append(float(np.dot(dn, vn)))
    return np.asarray(out, dtype=np.float64)


def measure(fx, model, cfg: dict, delta_scale: float, n_chance: int,
            n_per_anchor: int) -> dict:
    """Criterion 6, plus its chance level and the latent dimensionality."""
    from ldva.envs.synthetic.oracle import measure_realized_latent_movement

    pctx = PolicyContextRef.from_checkpoint(fx.ref, fx.train_ds.checkpoint_ids)
    z_all = model.encode_store(fx.store, pctx.features, ckpt_index=pctx.ckpt_index)
    model.set_dataset_context(z_all)
    pr = participation_ratio(z_all)

    clusters = LatentClustering(
        ClusteringConfig(n_clusters=cfg["n_clusters"], seed=fx.seeds["latent"])
    ).fit(z_all, fx.store.metadata)
    directions = DirectionGenerator(
        DirectionConfig(r_max=cfg["r_max"], delta_scale=delta_scale,
                        seed=fx.seeds["acquisition"])
    ).generate(clusters, z_all)
    if not directions:
        return {"error": "no directions"}
    mapper = MetadataMapper(
        fx.world.metadata_spec, MetadataMapperConfig(seed=fx.seeds["acquisition"]))
    mapper.fit(z_all, fx.store.metadata, clusters)
    directions, _ = filter_actionable_directions(
        directions, mapper, fx.store.metadata, ActionabilityConfig())
    if not directions:
        return {"error": "no actionable directions"}

    rng = fx.seeds.rng("acquisition")
    rng_chance = np.random.default_rng(fx.seeds["acquisition"] + 101)
    all_vectors = [(d.direction_id, np.asarray(d.vector, dtype=np.float64))
                   for d in directions]
    rows = []
    for d in directions:
        plan = mapper.plan_direction(d, 4, fx.store.metadata, rng=rng)
        m = measure_realized_latent_movement(
            model, fx.world, plan, d, fx.ref.features, rng,
            n_per_anchor=n_per_anchor)
        # the realized displacement, reconstructed from what the measurement
        # reports, so the chance level is computed against the same vector the
        # criterion scored
        along = m.get("displacement_along_direction", np.nan)
        norm = m.get("displacement_norm", np.nan)
        cos_des = m["direction_cosine"]
        v = np.asarray(d.vector, dtype=np.float64)
        # delta = along * v_unit + orthogonal part; its direction is all the
        # chance computation needs, and cos is scale free, so rebuild a vector
        # with the measured parallel/orthogonal split
        v_unit = v / (np.linalg.norm(v) + 1e-12)
        ortho_mag = float(np.sqrt(max(norm ** 2 - along ** 2, 0.0)))
        o = rng_chance.normal(size=v.shape)
        o -= np.dot(o, v_unit) * v_unit
        o /= np.linalg.norm(o) + 1e-12
        delta = along * v_unit + ortho_mag * o

        # the real null: the other directions the planner could have requested
        perm = chance_cosines_permutation(delta, d.direction_id, all_vectors)
        iso = chance_cosines_isotropic(delta, n_chance, rng_chance)
        p_mean = float(np.mean(perm)) if perm.size else float("nan")
        p_std = float(np.std(perm)) if perm.size else float("nan")
        rows.append({
            "direction_id": d.direction_id,
            "cos_desired": float(cos_des),
            # permutation null (primary)
            "perm_mean": p_mean,
            "perm_abs_mean": float(np.mean(np.abs(perm))) if perm.size else float("nan"),
            "perm_std": p_std,
            "z_score": (float((cos_des - p_mean) / max(p_std, 1e-9))
                        if perm.size else float("nan")),
            "frac_perm_below": (float(np.mean(perm < cos_des)) if perm.size
                                else float("nan")),
            # isotropic null (reference only; see the docstring)
            "iso_abs_mean": float(np.mean(np.abs(iso))),
            "z_score_isotropic": float(
                (cos_des - float(np.mean(iso))) / max(float(np.std(iso)), 1e-9)),
            "displacement_norm": float(norm),
        })

    cos = np.array([r["cos_desired"] for r in rows])
    z = np.array([r["z_score"] for r in rows])
    ch = np.array([r["perm_abs_mean"] for r in rows])
    iso = np.array([r["iso_abs_mean"] for r in rows])
    return {
        "n_directions": len(rows),
        "latent_dim": int(z_all.shape[1]),
        "participation_ratio": pr,
        "cos_desired_mean": float(np.nanmean(cos)),
        "chance_abs_mean": float(np.nanmean(ch)),
        "chance_abs_mean_isotropic": float(np.nanmean(iso)),
        "z_score_isotropic_mean": float(
            np.nanmean([r["z_score_isotropic"] for r in rows])),
        # the criterion, expressed relative to what a random direction gets
        "cos_over_chance": float(np.nanmean(cos) / max(np.nanmean(ch), 1e-9)),
        "z_score_mean": float(np.nanmean(z)),
        "z_score_sem": float(np.nanstd(z) / max(np.sqrt(len(z)), 1)),
        "frac_above_chance": float(np.mean(cos > ch)),
        "null": "permutation over the other candidate directions",
        "per_direction": rows,
    }


def run_cell(fx, cfg: dict, epochs: int, args, say, replicates: int) -> dict:
    local = dict(cfg)
    local["epochs"] = epochs
    cos, zs, pr, fits = [], [], [], []
    for k in range(max(replicates, 1)):
        fx.cfg = local
        model, final = fx.train(weights=LossWeights(1.0, 1.0, 0.1, 0.01),
                                seed_offset=7919 * k)
        m = measure(fx, model, local, args.delta_scale, args.n_chance,
                    args.n_per_anchor)
        if "error" in m:
            continue
        cos.append(m["cos_desired_mean"])
        zs.append(m["z_score_mean"])
        pr.append(m["participation_ratio"])
        fits.append(final.get("effect_spearman", np.nan))
    if not cos:
        return {"epochs": epochs, "error": "no usable replicate"}
    out = {
        "epochs": epochs,
        "replicates": len(cos),
        "effect_spearman": float(np.nanmean(fits)),
        "cos_desired": float(np.nanmean(cos)),
        "cos_desired_std": float(np.nanstd(cos)),
        "z_score": float(np.nanmean(zs)),
        "z_score_std": float(np.nanstd(zs)),
        "participation_ratio": float(np.nanmean(pr)),
    }
    say(f"    epochs={epochs:<3d} fit={out['effect_spearman']:+.3f} | "
        f"C6={out['cos_desired']:+.3f}+/-{out['cos_desired_std']:.3f} "
        f"z={out['z_score']:+.2f}+/-{out['z_score_std']:.2f} "
        f"eff_dim={out['participation_ratio']:.2f}")
    return out


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--epochs", type=str, default="5,60")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--delta-scale", type=float, default=0.4)
    ap.add_argument("--n-chance", type=int, default=2000,
                    help="random directions per measured direction")
    ap.add_argument("--n-per-anchor", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="runs/diagnose_direction_control")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--experiment", type=str, default="dirctrl")
    args = ap.parse_args()

    mod = _load_fixture_module()
    cfg = dict(mod.DEFAULTS)
    if args.quick:
        cfg.update(mod.QUICK)
    cfg["label_lr"] = 0.3

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    say = lambda m: print(f"[dirctrl] {m}", flush=True)  # noqa: E731
    say(f"{len(seeds)} seeds x epochs {epochs} x {args.replicates} replicates, "
        f"{args.n_chance} chance directions each")

    log = make_logger(
        enabled=args.wandb, project=args.wandb_project,
        name=f"{args.experiment}-direction-control",
        group=f"{args.experiment}-direction-control", job_type="diagnosis",
        config={"seeds": seeds, "epochs": epochs, "replicates": args.replicates,
                "n_chance": args.n_chance, "delta_scale": args.delta_scale},
        tags=("synthetic", "criterion6", args.experiment),
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
        }, out_dir / "diagnose_direction_control.json")

    for s in seeds:
        say(f"  seed {s}")
        try:
            fx = mod.Fixture(cfg, s)
        except Exception as e:
            say(f"    fixture FAILED {type(e).__name__}: {e}")
            continue
        for e in epochs:
            try:
                r = run_cell(fx, cfg, e, args, say, args.replicates)
                r["seed"] = s
                cells.append(r)
            except Exception as exc:
                say(f"    epochs={e}: FAILED {type(exc).__name__}: {exc}")
                cells.append({"seed": s, "epochs": e,
                              "error": f"{type(exc).__name__}: {exc}"})
            _save()

    a = analyse(cells)
    _save()
    _print(cells, a, out_dir)
    if log.active:
        log.table("cells",
                  ["seed", "epochs", "effect_spearman", "cos_desired",
                   "z_score", "participation_ratio"],
                  [[c["seed"], c["epochs"], c.get("effect_spearman", float("nan")),
                    c.get("cos_desired", float("nan")), c.get("z_score", float("nan")),
                    c.get("participation_ratio", float("nan"))]
                   for c in cells if "error" not in c])
        log.summary({f"analysis/{k}": v for k, v in a.items()
                     if isinstance(v, (int, float, bool, str))})
        log.finish()
    return a


def analyse(cells: list[dict]) -> dict:
    """Does the raw cosine fall while the above-chance z-score holds?"""
    ok = [c for c in cells if "error" not in c and "cos_desired" in c]
    if len(ok) < 2:
        return {"n_cells": len(ok)}
    by_seed: dict[int, list] = {}
    for c in ok:
        by_seed.setdefault(c["seed"], []).append(c)

    per_seed, d_cos, d_z, d_pr = [], [], [], []
    for sd, g in sorted(by_seed.items()):
        g = sorted(g, key=lambda r: r["epochs"])
        if len(g) < 2:
            continue
        lo, hi = g[0], g[-1]
        dc = hi["cos_desired"] - lo["cos_desired"]
        dz = hi["z_score"] - lo["z_score"]
        dp = hi["participation_ratio"] - lo["participation_ratio"]
        n = max(min(lo.get("replicates", 1), hi.get("replicates", 1)), 1)
        se_c = float(np.sqrt(lo.get("cos_desired_std", 0.0) ** 2 / n
                             + hi.get("cos_desired_std", 0.0) ** 2 / n))
        se_z = float(np.sqrt(lo.get("z_score_std", 0.0) ** 2 / n
                             + hi.get("z_score_std", 0.0) ** 2 / n))
        per_seed.append({
            "seed": sd,
            "fit_low": lo["effect_spearman"], "fit_high": hi["effect_spearman"],
            "cos_low": lo["cos_desired"], "cos_high": hi["cos_desired"],
            "delta_cos": float(dc), "se_cos": se_c,
            "z_low": lo["z_score"], "z_high": hi["z_score"],
            "delta_z": float(dz), "se_z": se_z,
            "pr_low": lo["participation_ratio"], "pr_high": hi["participation_ratio"],
            "delta_pr": float(dp),
            "cos_fell": bool(dc < -2 * se_c) if se_c > 0 else False,
            "z_fell": bool(dz < -2 * se_z) if se_z > 0 else False,
        })
        d_cos.append(dc)
        d_z.append(dz)
        d_pr.append(dp)

    d_cos = np.array(d_cos, dtype=np.float64)
    d_z = np.array(d_z, dtype=np.float64)
    d_pr = np.array(d_pr, dtype=np.float64)
    cos_all = np.array([c["cos_desired"] for c in ok], dtype=np.float64)
    pr_all = np.array([c["participation_ratio"] for c in ok], dtype=np.float64)
    m = np.isfinite(cos_all) & np.isfinite(pr_all)

    n_cos_fell = int(sum(1 for r in per_seed if r["cos_fell"]))
    n_z_fell = int(sum(1 for r in per_seed if r["z_fell"]))
    return {
        "n_cells": len(ok),
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "delta_cos_mean": float(np.nanmean(d_cos)),
        "delta_cos_sem": float(np.nanstd(d_cos) / max(np.sqrt(len(d_cos)), 1)),
        "delta_z_mean": float(np.nanmean(d_z)),
        "delta_z_sem": float(np.nanstd(d_z) / max(np.sqrt(len(d_z)), 1)),
        "delta_participation_ratio_mean": float(np.nanmean(d_pr)),
        "n_seeds_cos_fell": n_cos_fell,
        "n_seeds_z_fell": n_z_fell,
        # if the raw cosine tracks dimensionality, that is the artefact
        "rho_cos_vs_participation_ratio": (
            float(spearmanr(pr_all[m], cos_all[m]).statistic) if m.sum() >= 3
            else float("nan")),
        "verdict": (
            "criterion 6 is partly a DIMENSIONALITY ARTEFACT: the raw cosine "
            "falls while the above-chance z-score holds"
            if n_cos_fell > n_z_fell and n_cos_fell > 0 else
            "control genuinely degrades: both the raw cosine and the "
            "above-chance z-score fall"
            if n_z_fell > 0 and n_z_fell >= n_cos_fell else
            "neither falls resolvably at this replicate count"
        ),
    }


def _print(cells: list[dict], a: dict, out_dir: Path) -> None:
    w = 104
    print("\n" + "=" * w)
    print("CRITERION 6: RAW COSINE vs ABOVE-CHANCE z-SCORE")
    print("=" * w)
    ok = [c for c in cells if "error" not in c and "cos_desired" in c]
    print(f"  {'seed':>5s} {'epochs':>7s} {'fit':>8s} {'C6 (cos)':>10s} "
          f"{'z vs chance':>12s} {'eff_dim':>9s}")
    for c in sorted(ok, key=lambda x: (x["seed"], x["epochs"])):
        print(f"  {c['seed']:>5d} {c['epochs']:>7d} {c['effect_spearman']:>+8.3f} "
              f"{c['cos_desired']:>+10.3f} {c['z_score']:>+12.2f} "
              f"{c['participation_ratio']:>9.2f}")
    if a.get("n_seeds"):
        print("-" * w)
        print(f"  {'seed':>5s} {'dC6':>9s} {'SE':>7s} {'dz':>9s} {'SE':>7s} "
              f"{'d eff_dim':>10s}  fell?")
        for r in a["per_seed"]:
            print(f"  {r['seed']:>5d} {r['delta_cos']:>+9.3f} {r['se_cos']:>7.3f} "
                  f"{r['delta_z']:>+9.2f} {r['se_z']:>7.2f} "
                  f"{r['delta_pr']:>+10.2f}  "
                  f"cos={'Y' if r['cos_fell'] else 'n'} z={'Y' if r['z_fell'] else 'n'}")
        print("-" * w)
        print(f"  mean dC6 {a['delta_cos_mean']:+.3f} +/- {a['delta_cos_sem']:.3f}"
              f"   mean dz {a['delta_z_mean']:+.2f} +/- {a['delta_z_sem']:.2f}"
              f"   mean d eff_dim {a['delta_participation_ratio_mean']:+.2f}")
        print(f"  cosine fell resolvably in {a['n_seeds_cos_fell']}/{a['n_seeds']} seeds,"
              f" z-score in {a['n_seeds_z_fell']}/{a['n_seeds']}")
        print(f"  rho(cosine, effective dim) = "
              f"{a['rho_cos_vs_participation_ratio']:+.3f}")
        print(f"\n  VERDICT: {a['verdict']}")
    print(f"  report: {out_dir / 'diagnose_direction_control.json'}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
