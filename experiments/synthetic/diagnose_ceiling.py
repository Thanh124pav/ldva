"""Is criterion 6 achievable on this benchmark at all?

Every configuration tried so far fails criterion 6 when measured against the
right null: over 20 cells spanning metric-loss weights 0 to 5 and latent
dimensions 8 and 32, the best direction-specificity z-score was +1.41 and none
reached 2. The latent space sits at 1.6-2.0 effective dimensions regardless, so
the collapse is neither a loss-weighting problem nor a capacity problem.

That leaves an upstream possibility which has to be checked before any more
model changes: **the benchmark may not contain enough effect geometry to
control.** The synthetic world has 4 true hidden factors and a 3-dimensional
metadata space, and if the effect signal that leave-one-out can actually
measure lives on ~2 of those, then an encoder collapsing to ~2 dimensions has
lost nothing and criterion 6 is unreachable here no matter what the model does.

Three ceilings are measured, each an upper bound the learned model cannot
exceed:

1. **Effect-label dimensionality.** The samples-by-contexts matrix of
   leave-one-out effects, as the participation ratio of its singular values.
   This is how many independent directions of *effect variation* the
   supervision contains. If it is ~2, a 2-dimensional latent is correct.
2. **True-latent ceiling.** Run the whole direction pipeline on the world's
   ground-truth factors `store.latent_true` instead of the learned encoding.
   This is the best any encoder could do, since it *is* the generative
   representation. If specificity still falls short of 2, the limit is the
   benchmark, not the model.
3. **Metadata ceiling.** The same pipeline on the raw metadata. Directions in
   metadata space are executable by construction - collection happens in
   exactly those coordinates - so this isolates how much of the failure comes
   from the encoding step versus the direction/mapping machinery.

Together these separate three explanations that have been conflated: a bad
encoder, a bad direction generator, and a benchmark with nothing to find.

    python experiments/synthetic/diagnose_ceiling.py --seeds 0,1,2,3,4,5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

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
    estimate_reach,
    participation_ratio,
)
from ldva.analysis.wandb_logger import make_logger  # noqa: E402
from ldva.policy.checkpoints import PolicyContextRef  # noqa: E402
from ldva.utils import run_provenance, save_json  # noqa: E402


def _load_fixture_module():
    path = ROOT / "experiments" / "synthetic" / "run_ablations.py"
    spec = importlib.util.spec_from_file_location("ldva_abl_fixture", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ldva_abl_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


def effect_label_dimensionality(records, n_samples: int) -> dict:
    """How many independent directions of effect variation the labels contain.

    Builds the samples-by-contexts matrix of leave-one-out effects and takes
    the participation ratio of its singular values. This is an upper bound on
    the effect geometry any encoder can represent: a latent space cannot carry
    structure the supervision does not contain.
    """
    ctx_ids = {id(r): i for i, r in enumerate(records)}
    M = np.full((n_samples, len(records)), np.nan, dtype=np.float64)
    for r in records:
        j = ctx_ids[id(r)]
        M[np.asarray(r.batch_sample_ids, dtype=np.int64), j] = r.per_sample_effects
    # keep samples seen in enough contexts for a profile to mean anything
    seen = np.sum(np.isfinite(M), axis=1)
    rows = M[seen >= 3]
    if rows.shape[0] < 3:
        return {"n_samples_with_profile": int(rows.shape[0]),
                "effect_participation_ratio": float("nan")}
    # mean-impute the gaps so the SVD is defined; the row means carry the
    # between-sample signal, which is what the dimensionality question is about
    col_mean = np.nanmean(rows, axis=0)
    idx = np.where(np.isnan(rows))
    filled = rows.copy()
    filled[idx] = np.take(col_mean, idx[1])
    filled = filled - filled.mean(0, keepdims=True)
    sv = np.linalg.svd(filled, compute_uv=False)
    sv2 = sv ** 2
    sv2 = sv2[sv2 > 0]
    pr = float(sv2.sum() ** 2 / np.sum(sv2 ** 2)) if sv2.size else float("nan")
    return {
        "n_samples_with_profile": int(rows.shape[0]),
        "n_contexts": int(rows.shape[1]),
        "effect_participation_ratio": pr,
        "top_singular_share": float(sv2[0] / sv2.sum()) if sv2.size else float("nan"),
        "singular_share_top3": (float(sv2[:3].sum() / sv2.sum())
                                if sv2.size >= 3 else float("nan")),
    }


def pipeline_specificity(fx, z: np.ndarray, cfg: dict, label: str,
                         delta_scale: float, n_per_anchor: int,
                         encode_fn) -> dict:
    """Run clustering -> directions -> mapping -> realized movement on `z`.

    `encode_fn(store)` must return the representation for a freshly collected
    store in the same space as `z`, so the realized movement is measured in the
    representation whose ceiling is being tested.
    """
    from ldva.acquisition.metadata_mapper import evaluate_realized_direction

    out = {"label": label, "representation_dim": int(z.shape[1]),
           "participation_ratio": participation_ratio(z)}
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=cfg["n_clusters"], seed=fx.seeds["latent"])
    ).fit(z, fx.store.metadata)
    directions = DirectionGenerator(
        DirectionConfig(r_max=cfg["r_max"], delta_scale=delta_scale,
                        seed=fx.seeds["acquisition"])
    ).generate(clusters, z)
    if not directions:
        out["error"] = "no directions"
        return out
    mcfg = MetadataMapperConfig(seed=fx.seeds["acquisition"])
    if cfg.get("max_step_norm") is not None:
        mcfg.max_step_norm = float(cfg["max_step_norm"])
    mapper = MetadataMapper(fx.world.metadata_spec, mcfg)
    mapper.fit(z, fx.store.metadata, clusters)
    directions, act = filter_actionable_directions(
        directions, mapper, fx.store.metadata, ActionabilityConfig())
    if not directions:
        out["error"] = "no actionable directions"
        return out

    rng = fx.seeds.rng("acquisition")
    deltas, vectors, cos = {}, {}, []
    for d in directions:
        plan = mapper.plan_direction(d, 4, fx.store.metadata, rng=rng)
        per_anchor = []
        for row_m, anchor_m in zip(plan.metadata, plan.anchor_metadata):
            shared = int(rng.integers(0, 2 ** 31 - 1))
            base = fx.world.build_store(
                n_per_anchor, np.random.default_rng(shared),
                metadata=np.tile(anchor_m, (n_per_anchor, 1)), round_id=-2)
            new = fx.world.build_store(
                n_per_anchor, np.random.default_rng(shared),
                metadata=np.tile(row_m, (n_per_anchor, 1)), round_id=-2)
            per_anchor.append(encode_fn(new).mean(0) - encode_fn(base).mean(0))
        per_anchor = np.asarray(per_anchor, dtype=np.float64)
        res = evaluate_realized_direction(
            np.zeros((1, per_anchor.shape[1])), per_anchor, d.vector)
        cos.append(res["direction_cosine"])
        deltas[d.direction_id] = per_anchor.mean(0)
        vectors[d.direction_id] = np.asarray(d.vector, dtype=np.float64)

    spec = direction_specificity(deltas, vectors)
    # The direct quantity: how many independent directions the candidate SET
    # spans. If the candidates are nearly parallel then any displacement
    # aligns with any of them, and specificity stays low however perfectly the
    # requests are executed - which is exactly what metadata space shows
    # (cos = 1.000 yet z = +1.71).
    V = np.stack([vectors[i] / (np.linalg.norm(vectors[i]) + 1e-12)
                  for i in sorted(vectors)]) if vectors else np.zeros((0, 1))
    out.update({
        "direction_set_participation_ratio": (
            participation_ratio(V) if V.shape[0] >= 2 else float("nan")),
        "mean_abs_pairwise_cos": (
            float(np.mean(np.abs(V @ V.T)[np.triu_indices(V.shape[0], k=1)]))
            if V.shape[0] >= 2 else float("nan")),
        "n_directions": len(directions),
        "actionable_survival": act["survival_rate"],
        "cos_desired_mean": float(np.nanmean(cos)) if cos else float("nan"),
        "specificity_gap": spec.get("gap_mean", float("nan")),
        "specificity_gap_sem": spec.get("gap_sem", float("nan")),
        "specificity_z": spec.get("z_score_mean", float("nan")),
        "null_abs_mean": spec.get("null_abs_mean", float("nan")),
        "frac_above_2sd": spec.get("frac_directions_above_2sd", float("nan")),
        "passes_criterion6": bool(spec.get("gap_mean", float("nan")) >= 0.3),
    })
    return out


def run_seed(fx, cfg: dict, args, say) -> dict:
    out: dict = {"seed": fx.seeds.base}

    # ---- reach: how far a LOCAL LINEAR map can be trusted ------------------
    # Measured in *normalised* metadata units, because that is the unit the
    # mapper's trust region (`max_step_norm`) is expressed in. A step longer
    # than the reach asks the linear solve for something the linearisation
    # cannot deliver, which is the failure the hand-picked constant was hiding.
    spec = fx.world.metadata_spec
    centre_norm = spec.normalize(fx.store.metadata).mean(0)

    def _encode_norm(m_norm):
        return fx.world.latent_from_metadata(spec.denormalize(np.clip(m_norm, 0, 1)))

    reach = estimate_reach(_encode_norm, centre_norm,
                           rng=np.random.default_rng(fx.seeds["acquisition"] + 7))
    out["reach"] = reach
    if args.auto_step and reach["reach"] is not None:
        cfg = dict(cfg)
        cfg["max_step_norm"] = float(reach["reach"])
    elif args.max_step_norm is not None:
        cfg = dict(cfg)
        cfg["max_step_norm"] = float(args.max_step_norm)
    say(f"    reach (normalised metadata) = {reach['reach']}"
        f"   mapper step = {cfg.get('max_step_norm', 'default 0.35')}")

    # ---- ceiling 1: what the labels contain --------------------------------
    out["effect_labels"] = effect_label_dimensionality(fx.records, len(fx.store))
    say(f"    effect labels: eff_dim="
        f"{out['effect_labels']['effect_participation_ratio']:.2f} "
        f"(top singular share "
        f"{out['effect_labels'].get('top_singular_share', float('nan')):.2f})")

    # ---- ceiling 2: the world's own generative factors --------------------
    lt = fx.store.latent_true
    if lt is not None and np.asarray(lt).ndim == 2:
        u = np.asarray(lt, dtype=np.float64)
        out["true_latent"] = pipeline_specificity(
            fx, u, cfg, "true_latent", args.delta_scale, args.n_per_anchor,
            encode_fn=lambda s: np.asarray(s.latent_true, dtype=np.float64),
        )
        r = out["true_latent"]
        say(f"    TRUE latent  : eff_dim={r['participation_ratio']:.2f} "
            f"gap={r.get('specificity_gap', float('nan')):+.3f} "
            f"cos={r.get('cos_desired_mean', float('nan')):+.3f} "
            f"pass={r.get('passes_criterion6')}")
    else:
        out["true_latent"] = {"error": "world exposes no latent_true"}

    # ---- ceiling 3: raw metadata, executable by construction --------------
    mn = fx.world.metadata_spec.normalize(fx.store.metadata)
    out["metadata_space"] = pipeline_specificity(
        fx, np.asarray(mn, dtype=np.float64), cfg, "metadata",
        args.delta_scale, args.n_per_anchor,
        encode_fn=lambda s: np.asarray(
            fx.world.metadata_spec.normalize(s.metadata), dtype=np.float64),
    )
    r = out["metadata_space"]
    say(f"    METADATA     : eff_dim={r['participation_ratio']:.2f} "
        f"gap={r.get('specificity_gap', float('nan')):+.3f} "
        f"cos={r.get('cos_desired_mean', float('nan')):+.3f} "
        f"dir_set_dim={r.get('direction_set_participation_ratio', float('nan')):.2f} "
        f"pair_cos={r.get('mean_abs_pairwise_cos', float('nan')):.3f} "
        f"pass={r.get('passes_criterion6')}")

    # ---- the learned model, for comparison --------------------------------
    model, final = fx.train(seed_offset=0)
    pctx = PolicyContextRef.from_checkpoint(fx.ref, fx.train_ds.checkpoint_ids)
    z = model.encode_store(fx.store, pctx.features, ckpt_index=pctx.ckpt_index)
    model.set_dataset_context(z)
    out["learned"] = pipeline_specificity(
        fx, z, cfg, "learned", args.delta_scale, args.n_per_anchor,
        encode_fn=lambda s: model.encode_store(
            s, pctx.features, ckpt_index=pctx.ckpt_index).astype(np.float64),
    )
    out["learned"]["effect_spearman"] = final.get("effect_spearman", float("nan"))
    r = out["learned"]
    say(f"    LEARNED      : eff_dim={r['participation_ratio']:.2f} "
        f"gap={r.get('specificity_gap', float('nan')):+.3f} "
        f"cos={r.get('cos_desired_mean', float('nan')):+.3f} "
        f"pass={r.get('passes_criterion6')}")
    return out


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--delta-scale", type=float, default=0.4)
    ap.add_argument("--n-per-anchor", type=int, default=24)
    ap.add_argument("--map-kind", type=str, default=None, choices=("tanh", "rff"))
    ap.add_argument("--rff-omega", type=float, default=None,
                    help="inverse kernel bandwidth: higher is richer but has a "
                         "shorter reach, so the mapper's step must shrink with it")
    ap.add_argument("--world-latent-dim", type=int, default=None)
    ap.add_argument("--world-obs-dim", type=int, default=None)
    ap.add_argument("--max-step-norm", type=float, default=None,
                    help="mapper trust region; 'auto' behaviour is to take the "
                         "measured reach, see --auto-step")
    ap.add_argument("--auto-step", action="store_true",
                    help="set the mapper trust region from the measured reach "
                         "instead of a hand-picked constant")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="runs/diagnose_ceiling")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--experiment", type=str, default="ceiling")
    args = ap.parse_args()

    mod = _load_fixture_module()
    cfg = dict(mod.DEFAULTS)
    if args.quick:
        cfg.update(mod.QUICK)
    cfg["label_lr"] = 0.3
    world: dict = {}
    if args.map_kind:
        world["map_kind"] = args.map_kind
    if args.rff_omega is not None:
        world["rff_omega"] = args.rff_omega
    if args.world_latent_dim is not None:
        world["latent_dim"] = args.world_latent_dim
    if args.world_obs_dim is not None:
        world["obs_dim"] = args.world_obs_dim
    if world:
        cfg["world"] = world

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    say = lambda m: print(f"[ceiling] {m}", flush=True)  # noqa: E731
    if world:
        say(f"world overrides: {world}")
    say(f"{len(seeds)} seeds: effect-label dim, true-latent ceiling, "
        f"metadata ceiling, learned model")

    log = make_logger(
        enabled=args.wandb, project=args.wandb_project,
        name=f"{args.experiment}-ceiling", group=f"{args.experiment}-ceiling",
        job_type="diagnosis", config={"seeds": seeds},
        tags=("synthetic", "ceiling", args.experiment),
    )

    out_dir = Path(args.out)
    rows: list[dict] = []

    def _save():
        save_json({
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items()},
            "args": vars(args),
            "seeds": rows,
            "analysis": analyse(rows),
            "provenance": run_provenance({"experiment": args.experiment}),
        }, out_dir / "diagnose_ceiling.json")

    for s in seeds:
        say(f"  seed {s}")
        try:
            fx = mod.Fixture(cfg, s)
            rows.append(run_seed(fx, cfg, args, say))
        except Exception as e:
            import traceback
            say(f"    FAILED {type(e).__name__}: {e}")
            say("    " + traceback.format_exc().strip().splitlines()[-3].strip())
            rows.append({"seed": s, "error": f"{type(e).__name__}: {e}"})
        _save()

    a = analyse(rows)
    _save()
    _print(rows, a, out_dir)
    if log.active:
        log.table("ceilings",
                  ["seed", "representation", "eff_dim", "specificity_z",
                   "cos_desired", "n_directions", "passes"],
                  [[r["seed"], k, r[k].get("participation_ratio", float("nan")),
                    r[k].get("specificity_gap", float("nan")),
                    r[k].get("cos_desired_mean", float("nan")),
                    r[k].get("n_directions", 0), r[k].get("passes_criterion6", False)]
                   for r in rows if "error" not in r
                   for k in ("true_latent", "metadata_space", "learned")
                   if isinstance(r.get(k), dict) and "error" not in r[k]])
        log.summary({f"analysis/{k}": v for k, v in a.items()
                     if isinstance(v, (int, float, bool, str))})
        log.finish()
    return a


def analyse(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n_seeds": 0}

    def col(key, field):
        v = [r[key].get(field, np.nan) for r in ok
             if isinstance(r.get(key), dict) and "error" not in r[key]]
        v = np.array(v, dtype=np.float64)
        return v[np.isfinite(v)]

    eff_lbl = np.array([r["effect_labels"].get("effect_participation_ratio", np.nan)
                        for r in ok], dtype=np.float64)
    eff_lbl = eff_lbl[np.isfinite(eff_lbl)]

    out = {"n_seeds": len(ok)}
    out["effect_label_eff_dim_mean"] = (
        float(eff_lbl.mean()) if eff_lbl.size else float("nan"))
    for key, name in (("true_latent", "true"), ("metadata_space", "metadata"),
                      ("learned", "learned")):
        z = col(key, "specificity_gap")
        pr = col(key, "participation_ratio")
        cos = col(key, "cos_desired_mean")
        out[f"{name}_eff_dim_mean"] = float(pr.mean()) if pr.size else float("nan")
        out[f"{name}_specificity_gap_mean"] = float(z.mean()) if z.size else float("nan")
        out[f"{name}_cos_mean"] = float(cos.mean()) if cos.size else float("nan")
        ds = col(key, "direction_set_participation_ratio")
        pc = col(key, "mean_abs_pairwise_cos")
        out[f"{name}_direction_set_dim_mean"] = (
            float(ds.mean()) if ds.size else float("nan"))
        out[f"{name}_pairwise_cos_mean"] = float(pc.mean()) if pc.size else float("nan")
        out[f"{name}_n_pass"] = int(np.sum(z >= 0.3)) if z.size else 0
        out[f"{name}_n"] = int(z.size)

    tz = out.get("true_specificity_gap_mean", float("nan"))
    mz = out.get("metadata_specificity_gap_mean", float("nan"))
    lz = out.get("learned_specificity_gap_mean", float("nan"))
    if np.isfinite(tz) and tz < 0.3 and np.isfinite(mz) and mz < 0.3:
        verdict = ("BENCHMARK CEILING: even the world's true latents and the raw "
                   "metadata fail criterion 6, so it is not achievable on this "
                   "synthetic world regardless of the model")
    elif np.isfinite(tz) and tz >= 0.3 and np.isfinite(lz) and lz < 0.3:
        verdict = ("MODEL GAP: the true latents clear criterion 6 but the learned "
                   "encoding does not - the headroom is in the encoder")
    elif np.isfinite(mz) and mz >= 0.3 and np.isfinite(tz) and tz < 0.3:
        verdict = ("REPRESENTATION GAP: metadata-space directions are specific but "
                   "latent ones are not - the loss is in the encoding step")
    else:
        verdict = "inconclusive - see the per-seed table"
    out["verdict"] = verdict
    return out


def _print(rows: list[dict], a: dict, out_dir: Path) -> None:
    w = 104
    print("\n" + "=" * w)
    print("CEILINGS FOR CRITERION 6 (direction specificity, z >= 2 to pass)")
    print("=" * w)
    print(f"  {'seed':>5s} {'label dim':>10s} | {'representation':>14s} "
          f"{'eff_dim':>9s} {'spec gap':>9s} {'cos':>8s} {'n_dir':>6s} {'pass':>6s}")
    for r in rows:
        if "error" in r:
            print(f"  {r['seed']:>5d}  ERROR {r['error']}")
            continue
        ld = r["effect_labels"].get("effect_participation_ratio", float("nan"))
        first = True
        for key, nm in (("true_latent", "true latent"),
                        ("metadata_space", "metadata"), ("learned", "learned")):
            d = r.get(key, {})
            if not isinstance(d, dict) or "specificity_gap" not in d:
                continue
            lead = f"  {r['seed']:>5d} {ld:>10.2f} |" if first else f"  {'':>5s} {'':>10s} |"
            first = False
            print(f"{lead} {nm:>14s} {d.get('participation_ratio', float('nan')):>9.2f} "
                  f"{d.get('specificity_gap', float('nan')):>+9.3f} "
                  f"{d.get('cos_desired_mean', float('nan')):>+8.3f} "
                  f"{d.get('n_directions', 0):>6d} "
                  f"{('YES' if d.get('passes_criterion6') else 'no'):>6s}")
    print("-" * w)
    print(f"  effect-label effective dim (mean) "
          f"{a.get('effect_label_eff_dim_mean', float('nan')):.2f}")
    for nm in ("true", "metadata", "learned"):
        print(f"  {nm:>9s}: eff_dim {a.get(f'{nm}_eff_dim_mean', float('nan')):>5.2f}  "
              f"dir_set_dim {a.get(f'{nm}_direction_set_dim_mean', float('nan')):>5.2f}  "
              f"pair|cos| {a.get(f'{nm}_pairwise_cos_mean', float('nan')):>5.3f}  "
              f"spec gap {a.get(f'{nm}_specificity_gap_mean', float('nan')):>+6.3f}  "
              f"cos {a.get(f'{nm}_cos_mean', float('nan')):>+6.3f}  "
              f"pass {a.get(f'{nm}_n_pass', 0)}/{a.get(f'{nm}_n', 0)}")
    print(f"\n  VERDICT: {a.get('verdict')}")
    print(f"  report: {out_dir / 'diagnose_ceiling.json'}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
