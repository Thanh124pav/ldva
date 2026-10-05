"""Stage 0: synthetic sanity test (PLAN.md 14, 15, 20 Phase 0).

Runs the whole LDVA pipeline on the synthetic world and checks the six success
criteria of PLAN.md 14 explicitly. PLAN.md 14 is blunt about the purpose:
"Do not move to robotics if this fails", so this script's real output is the
pass/fail table at the end, not the plots.

    python experiments/synthetic/run_stage0.py --quick
    python experiments/synthetic/run_stage0.py --config configs/env/synthetic.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ldva.acquisition.baselines import BaselineSuite  # noqa: E402
from ldva.acquisition.beam_search import beam_search  # noqa: E402
from ldva.acquisition.clustering import ClusteringConfig, LatentClustering  # noqa: E402
from ldva.acquisition.directions import (  # noqa: E402
    DirectionConfig,
    DirectionGenerator,
    directions_report,
)
from ldva.acquisition.exact_search import exact_search  # noqa: E402
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
    n_allocations,
)
from ldva.analysis import plotting as P  # noqa: E402
from ldva.analysis.acquisition_calibration import (  # noqa: E402
    CalibrationRecord,
    calibration_report,
    candidate_allocations,
    deduplicate,
)
from ldva.analysis.direction_validation import (  # noqa: E402
    direction_specificity,
    participation_ratio,
)
from ldva.analysis.latent_geometry import GeometryGate, latent_geometry_report  # noqa: E402
from ldva.analysis.wandb_logger import make_logger  # noqa: E402
from ldva.data.context_dataset import ContextDataset  # noqa: E402
from ldva.data.effect_profiles import EffectProfileTable  # noqa: E402
from ldva.envs.synthetic.generator import SyntheticConfig, SyntheticWorld  # noqa: E402
from ldva.envs.synthetic.oracle import (  # noqa: E402
    OracleConfig,
    SyntheticAcquisitionOracle,
    measure_realized_latent_movement,
)
from ldva.models.datamodel import LDVAConfig, LDVADataModel  # noqa: E402
from ldva.policy.checkpoints import PolicyContextRef  # noqa: E402
from ldva.policy.evaluate import evaluate_bc  # noqa: E402
from ldva.policy.train import BCTrainConfig, train_bc  # noqa: E402
from ldva.supervision.bc_task import BCSupervisionTask  # noqa: E402
from ldva.supervision.generate_context_records import (  # noqa: E402
    ContextGenConfig,
    generate_context_records,
)
from ldva.supervision.gradient_alignment import GradientAlignmentEstimator  # noqa: E402
from ldva.supervision.leave_one_out import LeaveOneOutEstimator  # noqa: E402
from ldva.training.losses import LossWeights  # noqa: E402
from ldva.training.train_datamodel import TrainConfig, train_datamodel  # noqa: E402
from ldva.utils import (  # noqa: E402
    SeedBundle,
    load_config,
    run_provenance,
    save_json,
    set_seed,
)

DEFAULTS = dict(
    n_samples=300,
    n_eval=400,
    policy_steps=150,
    policy_snapshot_every=30,
    policy_restarts=3,
    contexts_per_sample=24,
    context_batch_size=8,
    # Label update size matters more than it looks. The additive part of a batch
    # gain is O(lr) but the interaction part is O(lr^2), so a small lr produces
    # gains that are nearly additive and Stage 0 cannot test the set-level
    # claim at all. Measured on this world: at lr=0.1 only 7% of gain variance
    # is composition-dependent and an additive fit reaches R^2 +0.45, while at
    # lr=0.3 / 4 steps composition explains 84% and the additive fit fails
    # (R^2 -0.36). The cost is that the cheap first-order proxies stop tracking
    # the leave-one-out target at this lr, which the calibration report shows.
    label_lr=0.3,
    label_steps=4,
    latent_dim=32,
    hidden=(128, 128),
    epochs=60,
    n_clusters=4,
    #: MIXED random allocations in the calibration candidate set, on top of the
    #: one-hot allocations which are always included.
    #:
    #: Which allocation type carries the realized signal is **seed-dependent**,
    #: so this is deliberately NOT tuned. On E0 seed 0 the mixed allocations
    #: spanned a realized range of only 0.098 and diluted the pooled Spearman
    #: from +0.522 to +0.177; on seed 1 the same mixed allocations were the
    #: most informative group of all (+0.402 on their own, against +0.271 for
    #: planner+one-hot) and are what let that seed pass. Choosing the set per
    #: seed by whichever gives the highest correlation would be selecting on
    #: the outcome, which is exactly what invalidates a gate - so all three
    #: types are included on a fixed rule and `by_source` reports the
    #: breakdown as a diagnostic.
    n_random_allocations=14,
    r_max=2,
    delta_scale=0.4,
    budget=8,
    n_mc=16,
    exact_max_allocations=200_000,
    min_reachability=0.5,
    min_achievable_cosine=0.9,
    beam_widths=(1, 5, 10, 20),
    seed=0,
)
QUICK = dict(
    n_samples=120, n_eval=200, policy_steps=90, policy_snapshot_every=30,
    policy_restarts=2, contexts_per_sample=12, epochs=25, latent_dim=16,
    hidden=(64, 64), budget=6, n_mc=8, beam_widths=(1, 5, 10),
    n_random_allocations=8,
)


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default="runs/stage0")
    ap.add_argument("--quick", action="store_true", help="small, fast settings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (PLAN.md 15 P1): the six "
                         "criteria, the calibration scatter and the figures")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--experiment", type=str, default="E0")
    ap.add_argument("--ckpt-id-ablation", action="store_true",
                    help="PLAN.md 12.1: enable the checkpoint-ID embedding. Off by "
                         "default because PLAN.md 4.2 forbids relying on it in the "
                         "main result.")
    ap.add_argument("--val-split-by", type=str, default=None,
                    choices=("checkpoint", "context"),
                    help="held-out checkpoints (default, PLAN.md 4.2) or contexts")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.quick:
        cfg.update(QUICK)
    if args.config:
        file_cfg = load_config(args.config)
        cfg.update(file_cfg.get("stage0", file_cfg))
    if args.ckpt_id_ablation:
        cfg["ckpt_id_ablation"] = True
    if args.val_split_by:
        cfg["val_split_by"] = args.val_split_by
    cfg["seed"] = args.seed

    seeds = SeedBundle(cfg["seed"])
    set_seed(seeds["latent"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = run_provenance({"experiment": args.experiment})
    report: dict = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                    "seeds": seeds.as_dict(),
                    "provenance": prov}
    say = lambda m: print(f"[stage0] {m}", flush=True)  # noqa: E731

    # Opened before the data model trains so the per-epoch losses and the final
    # criteria share one run; `--quick` is tagged so a smoke run can never be
    # mistaken for a measurement in the wandb runs table (PLAN.md 17).
    log = make_logger(
        enabled=bool(args.wandb),
        project=args.wandb_project,
        name=f"{args.experiment}-stage0-s{cfg['seed']}"
             + ("-quick" if args.quick else "")
             + ("-ckptid" if cfg.get("ckpt_id_ablation") else ""),
        group=args.wandb_group or f"{args.experiment}-stage0",
        job_type="quick" if args.quick else "full",
        config={**{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                **{f"prov/{k}": v for k, v in prov.items()
                   if isinstance(v, (str, int, float, bool))}},
        tags=("synthetic", args.experiment,
              "quick" if args.quick else "full",
              "ckpt_id_ablation" if cfg.get("ckpt_id_ablation") else "no_ckpt_id"),
    )
    log.define_steps({"datamodel/epoch": None, "datamodel/*": "datamodel/epoch"})
    if args.wandb:
        say(f"wandb: {log.url or 'init failed; continuing without logging'}")

    # ---- 1. world, incomplete initial dataset, fixed evaluation set -------
    world = SyntheticWorld(SyntheticConfig(seed=seeds["env"]))
    rng_env = seeds.rng("env")
    regions = world.default_initial_regions()
    store = world.build_store(cfg["n_samples"], rng_env, regions=regions)
    eval_m = world.evaluation_metadata(cfg["n_eval"], np.random.default_rng(seeds["env"] + 555))
    vo, va, _ = world.generate_chunks(eval_m, np.random.default_rng(seeds["env"] + 556))
    vo, va = torch.from_numpy(vo), torch.from_numpy(va)
    say(f"world: {len(store)} initial chunks from {len(regions)} modes, {len(eval_m)} eval chunks")

    # ---- 2. policy checkpoints ------------------------------------------
    policy, ckpts = train_bc(
        store, vo, va,
        BCTrainConfig(steps=cfg["policy_steps"], snapshot_every=cfg["policy_snapshot_every"],
                      n_restarts=cfg["policy_restarts"]),
        seed=seeds["policy"],
    )
    report["policy"] = {"eval": evaluate_bc(policy, vo, va), **ckpts.diversity_report()}
    say(f"policy: {len(ckpts)} checkpoints, val_loss={report['policy']['eval']['val_loss']:.4f}")

    # ---- 3. multi-context supervision -----------------------------------
    task = BCSupervisionTask(policy, store, vo, va)
    records, gen_report = generate_context_records(
        task, ckpts,
        LeaveOneOutEstimator(lr=cfg["label_lr"], n_steps=cfg["label_steps"]),
        len(store),
        ContextGenConfig(contexts_per_sample=cfg["contexts_per_sample"],
                         batch_size=cfg["context_batch_size"], seed=seeds["context"],
                         calibration_fraction=0.1),
        calibration_estimator=GradientAlignmentEstimator("dot", lr=cfg["label_lr"]),
    )
    ds = ContextDataset(store, records)
    report["supervision"] = {
        "coverage": ds.coverage_report(min_contexts=20),
        "cheap_vs_expensive_calibration": gen_report.calibration,
        "n_rejected_compositions": gen_report.n_rejected,
    }
    say(f"supervision: {len(records)} contexts, "
        f"{report['supervision']['coverage']['contexts_per_sample_mean']:.1f} per sample")

    # ---- 4. data model --------------------------------------------------
    # PLAN.md 4.2 / 15 P0.4: held-out CHECKPOINTS, the honest policy-transfer
    # test, rather than held-out batch compositions
    train_ds, val_ds = ds.split(
        0.2, seed=seeds["context"], by=cfg.get("val_split_by", "checkpoint"))
    table = EffectProfileTable(train_ds.records, len(store), min_shared=2)
    report["effect_table"] = table.report()

    # PLAN.md 15 P0.4: the main model carries no checkpoint-ID embedding
    n_ckpt_vocab = train_ds.n_checkpoints if cfg.get("ckpt_id_ablation") else 0

    def build(readout="contextual", utility="deepsets", **kw):
        return LDVADataModel(LDVAConfig.build(
            obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
            meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
            n_checkpoints=n_ckpt_vocab, latent_dim=cfg["latent_dim"],
            hidden=tuple(cfg["hidden"]), readout_kind=readout, utility_kind=utility, **kw))

    tcfg = TrainConfig(epochs=cfg["epochs"], eval_every=max(cfg["epochs"] // 4, 1),
                       seed=seeds["latent"], weights=LossWeights(1.0, 1.0, 0.1, 0.01),
                       # attach to the run opened at the top of main(), so the
                       # per-epoch losses and the final criteria land in ONE run
                       wandb=log.active, wandb_attach=True,
                       wandb_prefix="datamodel/")
    model = build()
    model.set_dataset_context(np.zeros((8, cfg["latent_dim"])))
    model, hist = train_datamodel(model, train_ds, val_ds, tcfg, table,
                                  save_path=out_dir / "datamodel.pt")
    final = {k[4:]: v for k, v in hist[-1].items() if k.startswith("val/")}
    report["datamodel"] = {"final_val": final, "history": hist}
    say(f"datamodel: effect spearman={final['effect_spearman']:.3f}, "
        f"gain spearman={final['gain_spearman']:.3f} "
        f"(within-ckpt r2={final.get('gain_within_r2', float('nan')):.3f}), "
        f"beats scalar baseline by {final['effect_gain_over_scalar']:.2f}x")

    # ---- 4b. ablations for success criteria 1 and 2 ----------------------
    say("ablations: scalar readout and additive utility")
    abl = {}
    for name, kw in [("scalar_readout", dict(readout="scalar")),
                     ("additive_utility", dict(utility="additive"))]:
        m = build(**kw)
        m.set_dataset_context(np.zeros((8, cfg["latent_dim"])))
        m, h = train_datamodel(m, train_ds, val_ds, tcfg, table)
        abl[name] = {k[4:]: v for k, v in h[-1].items() if k.startswith("val/")}
    report["ablations"] = abl

    # ---- 5. latent geometry gate ----------------------------------------
    # P0.4: the reference checkpoint's features and its vocabulary index travel
    # as one object, so they cannot describe different checkpoints
    ref_ckpt = ckpts[len(ckpts) // 2]
    pctx = PolicyContextRef.from_checkpoint(
        ref_ckpt, train_ds.checkpoint_ids,
        use_ckpt_id=bool(cfg.get("ckpt_id_ablation")))
    report["policy_context"] = pctx.to_dict()
    z_all = model.encode_store(store, pctx.features, ckpt_index=pctx.ckpt_index)
    model.set_dataset_context(z_all)
    geo = latent_geometry_report(z_all, records, table, n_clusters=cfg["n_clusters"],
                                 gate=GeometryGate(), seed=seeds["latent"])
    report["latent_geometry"] = geo
    say(f"geometry gate: {'PASS' if geo['gate_passed'] else 'CHECK'} -> {geo['checks']}")
    say(f"gain signal: {100*geo['gain_signal']['gain_within_group_share']:.1f}% of batch-gain "
        f"variance is within-checkpoint (composition-dependent)")

    # ---- 6. clusters and directions -------------------------------------
    clustering = LatentClustering(ClusteringConfig(n_clusters=cfg["n_clusters"], seed=seeds["latent"]))
    clusters = clustering.fit(z_all, store.metadata)
    gen = DirectionGenerator(DirectionConfig(r_max=cfg["r_max"], delta_scale=cfg["delta_scale"],
                                             seed=seeds["acquisition"]))
    directions = gen.generate(clusters, z_all)
    report["clustering"] = clustering.report()
    report["directions_pre_actionability"] = {**directions_report(directions), **gen.report()}
    if not directions:
        raise RuntimeError("no candidate directions survived the filters; relax DirectionConfig")

    # PLAN.md 7's fourth filter: a latent direction is only a candidate if
    # some feasible metadata change can actually produce it. The mapper has to
    # be fitted first, so this happens here rather than inside the generator.
    mapper = MetadataMapper(world.metadata_spec, MetadataMapperConfig(seed=seeds["acquisition"]))
    mapper.fit(z_all, store.metadata, clusters)
    report["metadata_mapper"] = mapper.report()
    directions, act_report = filter_actionable_directions(
        directions, mapper, store.metadata,
        ActionabilityConfig(min_achievable_cosine=cfg["min_achievable_cosine"],
                            min_reachability_cosine=cfg["min_reachability"]),
    )
    report["actionability"] = act_report
    report["directions"] = directions_report(directions)
    say(f"acquisition: {len(clusters)} domains -> {act_report['n_input']} directions, "
        f"{act_report['n_actionable']} metadata-actionable "
        f"(survival {100*act_report['survival_rate']:.0f}%, "
        f"mean achievable cosine {act_report['mean_achievable_kept']:.2f})")
    if act_report["F4_warning"]:
        say("WARNING: few directions are metadata-actionable -> PLAN.md 19/F4")

    # ---- 7. planners -----------------------------------------------------
    sampler = LatentSampler(clusters, LatentSamplerConfig(sigma=0.3, seed=seeds["acquisition"]))
    budget = BudgetSpec.from_directions(directions, budget=cfg["budget"])
    objective = AllocationObjective(model, sampler, directions, budget,
                                    policy_context=pctx,
                                    cfg=ObjectiveConfig(n_mc=cfg["n_mc"], seed=seeds["acquisition"]))
    space = n_allocations(len(directions), cfg["budget"])
    solvers = {}
    if space <= cfg["exact_max_allocations"]:
        solvers["exact"] = exact_search(objective, max_allocations=cfg["exact_max_allocations"])
    solvers["greedy"] = greedy_search(objective)
    for h in cfg["beam_widths"]:
        solvers[f"beam_{h}"] = beam_search(objective, beam_width=h)
    report["planners"] = {k: v.as_dict() for k, v in solvers.items()}
    report["planners"]["search_space_size"] = space
    report["objective_stats"] = objective.stats()
    for k, v in solvers.items():
        say(f"  {k:10s} V_hat={v.best_value:+.4f} evals={v.n_evaluations:6d} alloc={v.best_allocation.tolist()}")

    # Criterion 4 is "beam matches exact *on small problems*". When the full
    # candidate set is too large to enumerate, build a small sub-problem rather
    # than skip the check: the top-scoring directions under a reduced budget,
    # sized so enumeration is affordable.
    small = _small_problem_check(
        objective, model, sampler, directions, pctx,
        cfg, seeds, say) if "exact" not in solvers else None
    if small is not None:
        report["small_problem_exact_check"] = small

    # ---- 8. baselines ----------------------------------------------------
    suite = BaselineSuite(z_support=z_all, n_draw=32, seed=seeds["acquisition"])
    baselines = suite.run(objective, model)
    report["baselines"] = {k: v.as_dict() for k, v in baselines.items()}

    # ---- 9. metadata control (criterion 6) -------------------------------
    rng_acq = seeds.rng("acquisition")
    moves = []
    for d in directions:
        plan = mapper.plan_direction(d, 4, store.metadata, rng=rng_acq)
        moves.append(measure_realized_latent_movement(
            model, world, plan, d, ref_ckpt.features, rng_acq, n_per_anchor=24))
    cos = np.array([m["direction_cosine"] for m in moves])
    # Is the movement SPECIFIC to the direction asked for? A raw cosine is not
    # interpretable alone: the encoder collapses the 32-dimensional latent
    # space to 1.0-2.2 effective dimensions, so every candidate direction
    # points into the same narrow subspace and any displacement aligns with
    # any direction. Measured over 8 seeds, a cosine of +0.910 sat only 1.3
    # standard deviations above requesting a *different* direction, and no
    # cell of 16 reached 2. See docs/E0_criteria_resolution.md.
    spec = direction_specificity(
        {m["direction_id"]: np.asarray(m["realized_delta"]) for m in moves
         if "realized_delta" in m},
        {d.direction_id: np.asarray(d.vector) for d in directions},
    )
    report["direction_specificity"] = spec
    report["latent_participation_ratio"] = participation_ratio(z_all)
    report["metadata_control"] = {
        "per_direction": moves,
        "direction_cosine_mean": float(cos.mean()),
        "direction_cosine_median": float(np.median(cos)),
        "frac_directions_positive": float((cos > 0).mean()),
        "achievable_cosine_mean": float(np.mean([m["achievable_cosine"] for m in moves])),
        "reachability_cosine_mean": float(np.mean([m["reachability_cosine"] for m in moves])),
        "jacobian_r2_heldout_mean": float(np.mean([m["jacobian_r2_heldout"] for m in moves])),
    }
    say(f"metadata control: realized cosine mean={cos.mean():+.3f}, "
        f"{100*(cos>0).mean():.0f}% of directions move the right way")
    say(f"  direction specificity: gap={spec.get('gap_mean', float('nan')):+.3f}"
        f"+/-{spec.get('gap_sem', float('nan')):.3f} vs the other candidates "
        f"(>=0.3 = distinguishable), "
        f"{100*spec.get('frac_directions_gap_positive', 0):.0f}% of directions positive; "
        f"latent effective dim={report['latent_participation_ratio']:.2f} "
        f"of {z_all.shape[1]}")

    # ---- 10. predicted vs realized gain (criterion 5) --------------------
    oracle = SyntheticAcquisitionOracle(
        world, policy, vo, va,
        OracleConfig(lr=cfg["label_lr"], n_steps=cfg["label_steps"], n_repeats=3,
                     seed=seeds["acquisition"]),
    )
    # The candidate set is where this criterion used to go wrong. Scoring only
    # the allocations the solvers *chose* measures the correlation inside a
    # narrow near-optimal band and duplicates heavily: a 24-cell sweep
    # (docs/E0_diagnosis.md) found one seed where 9 of 15 candidates shared one
    # allocation, so nine records carried one predicted value against realized
    # gains from -26.6 to +2.6 - a spread 15x the entire range of predicted
    # values, which then set the sign of Spearman. De-duplicating and adding
    # allocations that span the simplex turned that seed from -0.569 to +0.388
    # at the SAME step length, so this is a measurement fix, not a tuning one.
    solver_allocs = {k: v.best_allocation for k, v in solvers.items()}
    solver_allocs.update({k: v.best_allocation for k, v in baselines.items()})
    chosen_keys = {tuple(int(x) for x in a) for a in solver_allocs.values()}
    candidates = dict(solver_allocs)
    for i, a in enumerate(candidate_allocations(
            len(directions), cfg["budget"], cfg["n_random_allocations"],
            rng_acq, exclude=chosen_keys)):
        # one-hot allocations come first, so the label says which is which
        concentrated = int(np.count_nonzero(a)) == 1
        candidates[("concentrated_" if concentrated else "random_") + str(i)] = (
            np.asarray(a, dtype=np.int64))

    calib = []
    for name, alloc in candidates.items():
        plans = mapper.plan_allocation(directions, alloc, store.metadata, rng=rng_acq)
        realized = oracle.realized_allocation_gain(
            plans, ref_ckpt.flat_params, ref_ckpt.features, rng_acq)
        calib.append(CalibrationRecord(
            allocation=alloc, predicted=objective.value(alloc),
            realized=realized["realized_gain"], cost=budget.cost_of(alloc), label=name,
            extra={"source": ("concentrated" if name.startswith("concentrated_")
                              else "random" if name.startswith("random_")
                              else "planner")}))
    n_before = len(calib)
    calib = deduplicate(calib)

    # the planner-only subset reproduces the OLD measurement, kept so the
    # change to this criterion is auditable rather than asserted
    planner_only = [c for c in calib if c.extra.get("source") == "planner"]
    by_source = {}
    for src in ("planner", "concentrated", "random"):
        sub = [c for c in calib if c.extra.get("source") == src]
        if len(sub) >= 3:
            by_source[src] = calibration_report(sub, top_k=3)
    report["calibration"] = {
        "report": calibration_report(calib, top_k=3),
        "planner_only_report": calibration_report(planner_only, top_k=3),
        "by_source": by_source,
        "n_candidates_before_dedup": n_before,
        "n_distinct": len(calib),
        "n_random_added": cfg["n_random_allocations"],
        "measurement": "de-duplicated allocations, planner picks plus random "
                       "allocations spanning the simplex (docs/E0_diagnosis.md)",
        "records": [{"label": c.label, "allocation": c.allocation.tolist(),
                     "predicted": c.predicted, "realized": c.realized,
                     "source": c.extra.get("source")} for c in calib],
    }
    r = report["calibration"]["report"]
    po = report["calibration"]["planner_only_report"].get("spearman", float("nan"))
    say(f"calibration: spearman(predicted, realized)={r['spearman']:+.3f} over "
        f"{len(calib)} distinct compositions "
        f"(planner-only subset, the old measurement: {po:+.3f})")

    # ---- 11. success criteria (PLAN.md 14) ------------------------------
    crit = _success_criteria(report, solvers, calib, cfg)
    report["success_criteria"] = crit

    if not args.no_plots:
        pdir = out_dir / "figures"
        P.plot_latent_pca(z_all, clustering.labels_, pdir / "latent_pca.png")
        P.plot_contexts_per_sample(ds.contexts_per_sample(), 20, pdir / "coverage.png")
        P.plot_allocation({k: v.best_allocation for k, v in solvers.items()},
                          pdir / "allocations.png")
        P.plot_solver_comparison({k: v.best_value for k, v in solvers.items()},
                                 pdir / "solvers.png")
        P.plot_predicted_vs_realized([c.predicted for c in calib], [c.realized for c in calib],
                                     [c.label for c in calib], pdir / "calibration.png")
        P.plot_direction_alignment(cos, pdir / "direction_alignment.png")
        report["figures"] = str(pdir)

    save_json(report, out_dir / "stage0_report.json")
    _log_stage0(log, report, crit, calib)
    _print_criteria(crit, out_dir)
    return report


def _log_stage0(log, report: dict, crit: dict, calib: list) -> None:
    """Send the gate's verdict to wandb, not just the training curves.

    The six criteria *are* the Stage 0 result; the per-epoch data-model losses
    are only how it got there. Logging the criteria as a table plus
    pass/fail summary fields is what makes a seed sweep readable in the runs
    table - `criteria_passed` and `criterion_5_passed` become sortable columns,
    so a reproducible failure on one criterion is visible at a glance instead
    of requiring six JSON files to be opened.
    """
    if not log.active:
        return
    log.table(
        "success_criteria",
        ["criterion", "passed", "rule"],
        [[k, bool(v["passed"]), str(v.get("rule", ""))] for k, v in crit.items()],
    )
    log.table(
        "calibration",
        ["composition", "predicted", "realized"],
        [[str(c.label), float(c.predicted), float(c.realized)] for c in calib],
    )
    d = report.get("datamodel", {}).get("final_val", {})
    g = report.get("latent_geometry", {})
    cal = report.get("calibration", {}).get("report", {})
    mc = report.get("metadata_control", {})
    flat = {
        "criteria_passed": sum(1 for v in crit.values() if v["passed"]),
        "criteria_total": len(crit),
        "all_criteria_passed": all(v["passed"] for v in crit.values()),
        "effect_spearman": d.get("effect_spearman", float("nan")),
        "effect_gain_over_scalar": d.get("effect_gain_over_scalar", float("nan")),
        "gain_within_r2": d.get("gain_within_r2", float("nan")),
        "neighbor_consistency_ratio": g.get("neighbor", {}).get(
            "neighbor_consistency_ratio", float("nan")),
        "additive_r2_heldout": g.get("additivity", {}).get(
            "additive_r2_heldout", float("nan")),
        "composition_signal_share": g.get("gain_signal", {}).get(
            "gain_within_group_share", float("nan")),
        "calibration_spearman": cal.get("spearman", float("nan")),
        "direction_cosine_mean": mc.get("direction_cosine_mean", float("nan")),
        "frac_directions_positive": mc.get("frac_directions_positive", float("nan")),
    }
    for i, (k, v) in enumerate(crit.items(), start=1):
        flat[f"criterion_{i}_passed"] = bool(v["passed"])
    log.log(flat)
    log.summary(flat)
    fig = report.get("figures")
    if fig:
        for png in sorted(Path(fig).glob("*.png")):
            log.image(png.stem, png)
    log.finish()


def _beam_over_greedy(solvers: dict) -> float:
    """Relative improvement of the best beam over greedy on the full set."""
    beams = [v.best_value for k, v in solvers.items()
             if k.startswith("beam_") and k != "beam_1"]
    if not beams or "greedy" not in solvers:
        return float("nan")
    g = solvers["greedy"].best_value
    return float((max(beams) - g) / max(abs(g), 1e-12))


def _small_problem_check(objective, model, sampler, directions, pctx, cfg, seeds, say):
    """Exact vs beam on a deliberately small sub-problem (PLAN.md 14 crit. 4).

    Directions are ranked by their own one-unit marginal utility and the top few
    kept, with the budget reduced until `C(B + A - 1, A - 1)` is affordable.
    Keeping the *best* directions makes the sub-problem the one the planner
    would actually face, rather than an arbitrary slice.
    """
    base = objective.value(objective.zero_allocation())
    scores = []
    for a in range(len(directions)):
        alloc = objective.zero_allocation()
        alloc[a] = 1
        scores.append(objective.value(alloc) - base)
    order = np.argsort(scores)[::-1]

    for n_dir in (4, 3, 2):
        for budget in (cfg["budget"], 6, 4):
            if n_dir > len(directions):
                continue
            if n_allocations(n_dir, budget) <= 20_000:
                subset = [directions[int(i)] for i in order[:n_dir]]
                sub_budget = BudgetSpec.from_directions(subset, budget=budget)
                sub_obj = AllocationObjective(
                    model, sampler, subset, sub_budget,
                    policy_context=pctx,
                    cfg=ObjectiveConfig(n_mc=cfg["n_mc"], seed=seeds["acquisition"]))
                ex = exact_search(sub_obj)
                gr = greedy_search(sub_obj)
                beams = {h: beam_search(sub_obj, beam_width=h) for h in cfg["beam_widths"]}
                best_beam = max(v.best_value for h, v in beams.items() if h != 1)
                gap = float((ex.best_value - best_beam) / max(abs(ex.best_value), 1e-12))
                say(f"  small problem (A={n_dir}, B={budget}, space="
                    f"{n_allocations(n_dir, budget)}): exact={ex.best_value:+.4f} "
                    f"best_beam={best_beam:+.4f} greedy={gr.best_value:+.4f} gap={gap:+.4%}")
                return {
                    "n_directions": n_dir,
                    "budget": budget,
                    "search_space_size": n_allocations(n_dir, budget),
                    "exact": ex.as_dict(),
                    "greedy": gr.as_dict(),
                    "beams": {str(h): v.as_dict() for h, v in beams.items()},
                    "best_beam_value": best_beam,
                    "relative_gap": gap,
                    "greedy_gap": float(
                        (ex.best_value - gr.best_value) / max(abs(ex.best_value), 1e-12)),
                }
    return None


def _success_criteria(report: dict, solvers: dict, calib: list, cfg: dict) -> dict:
    """The six criteria of PLAN.md 14, each with the number behind it."""
    spec_gap = report.get("direction_specificity", {}).get("gap_mean", float("nan"))
    final = report["datamodel"]["final_val"]
    abl = report["ablations"]
    geo = report["latent_geometry"]

    # 1. contextual effect prediction beats scalar / constant baselines
    c1_val = final.get("effect_gain_over_scalar", float("nan"))
    c1_abl = abl["scalar_readout"].get("effect_mse", float("nan")) / max(final["effect_mse"], 1e-12)

    # 2. non-additive set utility beats additive utility prediction.
    # Judged on the WITHIN-CHECKPOINT gain: ~98% of raw batch-gain variance is
    # between-checkpoint, so a plain MSE compares checkpoint identification and
    # both heads score ~0.98 regardless of any set structure.
    c2 = abl["additive_utility"].get("gain_within_mse", float("nan")) / max(
        final.get("gain_within_mse", float("nan")), 1e-12
    )
    c2_raw = abl["additive_utility"].get("gain_mse", float("nan")) / max(
        final["gain_mse"], 1e-12
    )

    # 3. latent neighbours have similar effect profiles
    c3 = geo["neighbor"].get("neighbor_consistency_ratio", float("nan"))

    # 4. beam search approximately matches exact search on small problems
    c4 = float("nan")
    c4_source = "unavailable"
    if "exact" in solvers:
        ex = solvers["exact"].best_value
        beams = [v.best_value for k, v in solvers.items()
                 if k.startswith("beam_") and k != "beam_1"]
        if beams and abs(ex) > 1e-12:
            c4 = float((ex - max(beams)) / abs(ex))
            c4_source = "full candidate set"
    elif "small_problem_exact_check" in report:
        c4 = report["small_problem_exact_check"]["relative_gap"]
        c4_source = (
            f"sub-problem A={report['small_problem_exact_check']['n_directions']}, "
            f"B={report['small_problem_exact_check']['budget']}")

    # 5. predicted high-value batches realize higher gain
    c5 = report["calibration"]["report"].get("spearman", float("nan"))

    # 6. metadata interventions move samples along the intended direction
    c6 = report["metadata_control"]["direction_cosine_mean"]

    return {
        "1_contextual_beats_scalar": {
            "mse_ratio_vs_hindsight_scalar": c1_val,
            "mse_ratio_vs_scalar_readout_ablation": c1_abl,
            "passed": bool(np.isfinite(c1_val) and c1_val > 1.0),
            "rule": "scalar-baseline MSE / contextual MSE > 1",
        },
        "2_set_utility_beats_additive": {
            "within_ckpt_gain_mse_ratio_additive_over_set": c2,
            "raw_gain_mse_ratio_additive_over_set": c2_raw,
            "set_head_gain_within_r2": final.get("gain_within_r2"),
            "additive_head_gain_within_r2": abl["additive_utility"].get("gain_within_r2"),
            "data_additive_r2_heldout": geo["additivity"].get("additive_r2_heldout"),
            "composition_signal_share": geo["gain_signal"].get("gain_within_group_share"),
            "passed": bool(np.isfinite(c2) and c2 > 1.0),
            "rule": "additive-head / set-head MSE on the WITHIN-checkpoint gain > 1",
        },
        "3_latent_neighbors_similar_effects": {
            "neighbor_consistency_ratio": c3,
            "frac_probes_consistent": geo["neighbor"].get("frac_probes_consistent"),
            "passed": bool(np.isfinite(c3) and c3 < 1.0),
            "rule": "neighbour effect distance / random effect distance < 1",
        },
        "4_beam_matches_exact": {
            "relative_gap": c4,
            "measured_on": c4_source,
            "greedy_gap_same_problem": (
                report.get("small_problem_exact_check", {}).get("greedy_gap")
                if "exact" not in solvers
                else None
            ),
            # greedy often ties exact on a small sub-problem; the interesting
            # comparison is on the full candidate set, where complementarity
            # has enough directions to matter
            "beam_over_greedy_full_set": _beam_over_greedy(solvers),
            "passed": bool(np.isfinite(c4) and abs(c4) <= 0.02),
            "rule": "|V_exact - V_beam| / |V_exact| <= 2%",
            "note": "when the full candidate set is too large to enumerate, this "
                    "is measured on a small sub-problem of the top directions",
        },
        "5_predicted_gain_realizes": {
            "spearman_predicted_vs_realized": c5,
            "picked_the_best": report["calibration"]["report"].get("picked_the_best"),
            "regret_normalized": report["calibration"]["report"].get("regret_normalized"),
            "passed": bool(np.isfinite(c5) and c5 > 0.3),
            "rule": "spearman(predicted, realized) > 0.3",
        },
        # The rule changed from "cosine > 0.3" to a specificity z-score, and
        # the reason is measured: with the latent space collapsed to ~1
        # effective dimension the 0.3 threshold sits BELOW chance, so a model
        # with a degenerate representation passed while a richer one failed -
        # exactly backwards. The z-score asks whether the realized movement
        # matches the direction that was requested better than one that was
        # not, which is the claim the acquisition loop actually relies on.
        "6_metadata_moves_latents": {
            "specificity_gap": spec_gap,
            "specificity_gap_sem": report.get("direction_specificity", {}).get(
                "gap_sem", float("nan")),
            "frac_directions_gap_positive": report.get("direction_specificity", {}).get(
                "frac_directions_gap_positive", float("nan")),
            "latent_effective_dim": report.get("latent_participation_ratio", float("nan")),
            "direction_cosine_mean": c6,
            "null_abs_mean": report.get("direction_specificity", {}).get(
                "null_abs_mean", float("nan")),
            "frac_directions_positive": report["metadata_control"]["frac_directions_positive"],
            "reachability_cosine_mean": report["metadata_control"]["reachability_cosine_mean"],
            "jacobian_r2_heldout_mean": report["metadata_control"]["jacobian_r2_heldout_mean"],
            "passed": bool(np.isfinite(spec_gap) and spec_gap >= 0.3),
            "rule": "mean gap cos(desired, realized) - mean|cos(desired, other "
                    "candidates)| >= 0.3. A raw cosine is uninterpretable when "
                    "the latent space is collapsed, and dividing by the null's "
                    "spread (a z-score) breaks both when the directions are "
                    "near-identical (the spread vanishes and z explodes) and "
                    "when they are orthogonal (the spread also vanishes, but "
                    "that is the best case). 0.3 is ~1.5x the measured "
                    "replicate noise of the realized cosine (~0.2).",
        },
    }


def _print_criteria(crit: dict, out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("STAGE 0 SUCCESS CRITERIA (PLAN.md 14)")
    print("=" * 78)
    n_pass = 0
    for key, c in crit.items():
        ok = c["passed"]
        n_pass += int(ok)
        nums = ", ".join(
            f"{k}={v:+.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in c.items()
            if k not in ("passed", "rule", "note") and v is not None
        )
        print(f"  [{'PASS' if ok else 'FAIL'}] {key}")
        print(f"         rule: {c['rule']}")
        print(f"         {nums}")
    print("-" * 78)
    print(f"  {n_pass}/6 criteria passed")
    print(f"  report: {out_dir / 'stage0_report.json'}")
    if n_pass < 6:
        print("  PLAN.md 14: do not move to robotics until Stage 0 passes.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
