"""The LDVA pipeline on any environment (PLAN.md 15, 14, 11).

`run_stage0.py` is the synthetic gate and keeps its ground-truth oracle. This
script is the same pipeline driven entirely through `EnvAdapter`, so it runs on
a real simulator:

    python experiments/run_pipeline.py --env dmc --task reacher-easy
    python experiments/run_pipeline.py --env metaworld
    python experiments/run_pipeline.py --env synthetic --quick

It evaluates the five criteria of PLAN.md 14 that need no privileged access to
the generative process (1, 2, 4, 5, 6) - every one of them only needs the
adapter's `collect`, which is why they transfer from the synthetic world to a
simulator unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
)
from ldva.analysis.acquisition_oracle import AcquisitionOracle, OracleConfig  # noqa: E402
from ldva.analysis.direction_validation import validate_all_directions  # noqa: E402
from ldva.analysis.latent_geometry import GeometryGate, latent_geometry_report  # noqa: E402
from ldva.data.context_dataset import ContextDataset  # noqa: E402
from ldva.data.effect_profiles import EffectProfileTable  # noqa: E402
from ldva.envs import get_adapter  # noqa: E402
from ldva.models.datamodel import LDVAConfig, LDVADataModel  # noqa: E402
from ldva.policy.checkpoints import PolicyContextRef  # noqa: E402
from ldva.policy.evaluate import evaluate_bc  # noqa: E402
from ldva.policy.train import BCTrainConfig, train_bc  # noqa: E402
from ldva.supervision.bc_task import BCSupervisionTask  # noqa: E402
from ldva.supervision.generate_context_records import (  # noqa: E402
    ContextGenConfig,
    generate_context_records,
)
from ldva.supervision.leave_one_out import LeaveOneOutEstimator  # noqa: E402
from ldva.training.losses import LossWeights  # noqa: E402
from ldva.training.train_datamodel import TrainConfig, train_datamodel  # noqa: E402
from ldva.utils import SeedBundle, save_json, set_seed  # noqa: E402

DEFAULTS = dict(
    n_requests=120, n_eval=300, policy_steps=200, policy_snapshot_every=40,
    policy_restarts=3, contexts_per_sample=20, context_batch_size=8,
    label_lr=0.3, label_steps=4, latent_dim=32, hidden=(128, 128), epochs=50,
    n_clusters=4, r_max=2, delta_scale=0.4, min_achievable_cosine=0.9,
    budget=6, n_mc=16, beam_widths=(1, 5, 10), exact_max_allocations=100_000,
    n_anchors_per_direction=3, n_per_anchor=6,
)
QUICK = dict(
    n_requests=50, n_eval=150, policy_steps=120, policy_snapshot_every=40,
    policy_restarts=2, contexts_per_sample=10, epochs=20, latent_dim=16,
    hidden=(64, 64), budget=4, n_mc=8, beam_widths=(1, 5),
    n_anchors_per_direction=2, n_per_anchor=4,
)

#: per-environment adapter kwargs
ENV_KW = {
    "synthetic": {},
    "dmc": {"task": "reacher-easy", "chunk_len": 16, "max_steps": 120,
            "max_chunks_per_episode": 3},
    "metaworld": {"chunk_len": 16, "max_steps": 160, "max_chunks_per_episode": 3},
}


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="dmc", choices=sorted(ENV_KW))
    ap.add_argument("--task", default=None, help="DMC task, e.g. reacher-easy")
    ap.add_argument("--tasks", default=None, help="MetaWorld tasks, comma separated")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.quick:
        cfg.update(QUICK)
    seeds = SeedBundle(args.seed)
    set_seed(seeds["latent"])
    out_dir = Path(args.out or f"runs/pipeline_{args.env}")
    out_dir.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(f"[{args.env}] {m}", flush=True)  # noqa: E731

    # ---- 1. environment ---------------------------------------------------
    kw = dict(ENV_KW[args.env])
    if args.task and args.env == "dmc":
        kw["task"] = args.task
    if args.tasks and args.env == "metaworld":
        kw["tasks"] = tuple(t.strip() for t in args.tasks.split(","))
    kw.setdefault("seed", seeds["env"])
    adapter = get_adapter(args.env, **kw)
    report: dict = {
        "env": args.env,
        "adapter": adapter.report() if hasattr(adapter, "report") else {},
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "seeds": seeds.as_dict(),
    }

    rng_env = seeds.rng("env")
    store = adapter.initial_dataset(cfg["n_requests"], rng_env)
    vo, va = adapter.evaluation_set(cfg["n_eval"], np.random.default_rng(seeds["env"] + 555))
    say(f"dataset: {cfg['n_requests']} requests -> {len(store)} chunks "
        f"(obs {store.obs_dim}, act {store.act_dim}, chunk {store.chunk_len}); "
        f"{len(vo)} eval chunks")
    say(f"metadata: {adapter.metadata_spec.names} "
        f"controllable={adapter.metadata_spec.controllable_mask.tolist()}")

    # ---- 2. policy checkpoints -------------------------------------------
    hidden = tuple(adapter.policy_defaults().get("hidden", ()))
    policy, ckpts = train_bc(
        store, vo, va,
        BCTrainConfig(steps=cfg["policy_steps"], snapshot_every=cfg["policy_snapshot_every"],
                      n_restarts=cfg["policy_restarts"]),
        hidden=hidden, seed=seeds["policy"])
    report["policy"] = {"hidden": list(hidden), "eval": evaluate_bc(policy, vo, va),
                        **ckpts.diversity_report()}
    say(f"policy: MLP{list(hidden)}, {len(ckpts)} checkpoints, "
        f"val_loss={report['policy']['eval']['val_loss']:.4f}")

    # ---- 3. multi-context supervision ------------------------------------
    task_obj = BCSupervisionTask(policy, store, vo, va)
    records, gen = generate_context_records(
        task_obj, ckpts,
        LeaveOneOutEstimator(lr=cfg["label_lr"], n_steps=cfg["label_steps"]),
        len(store),
        ContextGenConfig(contexts_per_sample=cfg["contexts_per_sample"],
                         batch_size=cfg["context_batch_size"], seed=seeds["context"]))
    ds = ContextDataset(store, records)
    report["supervision"] = {"coverage": ds.coverage_report(min_contexts=cfg["contexts_per_sample"]),
                             "target_met": gen.target_met}
    say(f"supervision: {len(records)} contexts, "
        f"{report['supervision']['coverage']['contexts_per_sample_mean']:.1f} per sample")

    # ---- 4. data model + additive ablation -------------------------------
    # PLAN.md 4.2 / 15 P0.4: hold out whole CHECKPOINTS. A context split only
    # tests transfer to an unseen batch composition; the claim that matters is
    # transfer to an unseen future policy.
    train_ds, val_ds = ds.split(
        0.2, seed=seeds["context"], by=cfg.get("val_split_by", "checkpoint"))
    table = EffectProfileTable(train_ds.records, len(store), min_shared=2)

    # PLAN.md 15 P0.4: no checkpoint-ID embedding in the main model. With one,
    # the policy context can memorize the checkpoints it saw and defines
    # nothing for an unseen future policy.
    n_ckpt_vocab = train_ds.n_checkpoints if cfg.get("ckpt_id_ablation") else 0

    def build(**kwm):
        return LDVADataModel(LDVAConfig.build(
            obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
            meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
            n_checkpoints=n_ckpt_vocab, latent_dim=cfg["latent_dim"],
            hidden=tuple(cfg["hidden"]), **kwm))

    tcfg = TrainConfig(epochs=cfg["epochs"], eval_every=max(cfg["epochs"] // 3, 1),
                       seed=seeds["latent"], weights=LossWeights(1.0, 1.0, 0.1, 0.01))
    model = build()
    model.set_dataset_context(np.zeros((8, cfg["latent_dim"])))
    model, hist = train_datamodel(model, train_ds, val_ds, tcfg, table,
                                  save_path=out_dir / "datamodel.pt")
    final = {k[4:]: v for k, v in hist[-1].items() if k.startswith("val/")}
    report["datamodel"] = {"final_val": final}
    nan = float("nan")
    say(f"datamodel: effect spearman={final.get('effect_spearman', nan):.3f}, "
        f"gain within-ckpt r2={final.get('gain_within_r2', nan):.3f}, "
        f"beats scalar by {final.get('effect_gain_over_scalar', nan):.2f}x")
    if not np.isfinite(final.get("effect_gain_over_scalar", nan)):
        # a degenerate effect MSE means the baseline ratio is undefined; say so
        # rather than letting a criterion silently read a missing number
        say("WARNING: effect_gain_over_scalar unavailable (degenerate effect MSE)")

    abl = {}
    for name, kwm in [("scalar_readout", dict(readout_kind="scalar")),
                      ("additive_utility", dict(utility_kind="additive"))]:
        m = build(**kwm)
        m.set_dataset_context(np.zeros((8, cfg["latent_dim"])))
        _, h = train_datamodel(m, train_ds, val_ds, tcfg, table)
        abl[name] = {k[4:]: v for k, v in h[-1].items() if k.startswith("val/")}
    report["ablations"] = abl

    # ---- 5. latent geometry ----------------------------------------------
    # P0.4: features and vocabulary index travel together so they cannot
    # describe different checkpoints
    ref = ckpts[len(ckpts) // 2]
    pctx = PolicyContextRef.from_checkpoint(
        ref, train_ds.checkpoint_ids, use_ckpt_id=bool(cfg.get("ckpt_id_ablation")))
    report["policy_context"] = pctx.to_dict()
    z_all = model.encode_store(store, pctx.features, ckpt_index=pctx.ckpt_index)
    model.set_dataset_context(z_all)
    geo = latent_geometry_report(z_all, records, table, n_clusters=cfg["n_clusters"],
                                 gate=GeometryGate(), seed=seeds["latent"])
    report["latent_geometry"] = geo
    say(f"geometry gate: {'PASS' if geo['gate_passed'] else 'CHECK'} {geo['checks']}")
    say(f"gain signal: {100*geo['gain_signal']['gain_within_group_share']:.1f}% "
        f"of batch-gain variance is composition-dependent")

    # ---- 6. clusters, directions, actionability --------------------------
    clustering = LatentClustering(ClusteringConfig(n_clusters=cfg["n_clusters"],
                                                   seed=seeds["latent"]))
    clusters = clustering.fit(z_all, store.metadata)
    directions = DirectionGenerator(
        DirectionConfig(r_max=cfg["r_max"], delta_scale=cfg["delta_scale"],
                        seed=seeds["acquisition"])).generate(clusters, z_all)
    if not directions:
        raise RuntimeError("no candidate directions survived the filters")
    mapper = MetadataMapper(adapter.metadata_spec,
                            MetadataMapperConfig(seed=seeds["acquisition"]))
    mapper.fit(z_all, store.metadata, clusters)
    directions, act = filter_actionable_directions(
        directions, mapper, store.metadata,
        ActionabilityConfig(min_achievable_cosine=cfg["min_achievable_cosine"]))
    report["clustering"] = clustering.report()
    report["directions"] = directions_report(directions)
    report["actionability"] = act
    report["metadata_mapper"] = mapper.report()
    say(f"acquisition: {len(clusters)} domains -> {act['n_input']} directions, "
        f"{act['n_actionable']} actionable (survival {100*act['survival_rate']:.0f}%)")

    # ---- 7. planners ------------------------------------------------------
    sampler = LatentSampler(clusters, LatentSamplerConfig(sigma=0.3, seed=seeds["acquisition"]))
    budget = BudgetSpec.from_directions(directions, budget=cfg["budget"])
    objective = AllocationObjective(model, sampler, directions, budget,
                                    policy_context=pctx,
                                    cfg=ObjectiveConfig(n_mc=cfg["n_mc"],
                                                        seed=seeds["acquisition"]))
    space = n_allocations(len(directions), cfg["budget"])
    solvers = {}
    if space <= cfg["exact_max_allocations"]:
        solvers["exact"] = exact_search(objective, max_allocations=cfg["exact_max_allocations"])
    solvers["greedy"] = greedy_search(objective)
    for h in cfg["beam_widths"]:
        solvers[f"beam_{h}"] = beam_search(objective, beam_width=h)
    report["planners"] = {k: v.as_dict() for k, v in solvers.items()}
    report["planners"]["search_space_size"] = space
    for k, v in solvers.items():
        say(f"  {k:9s} V_hat={v.best_value:+.4f} evals={v.n_evaluations:6d} "
            f"alloc={v.best_allocation.tolist()}")

    # ---- 8. metadata control on the real environment ---------------------
    rng_acq = seeds.rng("acquisition")
    ctl = validate_all_directions(
        model, adapter, mapper, directions, store.metadata, ref.features, rng_acq,
        n_anchors_per_direction=cfg["n_anchors_per_direction"],
        n_per_anchor=cfg["n_per_anchor"])
    report["metadata_control"] = ctl
    say(f"metadata control: realized cosine={ctl['direction_cosine_mean']:+.3f}, "
        f"{100*ctl['frac_directions_positive']:.0f}% of directions move correctly"
        + (f" (actionable only: {ctl['actionable_only_cosine_mean']:+.3f})"
           if ctl["actionable_only_cosine_mean"] is not None else ""))

    # ---- 9. predicted vs realized gain -----------------------------------
    oracle = AcquisitionOracle(adapter, policy, vo, va,
                               OracleConfig(lr=cfg["label_lr"], n_steps=cfg["label_steps"],
                                            n_repeats=1, seed=seeds["acquisition"]))
    calib = []
    for name, res in solvers.items():
        if not hasattr(res, "best_allocation"):
            continue
        plans = mapper.plan_allocation(directions, res.best_allocation, store.metadata,
                                       rng=rng_acq)
        realized = oracle.realized_allocation_gain(plans, ref.flat_params, ref.features,
                                                   rng_acq)
        calib.append(CalibrationRecord(
            allocation=res.best_allocation, predicted=objective.value(res.best_allocation),
            realized=realized["realized_gain"], cost=budget.cost_of(res.best_allocation),
            label=name))
    # a few random allocations widen the comparison set
    for i in range(6):
        alloc = np.zeros(len(directions), dtype=np.int64)
        for _ in range(cfg["budget"]):
            alloc[rng_acq.integers(0, len(directions))] += 1
        plans = mapper.plan_allocation(directions, alloc, store.metadata, rng=rng_acq)
        realized = oracle.realized_allocation_gain(plans, ref.flat_params, ref.features,
                                                   rng_acq)
        calib.append(CalibrationRecord(allocation=alloc, predicted=objective.value(alloc),
                                       realized=realized["realized_gain"],
                                       cost=budget.cost_of(alloc), label=f"random_{i}"))
    report["calibration"] = {
        "report": calibration_report(calib, top_k=3),
        "records": [{"label": c.label, "allocation": c.allocation.tolist(),
                     "predicted": c.predicted, "realized": c.realized} for c in calib],
    }
    say(f"calibration: spearman(predicted, realized)="
        f"{report['calibration']['report']['spearman']:+.3f} over {len(calib)} compositions")

    # ---- 10. criteria -----------------------------------------------------
    crit = _criteria(report, solvers)
    report["success_criteria"] = crit

    if not args.no_plots:
        pdir = out_dir / "figures"
        P.plot_latent_pca(z_all, clustering.labels_, pdir / "latent_pca.png")
        P.plot_contexts_per_sample(ds.contexts_per_sample(), cfg["contexts_per_sample"],
                                   pdir / "coverage.png")
        P.plot_solver_comparison({k: v.best_value for k, v in solvers.items()},
                                 pdir / "solvers.png")
        P.plot_predicted_vs_realized([c.predicted for c in calib],
                                     [c.realized for c in calib],
                                     [c.label for c in calib], pdir / "calibration.png")
        P.plot_direction_alignment(
            np.array([r["direction_cosine"] for r in ctl["per_direction"]]),
            pdir / "direction_alignment.png")
        report["figures"] = str(pdir)

    save_json(report, out_dir / "pipeline_report.json")
    _print(crit, args.env, out_dir)
    return report


def _criteria(report: dict, solvers: dict) -> dict:
    """The PLAN.md 14 criteria that need no privileged generative access."""
    final = report["datamodel"]["final_val"]
    abl = report["ablations"]
    geo = report["latent_geometry"]
    ctl = report["metadata_control"]

    c1 = float(final.get("effect_gain_over_scalar", float("nan")))
    c2 = abl["additive_utility"].get("gain_within_mse", float("nan")) / max(
        final.get("gain_within_mse", float("nan")), 1e-12)
    c3 = geo["neighbor"].get("neighbor_consistency_ratio", float("nan"))
    c4, c4_src = float("nan"), "unavailable"
    if "exact" in solvers:
        ex = solvers["exact"].best_value
        beams = [v.best_value for k, v in solvers.items()
                 if k.startswith("beam_") and k != "beam_1"]
        if beams and abs(ex) > 1e-12:
            c4, c4_src = float((ex - max(beams)) / abs(ex)), "full candidate set"
    c5 = report["calibration"]["report"].get("spearman", float("nan"))
    c6 = ctl["direction_cosine_mean"]

    return {
        "1_contextual_beats_scalar": {
            "value": c1, "passed": bool(np.isfinite(c1) and c1 > 1.0),
            "rule": "scalar-baseline MSE / contextual MSE > 1"},
        "2_set_utility_beats_additive": {
            "value": c2, "composition_signal_share":
                geo["gain_signal"].get("gain_within_group_share"),
            "passed": bool(np.isfinite(c2) and c2 > 1.0),
            "rule": "additive / set MSE on the within-checkpoint gain > 1"},
        "3_latent_neighbors_similar_effects": {
            "value": c3, "passed": bool(np.isfinite(c3) and c3 < 1.0),
            "rule": "neighbour / random effect distance < 1"},
        "4_beam_matches_exact": {
            "value": c4, "measured_on": c4_src,
            "passed": bool(np.isfinite(c4) and abs(c4) <= 0.02),
            "rule": "|V_exact - V_beam| / |V_exact| <= 2%"},
        "5_predicted_gain_realizes": {
            "value": c5, "passed": bool(np.isfinite(c5) and c5 > 0.3),
            "rule": "spearman(predicted, realized) > 0.3"},
        "6_metadata_moves_latents": {
            "value": c6, "actionable_only": ctl.get("actionable_only_cosine_mean"),
            "passed": bool(np.isfinite(c6) and c6 > 0.3),
            "rule": "mean cosine(desired, realized) > 0.3"},
    }


def _print(crit: dict, env: str, out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print(f"LDVA PIPELINE ON {env.upper()} - PLAN.md 14 criteria")
    print("=" * 78)
    n = 0
    for k, c in crit.items():
        n += int(c["passed"])
        v = c["value"]
        extra = ""
        if c.get("actionable_only") is not None:
            extra = f"  (actionable only: {c['actionable_only']:+.4f})"
        if c.get("measured_on") and c["measured_on"] != "full candidate set":
            extra = f"  ({c['measured_on']})"
        vs = f"{v:+.4f}" if isinstance(v, float) and np.isfinite(v) else "   n/a"
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {k}")
        print(f"         {vs}   rule: {c['rule']}{extra}")
    print("-" * 78)
    print(f"  {n}/{len(crit)} criteria passed")
    print(f"  report: {out_dir / 'pipeline_report.json'}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
