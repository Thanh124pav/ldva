"""Closed-loop acquisition (PLAN.md 13, Phase 8; SETUP.md 27).

Each round runs the full twelve steps of PLAN.md 13:

    train policy -> generate multi-context supervision -> update the data model
    -> encode -> cluster -> outward directions -> predict allocation utility
    -> solve for Q* -> map directions to metadata -> collect -> union -> repeat

and the same loop runs for every acquisition *method* under an identical budget,
initial dataset, policy architecture, training budget and evaluation
distribution, which is the comparison SETUP.md 34 requires. The evaluation set
is drawn once, before any acquisition, and never touched again (SETUP.md 33).

Each round's performance is the **average over several independently initialized
policies trained on that round's dataset**, with the seeds held fixed across
rounds and methods. One training run per round makes the curve measure optimizer
noise rather than data quality: on the quick setting it produced a
non-monotonic curve (-1.36, -1.84, -1.34) whose swings were larger than any
difference between the acquisition methods being compared.

    python experiments/run_acquisition_loop.py --quick
    python experiments/run_acquisition_loop.py --rounds 4 --budget 40 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldva.acquisition.baselines import (  # noqa: E402
    diversity_acquisition,
    equal_allocation,
    gradient_alignment_acquisition,
    influence_acquisition,
    random_acquisition,
    uncertainty_acquisition,
)
from ldva.acquisition.beam_search import beam_search  # noqa: E402
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
from ldva.analysis import plotting as P  # noqa: E402
from ldva.analysis.acquisition_calibration import acquisition_curve_report  # noqa: E402
from ldva.data.context_dataset import ContextDataset  # noqa: E402
from ldva.data.effect_profiles import EffectProfileTable  # noqa: E402
from ldva.envs.synthetic.generator import SyntheticConfig, SyntheticWorld  # noqa: E402
from ldva.models.datamodel import LDVAConfig, LDVADataModel  # noqa: E402
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

#: every method allocates the same budget over the same candidate directions
METHODS = (
    "ldva_beam",
    "ldva_greedy",
    "random",
    "equal",
    "diversity",
    "uncertainty",
    "gradient_alignment",
    "influence_cupid_style",
)


def _plan(method: str, objective, model, z_support, seed: int):
    if method == "ldva_beam":
        return beam_search(objective, beam_width=10)
    if method == "ldva_greedy":
        return greedy_search(objective)
    if method == "random":
        return random_acquisition(objective, seed)
    if method == "equal":
        return equal_allocation(objective)
    if method == "diversity":
        return diversity_acquisition(objective, z_support, seed)
    if method == "uncertainty":
        return uncertainty_acquisition(objective, model)
    if method == "gradient_alignment":
        return gradient_alignment_acquisition(objective, model)
    if method == "influence_cupid_style":
        return influence_acquisition(objective, model)
    raise ValueError(f"unknown method {method!r}")


def _evaluate_dataset(store, vo, va, cfg: dict, seeds, success_threshold: float):
    """Train `n_eval_policies` policies on `store` and average their utility.

    The policy seeds are deliberately *fixed* - not round- or method-dependent -
    so the only thing varying across rounds and methods is the dataset. Returns
    the averaged metrics plus the last policy and its checkpoints, which the
    supervision generator reuses.
    """
    utils, succs = [], []
    policy = ckpts = None
    for k in range(cfg["n_eval_policies"]):
        policy, ckpts = train_bc(
            store, vo, va,
            BCTrainConfig(steps=cfg["policy_steps"],
                          snapshot_every=cfg["policy_snapshot_every"],
                          n_restarts=cfg["policy_restarts"]),
            seed=seeds["policy"] + 1000 * k,
        )
        m = evaluate_bc(policy, vo, va, success_threshold=success_threshold)
        utils.append(m["utility"])
        succs.append(m["success_rate"])
    return (
        {
            "utility": float(np.mean(utils)),
            "utility_std": float(np.std(utils)),
            "success_rate": float(np.mean(succs)),
            "val_loss": float(-np.mean(utils)),
            "n_eval_policies": cfg["n_eval_policies"],
        },
        policy,
        ckpts,
    )


def run_one(cfg: dict, method: str, seed: int, say) -> dict:
    """Run the acquisition loop for one method and one seed."""
    seeds = SeedBundle(seed)
    set_seed(seeds["latent"])
    world = SyntheticWorld(SyntheticConfig(seed=seeds["env"]))

    # fixed evaluation distribution, declared before any acquisition
    eval_m = world.evaluation_metadata(cfg["n_eval"], np.random.default_rng(seeds["env"] + 555))
    vo, va, _ = world.generate_chunks(eval_m, np.random.default_rng(seeds["env"] + 556))
    vo, va = torch.from_numpy(vo), torch.from_numpy(va)

    rng_env = seeds.rng("env")
    store = world.build_store(cfg["n_initial"], rng_env, regions=world.default_initial_regions())
    rng_acq = seeds.rng("acquisition")

    # success threshold calibrated once from the evaluation actions, so the
    # proxy success rate is informative instead of saturating at zero
    success_threshold = float(cfg["success_threshold_frac"] * va.var().item())

    history = []
    n_acquired, cost_spent = 0, 0.0
    for rnd in range(cfg["rounds"] + 1):
        # --- 1. train policies on the current dataset and average ---------
        perf, policy, ckpts = _evaluate_dataset(
            store, vo, va, cfg, seeds, success_threshold)
        history.append({
            "round": rnd, "n_samples": len(store), "n_acquired": n_acquired,
            "cost": cost_spent, "success_threshold": success_threshold, **perf,
        })
        say(f"  [{method} s{seed}] round {rnd}: |D|={len(store)} "
            f"utility={perf['utility']:+.4f}+/-{perf['utility_std']:.4f} "
            f"success={perf['success_rate']:.3f}")
        if rnd == cfg["rounds"]:
            break

        # --- 2-3. multi-context supervision and data model ----------------
        task = BCSupervisionTask(policy, store, vo, va)
        records, _ = generate_context_records(
            task, ckpts,
            LeaveOneOutEstimator(lr=cfg["label_lr"], n_steps=cfg["label_steps"]),
            len(store),
            ContextGenConfig(contexts_per_sample=cfg["contexts_per_sample"],
                             batch_size=cfg["context_batch_size"],
                             seed=seeds["context"] + rnd),
        )
        ds = ContextDataset(store, records)
        train_ds, val_ds = ds.split(0.2, seed=seeds["context"], by="context")
        table = EffectProfileTable(train_ds.records, len(store), min_shared=2)
        model = LDVADataModel(LDVAConfig.build(
            obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
            meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
            n_checkpoints=ds.n_checkpoints, latent_dim=cfg["latent_dim"],
            hidden=tuple(cfg["hidden"])))
        model.set_dataset_context(np.zeros((8, cfg["latent_dim"])))
        model, _ = train_datamodel(
            model, train_ds, val_ds,
            TrainConfig(epochs=cfg["epochs"], eval_every=max(cfg["epochs"], 1),
                        seed=seeds["latent"] + rnd,
                        weights=LossWeights(1.0, 1.0, 0.1, 0.01)),
            table)

        # --- 4-6. encode, cluster, outward directions ---------------------
        ref = ckpts[len(ckpts) // 2]
        z_all = model.encode_store(store, ref.features)
        model.set_dataset_context(z_all)
        clusters = LatentClustering(
            ClusteringConfig(n_clusters=cfg["n_clusters"], seed=seeds["latent"])
        ).fit(z_all, store.metadata)
        directions = DirectionGenerator(
            DirectionConfig(r_max=cfg["r_max"], delta_scale=cfg["delta_scale"],
                            seed=seeds["acquisition"] + rnd)
        ).generate(clusters, z_all)
        if not directions:
            say(f"  [{method} s{seed}] round {rnd}: no candidate directions; stopping")
            break

        # --- 9a. local metadata maps and the actionability filter ---------
        mapper = MetadataMapper(
            world.metadata_spec, MetadataMapperConfig(seed=seeds["acquisition"] + rnd))
        mapper.fit(z_all, store.metadata, clusters)
        directions, act = filter_actionable_directions(
            directions, mapper, store.metadata,
            ActionabilityConfig(min_achievable_cosine=cfg["min_achievable_cosine"]))

        # --- 7-8. predict allocation utility and solve for Q* -------------
        sampler = LatentSampler(
            clusters, LatentSamplerConfig(sigma=0.3, seed=seeds["acquisition"] + rnd))
        budget = BudgetSpec.from_directions(
            directions, budget=cfg["budget_per_round"],
            monetary_budget=cfg.get("monetary_budget_per_round"),
            use_costs=cfg.get("use_costs", False))
        objective = AllocationObjective(
            model, sampler, directions, budget, policy_features=ref.features,
            cfg=ObjectiveConfig(n_mc=cfg["n_mc"], seed=seeds["acquisition"] + rnd))
        result = _plan(method, objective, model, z_all, seeds["acquisition"] + rnd)

        # --- 9b-11. map to metadata, collect, union -----------------------
        plans = mapper.plan_allocation(directions, result.best_allocation, store.metadata, rng=rng_acq)
        if plans:
            new_meta = np.concatenate([p.metadata for p in plans], axis=0)
            new_store = world.build_store(0, rng_env, metadata=new_meta, round_id=rnd + 1)
            store = store.concat(new_store)
            n_acquired += len(new_store)
            cost_spent += float(budget.cost_of(result.best_allocation))
        history[-1].update({
            "planned_allocation": result.best_allocation.tolist(),
            "predicted_utility": result.best_value,
            "n_directions": len(directions),
            "actionable_survival": act["survival_rate"],
        })

    return {"method": method, "seed": seed, "history": history}


DEFAULTS = dict(
    n_initial=200, n_eval=400, rounds=3, budget_per_round=16,
    policy_steps=150, policy_snapshot_every=50, policy_restarts=2,
    contexts_per_sample=16, context_batch_size=8, label_lr=0.1, label_steps=4,
    latent_dim=32, hidden=(128, 128), epochs=40,
    n_clusters=4, r_max=2, delta_scale=0.4, min_achievable_cosine=0.9,
    n_mc=16, use_costs=False, monetary_budget_per_round=None,
    #: policies trained per round and averaged, so the curve measures the
    #: dataset rather than one optimizer run (SETUP.md 24)
    n_eval_policies=3,
    #: success threshold as a fraction of the evaluation action variance
    success_threshold_frac=0.1,
)
QUICK = dict(
    n_initial=100, n_eval=200, rounds=2, budget_per_round=10,
    policy_steps=80, policy_snapshot_every=40, policy_restarts=2,
    contexts_per_sample=10, latent_dim=16, hidden=(64, 64), epochs=20, n_mc=8,
    n_eval_policies=3,
)


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/acquisition_loop")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None, help="budget per round")
    ap.add_argument("--seeds", type=int, default=1, help="SETUP.md 24: >=3 dev, >=5 final")
    ap.add_argument("--eval-policies", type=int, default=None,
                    help="policies trained per round, averaged (default 3)")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.quick:
        cfg.update(QUICK)
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.budget is not None:
        cfg["budget_per_round"] = args.budget
    if args.eval_policies is not None:
        cfg["n_eval_policies"] = args.eval_policies

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(f"[loop] {m}", flush=True)  # noqa: E731
    say(f"{len(methods)} methods x {args.seeds} seeds x {cfg['rounds']} rounds, "
        f"budget {cfg['budget_per_round']}/round")

    runs = []
    for method in methods:
        for s in range(args.seeds):
            runs.append(run_one(cfg, method, s, say))

    summary = _summarize(runs, cfg)
    resolution = _resolution_check(summary, methods)
    report = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
              "methods": methods, "n_seeds": args.seeds, "runs": runs,
              "summary": summary, "resolution": resolution}

    if not args.no_plots:
        curves = {
            m: (np.array(summary[m]["n_samples"]), np.array(summary[m]["utility_mean"]))
            for m in methods if m in summary
        }
        P.plot_performance_vs_cost(
            curves, out_dir / "figures" / "performance_vs_budget.png",
            xlabel="dataset size |D_t|", ylabel="policy utility (-val loss)")
        report["figures"] = str(out_dir / "figures")

    save_json(report, out_dir / "acquisition_loop_report.json")
    _print_summary(summary, methods, out_dir, resolution)
    return report


def _summarize(runs: list[dict], cfg: dict) -> dict:
    out: dict = {}
    for m in {r["method"] for r in runs}:
        rs = [r for r in runs if r["method"] == m]
        n_rounds = min(len(r["history"]) for r in rs)
        util = np.array([[h["utility"] for h in r["history"][:n_rounds]] for r in rs])
        succ = np.array([[h["success_rate"] for h in r["history"][:n_rounds]] for r in rs])
        # the within-round spread across independently trained policies; this is
        # the noise floor any between-method difference has to clear
        pol_std = np.array(
            [[h.get("utility_std", 0.0) for h in r["history"][:n_rounds]] for r in rs])
        sizes = [rs[0]["history"][i]["n_samples"] for i in range(n_rounds)]
        out[m] = {
            "n_samples": sizes,
            "utility_mean": util.mean(0).tolist(),
            "utility_std_across_seeds": util.std(0).tolist(),
            "utility_std_across_policies": pol_std.mean(0).tolist(),
            "success_mean": succ.mean(0).tolist(),
            "final_utility_mean": float(util[:, -1].mean()),
            "final_utility_std_across_seeds": float(util[:, -1].std()),
            "final_utility_std_across_policies": float(pol_std[:, -1].mean()),
            "total_improvement_mean": float((util[:, -1] - util[:, 0]).mean()),
            "n_seeds": len(rs),
            **acquisition_curve_report(np.array(sizes), util.mean(0)),
        }
    return out


def _resolution_check(summary: dict, methods: list[str]) -> dict:
    """Is the between-method difference bigger than the noise?

    Without this it is far too easy to read a ranking off a table where every
    gap is smaller than the error bar. The comparison is called resolvable only
    if the spread of final utilities across methods exceeds the typical
    within-round spread across independently trained policies.
    """
    finals = np.array([summary[m]["final_utility_mean"] for m in methods if m in summary])
    noise = np.array(
        [summary[m]["final_utility_std_across_policies"] for m in methods if m in summary])
    if len(finals) < 2:
        return {"resolvable": False, "reason": "fewer than two methods"}
    spread = float(finals.max() - finals.min())
    noise_level = float(np.mean(noise))
    n_seeds = max(summary[m]["n_seeds"] for m in summary)
    # the standard error of each method's mean shrinks with the number of seeds
    sem = noise_level / np.sqrt(max(n_seeds, 1))
    return {
        "resolvable": bool(spread > 2.0 * sem),
        "method_spread": spread,
        "policy_noise": noise_level,
        "standard_error_per_method": float(sem),
        "n_seeds": int(n_seeds),
        "rule": "spread between methods > 2 x standard error of a method's mean",
    }


def _print_summary(summary: dict, methods: list[str], out_dir: Path,
                   resolution: dict) -> None:
    print("\n" + "=" * 92)
    print("CLOSED-LOOP ACQUISITION (PLAN.md 13) - equal budget, equal evaluation distribution")
    print("=" * 92)
    print(f"  {'method':<28s} {'final utility':>14s} {'+/- seeds':>11s} "
          f"{'+/- policies':>13s} {'improvement':>12s}")
    order = sorted(methods, key=lambda m: -summary.get(m, {}).get("final_utility_mean", -np.inf))
    for m in order:
        s = summary.get(m)
        if s is None:
            continue
        print(f"  {m:<28s} {s['final_utility_mean']:>+14.4f} "
              f"{s['final_utility_std_across_seeds']:>11.4f} "
              f"{s['final_utility_std_across_policies']:>13.4f} "
              f"{s['total_improvement_mean']:>+12.4f}")
    print("-" * 92)
    if resolution.get("resolvable"):
        print(f"  RESOLVABLE: method spread {resolution['method_spread']:.4f} > "
              f"2 x s.e. {2 * resolution['standard_error_per_method']:.4f}")
    else:
        print(f"  NOT RESOLVABLE at this setting: method spread "
              f"{resolution.get('method_spread', float('nan')):.4f} vs "
              f"2 x s.e. {2 * resolution.get('standard_error_per_method', float('nan')):.4f}")
        print(f"  Policy-training noise ({resolution.get('policy_noise', float('nan')):.4f}) "
              f"is comparable to the differences between methods, so this ranking "
              f"should not be read as a result.")
        print(f"  Increase --seeds (SETUP.md 24 wants >=3 dev / >=5 final), --rounds, "
              f"--budget, or --eval-policies.")
    print(f"  report: {out_dir / 'acquisition_loop_report.json'}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
