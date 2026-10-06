"""Closed-loop acquisition through `EnvAdapter` (PLAN.md 10, 15 P0.1/P0.2/P0.4).

Each round runs the twelve steps of PLAN.md 10:

    train policy -> generate multi-context supervision -> update the data model
    -> encode -> cluster -> outward directions -> predict allocation utility
    -> solve for Q* -> map directions to metadata -> collect -> union -> repeat

and the same loop runs for every acquisition *method* under an identical budget,
initial dataset, policy architecture, training budget and evaluation
distribution. That is the control PLAN.md 12 requires: a difference in the
final curve is then a difference in *what each method chose to collect*, not in
what it was allowed to collect or how it was measured.

Four things here are the P0 fixes of PLAN.md 15, and each one changes what the
numbers mean:

**P0.1 - the headline metric is a rollout, not a loss.** Where the adapter can
drive a simulator (`supports_rollout_eval`), every round reports real return
and real success rate from `evaluate_policy`, measured from conditions drawn
once before any acquisition. The BC validation loss is still recorded, as
supervision diagnostics, but PLAN.md 18 does not accept it as a robotics
outcome and neither does the summary table: `primary_metric` names which one is
being ranked, and on an adapter with no simulator it says so.

**P0.2 - nothing here knows which environment it is.** The loop touches only
`metadata_spec`, `initial_dataset`, `evaluation_set`, `eval_conditions`,
`collect` and `evaluate_policy`. `--env synthetic|dmc|metaworld` is the whole
difference between a preflight gate and a MuJoCo manipulation experiment.

**P0.3 - ablations and baselines are separated and labelled.** The default
method set is PLAN.md 17 E1's: Random, Diversity, Direct GradAlign, Direct
Influence, LDVA Greedy, LDVA Beam. The two Direct methods compute their own
scores from real gradients and never read the LDVA model. Methods that *do*
score through the LDVA model carry an `abl_` prefix and are tagged
`ldva_ablation` in the report, so they cannot be read as published baselines.

**P0.4 - the policy context cannot silently disagree with itself.** The
reference checkpoint travels as one `PolicyContextRef`, and the checkpoint-ID
embedding is off unless `--ckpt-id-ablation` asks for it. The data model's
validation split holds out whole checkpoints, so the reported validation number
measures transfer to an unseen policy rather than to an unseen batch.

    python experiments/run_acquisition_loop.py --env synthetic --quick
    python experiments/run_acquisition_loop.py --env dmc --seeds 3 --rounds 4
    python experiments/run_acquisition_loop.py --env metaworld --env-task push-v3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

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
from ldva.acquisition.external_baselines import (  # noqa: E402
    ProspectiveProxyConfig,
    classify_method,
    direct_gradient_alignment_acquisition,
    direct_influence_acquisition,
)
from ldva.acquisition.greedy import greedy_search  # noqa: E402
from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig  # noqa: E402
from ldva.acquisition.metadata_mapper import (  # noqa: E402
    ActionabilityConfig,
    MetadataMapper,
    MetadataMapperConfig,
    evaluate_realized_direction,
    filter_actionable_directions,
)
from ldva.acquisition.objective import (  # noqa: E402
    AllocationObjective,
    BudgetSpec,
    ObjectiveConfig,
)
from ldva.analysis import plotting as P  # noqa: E402
from ldva.analysis.acquisition_calibration import acquisition_curve_report  # noqa: E402
from ldva.analysis.wandb_logger import make_logger  # noqa: E402
from ldva.data.cache import RunCache  # noqa: E402
from ldva.data.context_dataset import ContextDataset  # noqa: E402
from ldva.data.effect_profiles import EffectProfileTable  # noqa: E402
from ldva.envs.base import get_adapter  # noqa: E402
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
from ldva.utils import (  # noqa: E402
    SeedBundle,
    load_config,
    run_provenance,
    save_json,
    set_seed,
)

#: PLAN.md 17 E1's required comparison, in the order it lists them. Two
#: independent external baselines, two model-free rules, two LDVA planners.
METHODS = (
    "random",
    "diversity",
    "direct_gradient_alignment",
    "direct_influence",
    "ldva_greedy",
    "ldva_beam",
)

#: LDVA internal ablations (PLAN.md 12.1). Opt in with --methods; the `abl_`
#: prefix is deliberate - these score through the LDVA model and must never
#: appear in a table as reproductions of published methods.
ABLATION_METHODS = (
    "abl_equal",
    "abl_uncertainty",
    "abl_gradient_alignment",
    "abl_influence_cupid_style",
)


def _plan(method: str, ctx: dict):
    """Dispatch one method to its allocation rule.

    `ctx` carries everything any method could need; each rule takes only what
    its own assumptions license. The external baselines are handed the real
    supervision task and the environment's metadata, and are never handed the
    data model.

    The name is validated first so a typo in `--methods` fails immediately with
    the list of valid names, rather than hours later as a missing-key error
    from whichever branch happened to run.
    """
    if method not in METHODS + ABLATION_METHODS:
        raise ValueError(
            f"unknown method {method!r}; known: {sorted(METHODS + ABLATION_METHODS)}"
        )
    objective = ctx["objective"]
    if method == "ldva_beam":
        return beam_search(objective, beam_width=ctx["beam_width"])
    if method == "ldva_greedy":
        return greedy_search(objective)
    if method == "random":
        return random_acquisition(objective, ctx["seed"])
    if method == "diversity":
        return diversity_acquisition(objective, ctx["z_all"], ctx["seed"])
    if method == "direct_gradient_alignment":
        return direct_gradient_alignment_acquisition(
            objective, ctx["task"], ctx["store"].metadata, ctx["metadata_spec"],
            ctx["mapper"], cfg=ctx["proxy_cfg"],
        )
    if method == "direct_influence":
        return direct_influence_acquisition(
            objective, ctx["task"], ctx["store"].metadata, ctx["metadata_spec"],
            ctx["mapper"], cfg=ctx["proxy_cfg"],
        )
    # ---- LDVA internal ablations (PLAN.md 12.1) --------------------------
    if method == "abl_equal":
        return equal_allocation(objective)
    if method == "abl_uncertainty":
        return uncertainty_acquisition(objective, ctx["model"])
    if method == "abl_gradient_alignment":
        return gradient_alignment_acquisition(objective, ctx["model"])
    if method == "abl_influence_cupid_style":
        return influence_acquisition(objective, ctx["model"])
    raise AssertionError(f"method {method!r} passed validation but has no branch")


def _supervision_policy(store, vo, va, cfg, seeds):
    """Train the policy whose *checkpoints* become supervision contexts.

    Optimized for checkpoint diversity, not for final performance: several
    restarts from perturbed initializations, snapshots along the way. If every
    context came from one theta the data model could not learn a
    policy-conditioned readout at all (PLAN.md 4.2), so the spread is the point.
    """
    return train_bc(
        store, vo, va,
        BCTrainConfig(steps=cfg["policy_steps"],
                      snapshot_every=cfg["policy_snapshot_every"],
                      n_restarts=cfg["policy_restarts"],
                      lr=cfg["policy_lr"],
                      optimizer=cfg["policy_optimizer"]),
        hidden=tuple(cfg["policy_hidden"]),
        seed=seeds["policy"],
    )


def _evaluate_dataset(env, store, vo, va, cfg, seeds, success_threshold, conditions):
    """Train `n_eval_policies` policies on `store`, average their performance.

    **Why this is not the same training run as the supervision policy.** The two
    serve opposite goals and one config cannot serve both. Supervision wants a
    spread-out checkpoint cloud, which `init_scale` and multiple restarts
    deliberately produce by *degrading* each run. The acquisition curve wants
    the opposite: the best policy this dataset can support, because the
    question being asked is what the data is worth. Sharing one config made the
    measurement meaningless on DMC reacher - the evaluated policy reached a
    rollout return of 3.2 where the scripted expert scores 100 and a properly
    trained BC policy on the same data reaches 79. With no dynamic range in the
    metric, no acquisition method could have looked better than another, and
    the experiment would have produced a confident flat result.

    So evaluation policies get a clean initialization (`init_scale=None`), one
    run each, and the optimizer and step budget that actually converge. Their
    seeds are *fixed* - not round- or method-dependent - so the only thing
    varying across rounds and methods is the dataset itself. One policy per
    round would make the curve measure optimizer noise: on the quick synthetic
    setting that produced a non-monotonic curve (-1.36, -1.84, -1.34) whose
    swings exceeded every between-method difference.
    """
    utils, succs, rollout_ret, rollout_succ = [], [], [], []
    for k in range(cfg["n_eval_policies"]):
        policy, _ = train_bc(
            store, vo, va,
            BCTrainConfig(steps=cfg["eval_policy_steps"],
                          snapshot_every=max(cfg["eval_policy_steps"], 1),
                          n_restarts=1,
                          lr=cfg["eval_policy_lr"],
                          optimizer=cfg["eval_policy_optimizer"],
                          init_scale=None),
            hidden=tuple(cfg["policy_hidden"]),
            seed=seeds["policy"] + 1000 * k,
        )
        m = evaluate_bc(policy, vo, va, success_threshold=success_threshold)
        utils.append(m["utility"])
        succs.append(m["success_rate"])
        if conditions is not None:
            r = env.evaluate_policy(policy, conditions)
            rollout_ret.append(r.mean_return)
            rollout_succ.append(r.success_rate)

    out = {
        "utility": float(np.mean(utils)),
        "utility_std": float(np.std(utils)),
        "bc_success_rate": float(np.mean(succs)),
        "val_loss": float(-np.mean(utils)),
        "n_eval_policies": cfg["n_eval_policies"],
    }
    if rollout_ret:
        out.update({
            "rollout_return": float(np.mean(rollout_ret)),
            "rollout_return_std": float(np.std(rollout_ret)),
            "rollout_success": float(np.mean(rollout_succ)),
            "rollout_success_std": float(np.std(rollout_succ)),
        })
    return out


def run_one(cfg: dict, env_name: str, env_kwargs: dict, method: str, seed: int,
            say, cache: RunCache, wb: dict | None = None) -> dict:
    """Run the acquisition loop for one method and one seed."""
    t_start = time.time()
    seeds = SeedBundle(seed)
    wb = wb or {}
    # one wandb run per (method, seed), grouped by experiment: the six methods
    # of PLAN.md 17 E1 then overlay on one chart and seeds average within a
    # method. The step axis is the acquisition round, which is the x-axis
    # PLAN.md 18 asks the curve to be plotted against.
    log = make_logger(
        enabled=bool(wb.get("enabled")),
        project=wb.get("project", "ldva"),
        name=f"{wb.get('group', env_name)}-{method}-s{seed}",
        group=wb.get("group"),
        job_type=method,
        config={
            "env": env_name, "env_kwargs": env_kwargs, "method": method,
            "method_class": classify_method(method), "seed": seed,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            **wb.get("provenance", {}),
        },
        tags=(env_name, classify_method(method), wb.get("experiment", "adhoc")),
    )
    # two clocks in one run: the acquisition curve advances per round, the data
    # model per epoch. Declaring each family's x-axis is what keeps them from
    # fighting over wandb's implicit step counter; see `define_steps`.
    log.define_steps({
        "round": None,
        "datamodel/epoch": None,
        "*": "round",
        "datamodel/*": "datamodel/epoch",
    })
    set_seed(seeds["latent"])
    env = get_adapter(env_name, **env_kwargs)

    # --- fixed evaluation, declared before any acquisition ----------------
    # cached per seed, so every method is scored on byte-identical data
    vo, va = cache.evaluation_set(env, env_kwargs, seed, cfg["n_eval"], seeds.rng("env"))
    conditions = None
    if env.supports_rollout_eval and cfg["n_eval_rollouts"] > 0:
        conditions = cache.eval_conditions(
            env, env_kwargs, seed, cfg["n_eval_rollouts"], seeds.rng("env"))

    rng_env = seeds.rng("env")
    store = cache.initial_dataset(env, env_kwargs, seed, cfg["n_initial"], rng_env)
    rng_acq = seeds.rng("acquisition")

    # success threshold calibrated once from the evaluation actions, so the BC
    # proxy success rate is informative instead of saturating at zero
    success_threshold = float(cfg["success_threshold_frac"] * va.var().item())

    history: list[dict] = []
    n_acquired, cost_spent = 0, 0.0
    # warm-start: keep one model across rounds so the latent FRAME is stable.
    # Only used when cfg["warm_start"] is true; otherwise `model` stays None
    # and each round reinstantiates (the historical behaviour).
    persistent_model = None
    for rnd in range(cfg["rounds"] + 1):
        # --- 1. train policies on the current dataset and average ---------
        perf = _evaluate_dataset(
            env, store, vo, va, cfg, seeds, success_threshold, conditions)
        history.append({
            "round": rnd, "n_samples": len(store), "n_acquired": n_acquired,
            "cost": cost_spent, "success_threshold": success_threshold, **perf,
        })
        msg = (f"  [{env_name}/{method} s{seed}] round {rnd}: |D|={len(store)} "
               f"utility={perf['utility']:+.4f}+/-{perf['utility_std']:.4f}")
        if "rollout_return" in perf:
            msg += (f" return={perf['rollout_return']:.2f}"
                    f" success={perf['rollout_success']:.3f}")
        say(msg)
        log.log(dict(history[-1]))
        if rnd == cfg["rounds"]:
            break

        # --- 2-3. multi-context supervision and the data model -------------
        # a SEPARATE training run, optimized for checkpoint diversity rather
        # than for performance; see `_evaluate_dataset` for why these cannot be
        # the same policy
        policy, ckpts = _supervision_policy(store, vo, va, cfg, seeds)
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
        # P0.4: hold out whole CHECKPOINTS, not whole contexts. A context split
        # only tests transfer to an unseen batch composition; PLAN.md 4.2 asks
        # for transfer to an unseen policy, which is the harder claim and the
        # one the acquisition loop actually relies on.
        train_ds, val_ds = ds.split(
            cfg["val_frac"], seed=seeds["context"], by=cfg["val_split_by"])
        table = EffectProfileTable(train_ds.records, len(store), min_shared=2)
        # P0.4: no checkpoint-ID embedding in the main model. With one, the
        # policy context can memorize the checkpoints it saw and has nothing to
        # say about an unseen future policy.
        n_ckpt_vocab = train_ds.n_checkpoints if cfg["ckpt_id_ablation"] else 0
        if cfg["warm_start"] and persistent_model is not None:
            # reuse the round-0 encoder; freeze sample and policy encoders so
            # the latent frame stays fixed and only the heads adapt to new data
            model = persistent_model
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            for p in model.policy_encoder.parameters():
                p.requires_grad_(False)
            round_epochs = cfg["epochs"]
        else:
            model = LDVADataModel(LDVAConfig.build(
                obs_dim=store.obs_dim, act_dim=store.act_dim, chunk_len=store.chunk_len,
                meta_dim=store.meta_dim, policy_feat_dim=ds.policy_feat_dim,
                n_checkpoints=n_ckpt_vocab, latent_dim=cfg["latent_dim"],
                hidden=tuple(cfg["hidden"])))
            model.set_dataset_context(np.zeros((8, cfg["latent_dim"])))
            round_epochs = (cfg["warm_start_epochs"]
                            if cfg["warm_start"] else cfg["epochs"])
        model, train_hist = train_datamodel(
            model, train_ds, val_ds,
            TrainConfig(epochs=round_epochs, eval_every=max(round_epochs, 1),
                        seed=seeds["latent"] + rnd,
                        weights=LossWeights(1.0, 1.0, 0.1, 0.01),
                        # attach, never init: this trains once per round inside
                        # the run opened above
                        wandb=log.active, wandb_attach=True,
                        wandb_prefix="datamodel/"),
            table)
        if cfg["warm_start"]:
            persistent_model = model

        # --- 4-6. encode, cluster, outward directions ---------------------
        # P0.4: features and vocabulary index travel together, so they cannot
        # describe different checkpoints
        ref = ckpts[len(ckpts) // 2]
        pctx = PolicyContextRef.from_checkpoint(
            ref, train_ds.checkpoint_ids, use_ckpt_id=cfg["ckpt_id_ablation"])
        z_all = model.encode_store(store, pctx.features, ckpt_index=pctx.ckpt_index)
        model.set_dataset_context(z_all)
        clusters = LatentClustering(
            ClusteringConfig(n_clusters=cfg["n_clusters"], seed=seeds["latent"])
        ).fit(z_all, store.metadata)
        directions = DirectionGenerator(
            DirectionConfig(r_max=cfg["r_max"], delta_scale=cfg["delta_scale"],
                            seed=seeds["acquisition"] + rnd)
        ).generate(clusters, z_all)
        if not directions:
            say(f"  [{env_name}/{method} s{seed}] round {rnd}: no candidate "
                f"directions; stopping")
            break

        # --- 9a. local metadata maps and the actionability filter ---------
        mapper = MetadataMapper(
            env.metadata_spec, MetadataMapperConfig(seed=seeds["acquisition"] + rnd))
        mapper.fit(z_all, store.metadata, clusters)
        directions, act = filter_actionable_directions(
            directions, mapper, store.metadata,
            ActionabilityConfig(min_achievable_cosine=cfg["min_achievable_cosine"]))
        if not directions:
            say(f"  [{env_name}/{method} s{seed}] round {rnd}: no actionable "
                f"directions; stopping")
            break

        # --- 7-8. predict allocation utility and solve for Q* -------------
        sampler = LatentSampler(
            clusters, LatentSamplerConfig(sigma=0.3, seed=seeds["acquisition"] + rnd))
        budget = BudgetSpec.from_directions(
            directions, budget=cfg["budget_per_round"],
            monetary_budget=cfg.get("monetary_budget_per_round"),
            use_costs=cfg.get("use_costs", False))
        objective = AllocationObjective(
            model, sampler, directions, budget, policy_context=pctx,
            cfg=ObjectiveConfig(n_mc=cfg["n_mc"], seed=seeds["acquisition"] + rnd))
        result = _plan(method, {
            "objective": objective, "model": model, "z_all": z_all,
            "seed": seeds["acquisition"] + rnd, "task": task, "store": store,
            "metadata_spec": env.metadata_spec, "mapper": mapper,
            "beam_width": cfg["beam_width"],
            "proxy_cfg": ProspectiveProxyConfig(
                k_neighbors=cfg["proxy_k_neighbors"],
                n_probe=cfg["proxy_n_probe"],
                seed=seeds["acquisition"] + rnd),
        })

        # --- 9b-11. map to metadata, collect, union -----------------------
        plans = mapper.plan_allocation(
            directions, result.best_allocation, store.metadata, rng=rng_acq)
        realized = []
        if plans:
            new_meta = np.concatenate([p.metadata for p in plans], axis=0)
            new_store = env.collect(new_meta, rng_env, round_id=rnd + 1)
            # metadata-direction control (PLAN.md 18): did the data that
            # arrived actually move the latents the way the plan asked? Encoded
            # at the SAME reference context as z_all, or the comparison would
            # confound a direction with a change of policy context.
            z_new = model.encode_store(
                new_store, pctx.features, ckpt_index=pctx.ckpt_index)
            off = 0
            for p in plans:
                k = len(p.metadata)
                z_slice = z_new[off:off + k] if off + k <= len(z_new) else z_new[off:]
                off += k
                if len(z_slice) == 0:
                    continue
                realized.append(evaluate_realized_direction(
                    z_all[p.anchor_sample_ids], z_slice, p.desired_latent_direction))
            store = store.concat(new_store)
            n_acquired += len(new_store)
            cost_spent += float(budget.cost_of(result.best_allocation))

        history[-1].update({
            "planned_allocation": result.best_allocation.tolist(),
            "predicted_utility": result.best_value,
            "n_directions": len(directions),
            "actionable_survival": act["survival_rate"],
            "solver": result.solver,
            "datamodel_val": train_hist[-1] if train_hist else {},
            "direction_control_cosine": (
                float(np.mean([r["direction_cosine"] for r in realized]))
                if realized else float("nan")),
            "direction_control_frac_positive": (
                float(np.mean([r["frac_samples_positive_cosine"] for r in realized]))
                if realized else float("nan")),
            "n_checkpoints_train": int(train_ds.n_checkpoints),
            "n_checkpoints_val": int(val_ds.n_checkpoints),
            "policy_context": pctx.to_dict(),
        })
        log.log({k: v for k, v in history[-1].items()
                 if k not in ("planned_allocation", "policy_context",
                              "datamodel_val")})

    out = {
        "method": method,
        "method_class": classify_method(method),
        "seed": seed,
        "env": env_name,
        "history": history,
        "wall_time_s": round(time.time() - t_start, 2),
        "eval_conditions": conditions.to_dict() if conditions is not None else None,
        "rollout_eval": bool(conditions is not None),
        "wandb_url": log.url,
    }
    last = history[-1] if history else {}
    log.summary({
        "final_utility": last.get("utility", float("nan")),
        "final_rollout_return": last.get("rollout_return", float("nan")),
        "final_rollout_success": last.get("rollout_success", float("nan")),
        "n_samples_final": last.get("n_samples", 0),
        "n_acquired_total": last.get("n_acquired", 0),
        "wall_time_s": out["wall_time_s"],
        "method_class": out["method_class"],
        "rollout_eval": bool(conditions is not None),
        "eval_conditions_fingerprint": (
            conditions.fingerprint() if conditions is not None else "none"),
    })
    log.finish()
    return out


# ---- configuration --------------------------------------------------------

DEFAULTS = dict(
    n_initial=200, n_eval=400, n_eval_rollouts=20, rounds=3, budget_per_round=16,
    #: SUPERVISION policy: spread checkpoints, quality secondary
    policy_steps=150, policy_snapshot_every=50, policy_restarts=2,
    policy_hidden=(128, 128), policy_lr=1e-2, policy_optimizer="sgd",
    #: EVALUATION policy: converge, because its rollout return is the headline
    #: metric. Measured on DMC reacher with 900 chunks: sgd/400 reaches return
    #: 27.7, adam-1e-3/2000 reaches 75.5, adam-1e-3/6000 reaches 79.2, against
    #: a scripted expert at 100.
    eval_policy_steps=2000, eval_policy_lr=1e-3, eval_policy_optimizer="adam",
    contexts_per_sample=16, context_batch_size=8, label_lr=0.1, label_steps=4,
    latent_dim=32, hidden=(128, 128), epochs=40,
    n_clusters=4, r_max=2, delta_scale=0.4, min_achievable_cosine=0.9,
    n_mc=16, beam_width=10, use_costs=False, monetary_budget_per_round=None,
    #: policies trained per round and averaged, so the curve measures the
    #: dataset rather than one optimizer run
    n_eval_policies=3,
    #: BC success threshold as a fraction of the evaluation action variance
    success_threshold_frac=0.1,
    #: P0.4: held-out *checkpoints*, the honest policy-transfer test
    val_frac=0.2, val_split_by="checkpoint", ckpt_id_ablation=False,
    #: external-baseline proxy (PLAN.md 12.2)
    proxy_k_neighbors=8, proxy_n_probe=4,
)

QUICK = dict(
    n_initial=100, n_eval=200, n_eval_rollouts=8, rounds=2, budget_per_round=10,
    policy_steps=80, policy_snapshot_every=40, policy_restarts=2,
    eval_policy_steps=600,
    contexts_per_sample=10, latent_dim=16, hidden=(64, 64), epochs=20, n_mc=8,
    n_eval_policies=3,
)

#: per-environment overrides. The policy width follows each adapter's
#: `policy_defaults`; the rest is what makes a round affordable on one machine.
ENV_PRESETS = {
    "synthetic": dict(policy_hidden=()),
    "dmc": dict(
        n_initial=60, n_eval=300, n_eval_rollouts=20, policy_hidden=(128, 128),
        policy_steps=400, policy_snapshot_every=100, budget_per_round=12,
        contexts_per_sample=12, n_clusters=4,
        eval_policy_steps=3000,
    ),
    "metaworld": dict(
        n_initial=60, n_eval=300, n_eval_rollouts=20, policy_hidden=(256, 256),
        policy_steps=600, policy_snapshot_every=150, budget_per_round=12,
        contexts_per_sample=12, n_clusters=4,
        eval_policy_steps=4000,
    ),
}


def _env_kwargs(env_name: str, args) -> dict:
    """Adapter constructor arguments, from the YAML config plus CLI overrides.

    PLAN.md 15 P1 asks for experiment scripts to load the YAML configs rather
    than duplicate hard-coded defaults, so `configs/env/<name>.yaml` is the
    source of truth for task choice, chunk length and horizon.
    """
    path = Path(__file__).resolve().parents[1] / "configs" / "env" / f"{env_name}.yaml"
    raw = load_config(path) if path.exists() else {}
    kw: dict = {}
    if env_name == "dmc":
        kw = {
            "task": args.env_task or raw.get("task", "reacher-easy"),
            "chunk_len": int(raw.get("chunk_len", 16)),
            "max_steps": int(raw.get("max_steps", 120)),
            "max_chunks_per_episode": int(raw.get("max_chunks_per_episode", 3)),
        }
    elif env_name == "metaworld":
        tasks = (
            (args.env_task,) if args.env_task
            else tuple(raw.get("tasks", ("push-v3",)))
        )
        kw = {
            "tasks": tasks,
            "chunk_len": int(raw.get("chunk_len", 16)),
            "max_steps": int(raw.get("max_steps", 160)),
            "max_chunks_per_episode": int(raw.get("max_chunks_per_episode", 3)),
        }
    return kw


def build_config(args) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(ENV_PRESETS.get(args.env, {}))
    if args.quick:
        cfg.update(QUICK)
        cfg.update({k: v for k, v in ENV_PRESETS.get(args.env, {}).items()
                    if k == "policy_hidden"})
        if args.env in ("dmc", "metaworld"):
            cfg.update(n_initial=40, n_eval=150, n_eval_rollouts=6,
                       policy_steps=200, policy_snapshot_every=50,
                       budget_per_round=8)
    if args.config:
        cfg.update(load_config(args.config))
    for key, val in (("rounds", args.rounds), ("budget_per_round", args.budget),
                     ("n_eval_policies", args.eval_policies),
                     ("n_eval_rollouts", args.eval_rollouts),
                     ("n_initial", args.n_initial)):
        if val is not None:
            cfg[key] = val
    if args.ckpt_id_ablation:
        cfg["ckpt_id_ablation"] = True
    if args.val_split_by:
        cfg["val_split_by"] = args.val_split_by
    cfg["warm_start"] = bool(getattr(args, "warm_start", False))
    _ws_e = getattr(args, "warm_start_epochs", None)
    cfg["warm_start_epochs"] = _ws_e if _ws_e is not None else 2 * cfg["epochs"]
    return cfg


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", type=str, default="synthetic",
                    choices=("synthetic", "dmc", "metaworld"))
    ap.add_argument("--env-task", type=str, default=None,
                    help="dmc task (reacher-easy) or single metaworld task (push-v3)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--config", type=str, default=None, help="YAML config override")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None, help="budget per round")
    ap.add_argument("--n-initial", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=1,
                    help="PLAN.md 14: >=3 development, >=5 final")
    ap.add_argument("--eval-policies", type=int, default=None,
                    help="policies trained per round, averaged (default 3)")
    ap.add_argument("--eval-rollouts", type=int, default=None,
                    help="fixed rollout conditions; 0 disables rollout eval")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--ckpt-id-ablation", action="store_true",
                    help="PLAN.md 12.1: enable the checkpoint-ID embedding")
    ap.add_argument("--val-split-by", type=str, default=None,
                    choices=("checkpoint", "context"))
    ap.add_argument("--cache", type=str, default=".cache/acquisition",
                    help="cache dir for D_0 / eval sets; 'none' disables")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--wandb", action="store_true",
                    help="log to Weights & Biases (PLAN.md 15 P1): one run per "
                         "(method, seed), grouped by experiment")
    ap.add_argument("--wandb-project", type=str, default="ldva")
    ap.add_argument("--wandb-group", type=str, default=None,
                    help="wandb group; defaults to <experiment>-<env>")
    ap.add_argument("--experiment", type=str, default=None,
                    help="experiment label recorded as a wandb tag, e.g. E1")
    ap.add_argument("--warm-start", action="store_true",
                    help="train the data model once at round 0 for "
                         "warm_start_epochs; freeze the sample encoder + "
                         "policy encoder after that so later rounds only "
                         "re-fit readout/utility on the same latent frame "
                         "(docs/E1_E2_first_results.md, C6 drift)")
    ap.add_argument("--warm-start-epochs", type=int, default=None,
                    help="epochs for the round-0 warm start; defaults to 2x "
                         "the per-round --epochs")
    args = ap.parse_args()

    cfg = build_config(args)
    env_kwargs = _env_kwargs(args.env, args)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.out or f"runs/acquisition_{args.env}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = RunCache(
        root=None if args.cache == "none" else Path(args.cache) / args.env,
        enabled=args.cache != "none",
    )

    prov = run_provenance({"env": args.env})
    experiment = args.experiment or "adhoc"
    wb = {
        "enabled": bool(args.wandb),
        "project": args.wandb_project,
        "group": args.wandb_group or f"{experiment}-{args.env}",
        "experiment": experiment,
        "provenance": {f"prov/{k}": v for k, v in prov.items()
                       if isinstance(v, (str, int, float, bool))},
    }

    say = lambda m: print(f"[loop] {m}", flush=True)  # noqa: E731
    say(f"env={args.env} {env_kwargs}")
    if args.wandb:
        say(f"wandb: project={wb['project']} group={wb['group']} "
            f"(one run per method x seed)")
    say(f"{len(methods)} methods x {args.seeds} seeds x {cfg['rounds']} rounds, "
        f"budget {cfg['budget_per_round']}/round")

    probe = get_adapter(args.env, **env_kwargs)
    rollout_ok = probe.supports_rollout_eval and cfg["n_eval_rollouts"] > 0
    primary = "rollout_return" if rollout_ok else "utility"
    if not rollout_ok:
        say(f"NOTE: {args.env} has no rollout evaluation; ranking on the BC "
            f"proxy utility, which PLAN.md 18 does not count as a robotics "
            f"outcome")
    say(f"primary metric: {primary}")

    runs = []
    for method in methods:
        for s in range(args.seeds):
            runs.append(run_one(cfg, args.env, env_kwargs, method, s, say, cache, wb))

    summary = _summarize(runs, primary)
    resolution = _resolution_check(summary, methods, primary)
    report = {
        "env": args.env,
        "env_kwargs": env_kwargs,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "methods": methods,
        "method_classes": {m: classify_method(m) for m in methods},
        "primary_metric": primary,
        "n_seeds": args.seeds,
        "runs": runs,
        "summary": summary,
        "resolution": resolution,
        "provenance": prov,
        "cache": cache.report(),
        "wandb": {"enabled": bool(args.wandb), "project": wb["project"],
                  "group": wb["group"]},
    }

    if not args.no_plots:
        key = f"{primary}_mean"
        curves = {
            m: (np.array(summary[m]["n_samples"]), np.array(summary[m][key]))
            for m in methods if m in summary and key in summary[m]
        }
        if curves:
            P.plot_performance_vs_cost(
                curves, out_dir / "figures" / f"{primary}_vs_budget.png",
                xlabel="dataset size |D_t|",
                ylabel=("rollout return" if rollout_ok else "policy utility (-val loss)"))
            report["figures"] = str(out_dir / "figures")

    save_json(report, out_dir / "acquisition_loop_report.json")
    _log_summary_run(wb, summary, methods, resolution, primary, args, report)
    _print_summary(summary, methods, out_dir, resolution, primary, args.env)
    return report


def _log_summary_run(wb, summary, methods, resolution, primary, args, report) -> None:
    """One extra wandb run holding the comparison table and the verdict.

    Kept separate from the per-method runs so the wandb UI has a single place
    showing which method won and - crucially - whether the difference was
    resolvable at all. Without that flag in the same view, a chart showing six
    overlapping curves invites reading a winner out of noise, which is exactly
    what `_resolution_check` exists to prevent.
    """
    if not wb.get("enabled"):
        return
    log = make_logger(
        enabled=True, project=wb["project"],
        name=f"{wb['group']}-summary", group=wb["group"], job_type="summary",
        config={"methods": list(methods), "primary_metric": primary,
                "env": args.env, "n_seeds": args.seeds,
                **wb.get("provenance", {})},
        tags=(args.env, "summary", wb.get("experiment", "adhoc")),
    )
    cols = ["method", "class", "final", "std_seeds", "std_policies",
            "improvement", "direction_cosine", "n_seeds"]
    rows = []
    for m in sorted(methods, key=lambda x: -summary.get(x, {}).get("final_mean", -np.inf)):
        s_ = summary.get(m)
        if s_ is None:
            continue
        rows.append([m, s_["method_class"], s_["final_mean"],
                     s_["final_std_across_seeds"], s_["final_std_across_policies"],
                     s_["total_improvement_mean"],
                     s_.get("direction_control_cosine", float("nan")),
                     s_["n_seeds"]])
    log.table("comparison", cols, rows)
    for m, s_ in summary.items():
        log.summary({f"final/{m}": s_["final_mean"]})
    log.summary({
        "primary_metric": primary,
        "resolvable": bool(resolution.get("resolvable", False)),
        "method_spread": float(resolution.get("method_spread", float("nan"))),
        "standard_error_per_method": float(
            resolution.get("standard_error_per_method", float("nan"))),
        "n_seeds": int(resolution.get("n_seeds", 0)),
        "best_method": rows[0][0] if rows else "none",
        "verdict": ("resolvable" if resolution.get("resolvable")
                    else "NOT resolvable - do not read a winner"),
    })
    fig = report.get("figures")
    if fig:
        from pathlib import Path as _P

        for png in sorted(_P(fig).glob("*.png")):
            log.image(png.stem, png)
    log.finish()


def _safe_nanmean(a: np.ndarray) -> float:
    """`nanmean` that returns NaN instead of warning on an all-NaN input.

    An all-NaN direction-control column is the ordinary state for a round in
    which no plan was executed, not an anomaly worth a RuntimeWarning.
    """
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _summarize(runs: list[dict], primary: str) -> dict:
    out: dict = {}
    for m in {r["method"] for r in runs}:
        rs = [r for r in runs if r["method"] == m]
        n_rounds = min(len(r["history"]) for r in rs)
        if n_rounds == 0:
            continue

        def col(name, default=float("nan")):
            return np.array([[h.get(name, default) for h in r["history"][:n_rounds]]
                             for r in rs], dtype=np.float64)

        util = col("utility")
        sizes = [rs[0]["history"][i]["n_samples"] for i in range(n_rounds)]
        rec = {
            "method_class": rs[0]["method_class"],
            "n_samples": sizes,
            "utility_mean": util.mean(0).tolist(),
            "utility_std_across_seeds": util.std(0).tolist(),
            # the within-round spread across independently trained policies:
            # the noise floor any between-method difference has to clear
            "utility_std_across_policies": col("utility_std", 0.0).mean(0).tolist(),
            "n_seeds": len(rs),
            "wall_time_s": float(np.mean([r["wall_time_s"] for r in rs])),
            "direction_control_cosine": _safe_nanmean(
                col("direction_control_cosine")),
        }
        if any("rollout_return" in h for r in rs for h in r["history"]):
            ret = col("rollout_return")
            suc = col("rollout_success")
            rec.update({
                "rollout_return_mean": ret.mean(0).tolist(),
                "rollout_return_std_across_seeds": ret.std(0).tolist(),
                "rollout_return_std_across_policies":
                    col("rollout_return_std", 0.0).mean(0).tolist(),
                "rollout_success_mean": suc.mean(0).tolist(),
                "rollout_success_std_across_seeds": suc.std(0).tolist(),
            })
        p = f"{primary}_mean"
        series = np.array(rec.get(p, rec["utility_mean"]), dtype=np.float64)
        pol_key = f"{primary}_std_across_policies"
        pol = np.array(
            rec.get(pol_key, rec["utility_std_across_policies"]), dtype=np.float64)
        seed_key = f"{primary}_std_across_seeds"
        seeds_std = np.array(
            rec.get(seed_key, rec["utility_std_across_seeds"]), dtype=np.float64)
        rec.update({
            "final_mean": float(series[-1]),
            "final_std_across_seeds": float(seeds_std[-1]),
            "final_std_across_policies": float(pol[-1]),
            "total_improvement_mean": float(series[-1] - series[0]),
            **acquisition_curve_report(np.array(sizes), series),
        })
        out[m] = rec
    return out


def _resolution_check(summary: dict, methods: list[str], primary: str) -> dict:
    """Is the between-method difference bigger than the noise?

    Without this it is far too easy to read a ranking off a table where every
    gap is smaller than the error bar. The comparison is called resolvable only
    if the spread of final scores across methods exceeds the standard error of
    a single method's mean, estimated from the within-round spread across
    independently trained policies.
    """
    present = [m for m in methods if m in summary]
    if len(present) < 2:
        return {"resolvable": False, "reason": "fewer than two methods"}
    finals = np.array([summary[m]["final_mean"] for m in present])
    noise = np.array([summary[m]["final_std_across_policies"] for m in present])
    spread = float(np.nanmax(finals) - np.nanmin(finals))
    noise_level = float(np.nanmean(noise))
    n_seeds = max(summary[m]["n_seeds"] for m in present)
    sem = noise_level / np.sqrt(max(n_seeds, 1))
    return {
        "metric": primary,
        "resolvable": bool(spread > 2.0 * sem) if np.isfinite(spread) else False,
        "method_spread": spread,
        "policy_noise": noise_level,
        "standard_error_per_method": float(sem),
        "n_seeds": int(n_seeds),
        "rule": "spread between methods > 2 x standard error of a method's mean",
    }


def _print_summary(summary, methods, out_dir, resolution, primary, env) -> None:
    w = 104
    print("\n" + "=" * w)
    print(f"CLOSED-LOOP ACQUISITION on {env} (PLAN.md 10, 17) - equal budget, "
          f"equal evaluation distribution")
    print(f"primary metric: {primary}")
    print("=" * w)
    print(f"  {'method':<28s} {'class':<14s} {'final':>12s} {'+/- seeds':>11s} "
          f"{'+/- policies':>13s} {'improvement':>12s}")
    order = sorted(methods, key=lambda m: -summary.get(m, {}).get("final_mean", -np.inf))
    for m in order:
        s = summary.get(m)
        if s is None:
            continue
        print(f"  {m:<28s} {s['method_class']:<14s} {s['final_mean']:>+12.4f} "
              f"{s['final_std_across_seeds']:>11.4f} "
              f"{s['final_std_across_policies']:>13.4f} "
              f"{s['total_improvement_mean']:>+12.4f}")
    print("-" * w)
    if resolution.get("resolvable"):
        print(f"  RESOLVABLE: method spread {resolution['method_spread']:.4f} > "
              f"2 x s.e. {2 * resolution['standard_error_per_method']:.4f}")
    else:
        print(f"  NOT RESOLVABLE at this setting: method spread "
              f"{resolution.get('method_spread', float('nan')):.4f} vs "
              f"2 x s.e. {2 * resolution.get('standard_error_per_method', float('nan')):.4f}")
        print("  Policy-training noise is comparable to the differences between "
              "methods, so this")
        print("  ranking should NOT be read as a result. Increase --seeds "
              "(PLAN.md 14 wants >=3 dev /")
        print("  >=5 final), --rounds, --budget, or --eval-policies.")
    print(f"  report: {out_dir / 'acquisition_loop_report.json'}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
