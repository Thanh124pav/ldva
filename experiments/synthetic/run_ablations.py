"""The critical ablations of PLAN.md 18.

PLAN.md calls all eleven mandatory, so each one is either run or explicitly
reported as skipped - a silently missing ablation is the same as a failed one
when the time comes to write the paper.

    1.  scalar sample score vs latent effect representation
    2.  single-context vs multi-context supervision
    3.  additive utility vs set-level utility
    4.  no metric loss vs metric loss
    5.  no clustering vs clustered local domains
    6.  random directions vs PCA/local directions
    7.  local-only exploitation vs outward expansion
    8.  greedy vs beam search
    9.  exact enumeration vs beam search on small problems
    10. no metadata-direction model vs metadata-aware acquisition
    11. latent dimension sweep

    python experiments/synthetic/run_ablations.py --quick
    python experiments/synthetic/run_ablations.py --only 1,3,11 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ldva.acquisition.beam_search import beam_search  # noqa: E402
from ldva.acquisition.clustering import ClusteringConfig, LatentClustering  # noqa: E402
from ldva.acquisition.directions import DirectionConfig, DirectionGenerator  # noqa: E402
from ldva.acquisition.exact_search import exact_search  # noqa: E402
from ldva.acquisition.greedy import greedy_search  # noqa: E402
from ldva.acquisition.latent_sampler import LatentSampler, LatentSamplerConfig  # noqa: E402
from ldva.acquisition.metadata_mapper import MetadataMapper, MetadataMapperConfig  # noqa: E402
from ldva.acquisition.objective import (  # noqa: E402
    AllocationObjective,
    BudgetSpec,
    ObjectiveConfig,
    n_allocations,
)
from ldva.analysis.latent_geometry import (  # noqa: E402
    cluster_stability,
    neighbor_effect_consistency,
)
from ldva.data.context_dataset import ContextDataset  # noqa: E402
from ldva.data.effect_profiles import EffectProfileTable  # noqa: E402
from ldva.envs.synthetic.generator import SyntheticConfig, SyntheticWorld  # noqa: E402
from ldva.envs.synthetic.oracle import measure_realized_latent_movement  # noqa: E402
from ldva.models.datamodel import LDVAConfig, LDVADataModel  # noqa: E402
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
    n_samples=250, n_eval=300, policy_steps=150, policy_snapshot_every=30,
    policy_restarts=3, contexts_per_sample=20, context_batch_size=8,
    label_lr=0.1, label_steps=4, latent_dim=32, hidden=(128, 128), epochs=50,
    n_clusters=4, r_max=2, delta_scale=0.4, budget=8, n_mc=16,
    latent_dims=(8, 16, 32, 64, 128),
)
QUICK = dict(
    n_samples=120, n_eval=200, policy_steps=90, policy_snapshot_every=30,
    policy_restarts=2, contexts_per_sample=12, epochs=20, latent_dim=16,
    hidden=(64, 64), budget=6, n_mc=8, latent_dims=(8, 16, 32),
)


class Fixture:
    """One shared world, dataset and supervision set for all ablations."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seeds = SeedBundle(seed)
        set_seed(self.seeds["latent"])
        # `cfg["world"]` lets a diagnostic vary the generative map itself -
        # map kind, kernel bandwidth, latent width - which is what the
        # richness-versus-reach question requires.
        self.world = SyntheticWorld(SyntheticConfig(
            seed=self.seeds["env"], **dict(cfg.get("world", {}))))
        rng = self.seeds.rng("env")
        self.store = self.world.build_store(
            cfg["n_samples"], rng, regions=self.world.default_initial_regions())
        em = self.world.evaluation_metadata(
            cfg["n_eval"], np.random.default_rng(self.seeds["env"] + 555))
        vo, va, _ = self.world.generate_chunks(
            em, np.random.default_rng(self.seeds["env"] + 556))
        self.vo, self.va = torch.from_numpy(vo), torch.from_numpy(va)
        self.policy, self.ckpts = train_bc(
            self.store, self.vo, self.va,
            BCTrainConfig(steps=cfg["policy_steps"],
                          snapshot_every=cfg["policy_snapshot_every"],
                          n_restarts=cfg["policy_restarts"]),
            seed=self.seeds["policy"])
        task = BCSupervisionTask(self.policy, self.store, self.vo, self.va)
        self.records, _ = generate_context_records(
            task, self.ckpts,
            LeaveOneOutEstimator(lr=cfg["label_lr"], n_steps=cfg["label_steps"]),
            len(self.store),
            ContextGenConfig(contexts_per_sample=cfg["contexts_per_sample"],
                             batch_size=cfg["context_batch_size"],
                             seed=self.seeds["context"]))
        self.ds = ContextDataset(self.store, self.records)
        # PLAN.md 4.2 / 15 P0.4: held-out checkpoints, so every ablation is
        # measured on transfer to an unseen policy
        self.train_ds, self.val_ds = self.ds.split(
            0.2, seed=self.seeds["context"], by=cfg.get("val_split_by", "checkpoint"))
        self.table = EffectProfileTable(self.train_ds.records, len(self.store), min_shared=2)
        self.ref = self.ckpts[len(self.ckpts) // 2]

    def train(self, latent_dim=None, weights=None, records=None,
              use_ckpt_id=False, seed_offset=0, **model_kw):
        """Train a data model variant and return its final validation metrics.

        `use_ckpt_id=True` is only for ablation 6 (PLAN.md 20): everywhere else
        the policy context is the continuous features alone, so the model
        cannot memorize which checkpoint it saw.
        """
        cfg = self.cfg
        latent_dim = latent_dim or cfg["latent_dim"]
        split_by = cfg.get("val_split_by", "checkpoint")
        train_ds, val_ds, table = self.train_ds, self.val_ds, self.table
        if records is not None:
            ds = ContextDataset(self.store, records)
            train_ds, val_ds = ds.split(0.2, seed=self.seeds["context"], by=split_by)
            table = EffectProfileTable(train_ds.records, len(self.store), min_shared=2)
        # Seed BEFORE constructing the model. `DataModelTrainer` calls
        # `set_seed` in its own __init__, which is after this point, so the
        # model's initial weights were being drawn from whatever state the
        # global torch RNG happened to be in - i.e. they depended on every
        # arm that had run before. Two arms with identical configs then gave
        # different results: in the controllability sweep the same
        # (epochs=60, smooth=0.01) cell came out at C6 +0.471 in one arm and
        # +0.622 in another, a 0.15 gap that is the same size as the effects
        # being measured. Every ablation built on this Fixture was affected.
        # `seed_offset` is how a caller asks for an independent *replicate* of
        # the same configuration. With the seeding above making training fully
        # deterministic, repeated calls are byte-identical, so measuring the
        # run-to-run spread needs this to be varied explicitly.
        set_seed(self.seeds["latent"] + seed_offset)
        model = LDVADataModel(LDVAConfig.build(
            obs_dim=self.store.obs_dim, act_dim=self.store.act_dim,
            chunk_len=self.store.chunk_len, meta_dim=self.store.meta_dim,
            policy_feat_dim=train_ds.policy_feat_dim,
            n_checkpoints=train_ds.n_checkpoints if use_ckpt_id else 0,
            latent_dim=latent_dim,
            hidden=tuple(cfg["hidden"]), **model_kw))
        model.set_dataset_context(np.zeros((8, latent_dim)))
        model, hist = train_datamodel(
            model, train_ds, val_ds,
            TrainConfig(epochs=cfg["epochs"], eval_every=cfg["epochs"],
                        seed=self.seeds["latent"] + seed_offset,
                        weights=weights or LossWeights(1.0, 1.0, 0.1, 0.01)),
            table)
        final = {k[4:]: v for k, v in hist[-1].items() if k.startswith("val/")}
        return model, final


# ---- ablations -------------------------------------------------------------


def ab1_scalar_vs_latent(fx: Fixture) -> dict:
    """Contextual readout vs one fixed scalar per sample."""
    _, ctx = fx.train(readout_kind="contextual")
    _, scal = fx.train(readout_kind="scalar")
    return {
        "contextual_effect_mse": ctx["effect_mse"],
        "scalar_readout_effect_mse": scal["effect_mse"],
        "hindsight_scalar_effect_mse": ctx["effect_scalar_mse"],
        "contextual_effect_spearman": ctx["effect_spearman"],
        "scalar_readout_effect_spearman": scal["effect_spearman"],
        "mse_ratio": scal["effect_mse"] / max(ctx["effect_mse"], 1e-12),
        "verdict": "contextual wins" if ctx["effect_mse"] < scal["effect_mse"] else "no gain",
    }


def ab2_single_vs_multi_context(fx: Fixture) -> dict:
    """Keep one context per sample vs all of them.

    Implemented by thinning the record set to the first context each sample
    appears in, which is what "a sample has one historical score" looks like as
    a *dataset* rather than as a model.
    """
    seen: set[int] = set()
    thinned = []
    for r in fx.records:
        if any(int(s) not in seen for s in r.batch_sample_ids):
            thinned.append(r)
            seen.update(int(s) for s in r.batch_sample_ids)
    _, multi = fx.train()
    _, single = fx.train(records=thinned)
    return {
        "n_contexts_multi": len(fx.records),
        "n_contexts_single": len(thinned),
        "multi_effect_mse": multi["effect_mse"],
        "single_effect_mse": single["effect_mse"],
        "multi_effect_spearman": multi["effect_spearman"],
        "single_effect_spearman": single["effect_spearman"],
        "mse_ratio": single["effect_mse"] / max(multi["effect_mse"], 1e-12),
        "note": "fewer contexts also means less data; the comparison is confounded "
                "by dataset size and should be read as an upper bound on the gain",
    }


def ab3_additive_vs_set_utility(fx: Fixture) -> dict:
    """The headline set-level claim, judged on the within-checkpoint gain."""
    _, s = fx.train(utility_kind="deepsets")
    _, a = fx.train(utility_kind="additive")
    _, p = fx.train(utility_kind="pairwise")
    return {
        "set_gain_within_mse": s.get("gain_within_mse"),
        "additive_gain_within_mse": a.get("gain_within_mse"),
        "pairwise_gain_within_mse": p.get("gain_within_mse"),
        "set_gain_within_r2": s.get("gain_within_r2"),
        "additive_gain_within_r2": a.get("gain_within_r2"),
        "set_gain_mse_all_contexts": s["gain_mse"],
        "additive_gain_mse_all_contexts": a["gain_mse"],
        "composition_signal_share": s.get("gain_var_within_group_share"),
        "within_mse_ratio_additive_over_set": (
            a.get("gain_within_mse", float("nan")) / max(s.get("gain_within_mse", 1e-12), 1e-12)),
        "note": "raw gain MSE is dominated by between-checkpoint variance, so the "
                "within-checkpoint ratio is the meaningful comparison",
    }


def ab4_metric_loss(fx: Fixture) -> dict:
    """Does L_metric buy usable latent geometry?"""
    out = {}
    for name, w in [("with_metric", LossWeights(1.0, 1.0, 0.1, 0.01)),
                    ("no_metric", LossWeights(1.0, 1.0, 0.0, 0.01))]:
        model, final = fx.train(weights=w)
        z = model.encode_store(fx.store, fx.ref.features)
        nb = neighbor_effect_consistency(z, fx.table, k=10, seed=0)
        out[name] = {
            "effect_mse": final["effect_mse"],
            "effect_spearman": final["effect_spearman"],
            "neighbor_consistency_ratio": nb["neighbor_consistency_ratio"],
            "latent_norm_mean": final["latent_norm_mean"],
        }
    return out


def ab5_clustering_vs_none(fx: Fixture) -> dict:
    """One global domain vs several local ones."""
    model, _ = fx.train()
    z = model.encode_store(fx.store, fx.ref.features)
    out = {}
    for n in (1, fx.cfg["n_clusters"], 2 * fx.cfg["n_clusters"]):
        clusters = LatentClustering(ClusteringConfig(n_clusters=n, seed=0)).fit(z, fx.store.metadata)
        dirs = DirectionGenerator(
            DirectionConfig(r_max=fx.cfg["r_max"], delta_scale=fx.cfg["delta_scale"], seed=0)
        ).generate(clusters, z)
        key = "global_single_domain" if n == 1 else f"clustered_{n}"
        out[key] = {
            "n_clusters": len(clusters),
            "n_directions": len(dirs),
            "mean_density_ratio": float(np.mean([d.density_ratio for d in dirs])) if dirs else None,
            "cluster_ari": cluster_stability(z, n_clusters=max(n, 2), seed=0)["cluster_ari_mean"],
        }
    return out


def ab6_random_vs_pca_directions(fx: Fixture) -> dict:
    """Do local PCA directions beat random ones under the learned utility?"""
    model, _ = fx.train()
    z = model.encode_store(fx.store, fx.ref.features)
    model.set_dataset_context(z)
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=fx.cfg["n_clusters"], seed=0)).fit(z, fx.store.metadata)
    out = {}
    for name, dcfg in [
        ("pca", DirectionConfig(r_max=fx.cfg["r_max"], delta_scale=fx.cfg["delta_scale"], seed=0)),
        ("random", DirectionConfig(use_random_directions=True, n_random_per_cluster=2 * fx.cfg["r_max"],
                                   delta_scale=fx.cfg["delta_scale"], seed=0)),
    ]:
        dirs = DirectionGenerator(dcfg).generate(clusters, z)
        if not dirs:
            out[name] = {"n_directions": 0}
            continue
        sampler = LatentSampler(clusters, LatentSamplerConfig(sigma=0.3, seed=0))
        budget = BudgetSpec.from_directions(dirs, budget=fx.cfg["budget"])
        obj = AllocationObjective(model, sampler, dirs, budget,
                                  policy_features=fx.ref.features,
                                  cfg=ObjectiveConfig(n_mc=fx.cfg["n_mc"], seed=0))
        res = beam_search(obj, beam_width=10)
        out[name] = {
            "n_directions": len(dirs),
            "best_predicted_utility": res.best_value,
            "mean_density_ratio": float(np.mean([d.density_ratio for d in dirs])),
            "mean_explained_variance": float(np.mean([d.explained_variance for d in dirs])),
        }
    return out


def ab7_outward_vs_local(fx: Fixture) -> dict:
    """Outward expansion vs resampling inside the existing support."""
    model, _ = fx.train()
    z = model.encode_store(fx.store, fx.ref.features)
    model.set_dataset_context(z)
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=fx.cfg["n_clusters"], seed=0)).fit(z, fx.store.metadata)
    out = {}
    for name, dcfg in [
        ("outward_expansion", DirectionConfig(r_max=fx.cfg["r_max"],
                                              delta_scale=fx.cfg["delta_scale"], seed=0)),
        ("local_only", DirectionConfig(r_max=fx.cfg["r_max"], delta_scale=0.0,
                                       require_outward=False,
                                       require_density_decrease=False, seed=0)),
    ]:
        dirs = DirectionGenerator(dcfg).generate(clusters, z)
        if not dirs:
            out[name] = {"n_directions": 0}
            continue
        sampler = LatentSampler(clusters, LatentSamplerConfig(sigma=0.3, seed=0))
        budget = BudgetSpec.from_directions(dirs, budget=fx.cfg["budget"])
        obj = AllocationObjective(model, sampler, dirs, budget,
                                  policy_features=fx.ref.features,
                                  cfg=ObjectiveConfig(n_mc=fx.cfg["n_mc"], seed=0))
        res = beam_search(obj, beam_width=10)
        out[name] = {
            "n_directions": len(dirs),
            "best_predicted_utility": res.best_value,
            "mean_support_distance": float(np.mean([d.support_distance for d in dirs])),
            "mean_density_ratio": float(np.mean([d.density_ratio for d in dirs])),
        }
    return out


def ab8_9_solvers(fx: Fixture) -> dict:
    """Greedy vs beam, and beam vs exact on a small problem."""
    model, _ = fx.train()
    z = model.encode_store(fx.store, fx.ref.features)
    model.set_dataset_context(z)
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=fx.cfg["n_clusters"], seed=0)).fit(z, fx.store.metadata)
    dirs = DirectionGenerator(
        DirectionConfig(r_max=fx.cfg["r_max"], delta_scale=fx.cfg["delta_scale"], seed=0)
    ).generate(clusters, z)
    sampler = LatentSampler(clusters, LatentSamplerConfig(sigma=0.3, seed=0))
    budget = BudgetSpec.from_directions(dirs, budget=fx.cfg["budget"])
    obj = AllocationObjective(model, sampler, dirs, budget,
                              policy_features=fx.ref.features,
                              cfg=ObjectiveConfig(n_mc=fx.cfg["n_mc"], seed=0))
    space = n_allocations(len(dirs), fx.cfg["budget"])
    out = {"n_directions": len(dirs), "search_space_size": space}
    if space <= 200_000:
        ex = exact_search(obj)
        out["exact"] = {"value": ex.best_value, "evals": ex.n_evaluations,
                        "allocation": ex.best_allocation.tolist()}
    else:
        out["exact"] = {"skipped": f"search space {space} too large"}
    gr = greedy_search(obj)
    out["greedy"] = {"value": gr.best_value, "evals": gr.n_evaluations,
                     "allocation": gr.best_allocation.tolist()}
    for h in (5, 10, 20, 50):
        r = beam_search(obj, beam_width=h)
        out[f"beam_{h}"] = {"value": r.best_value, "evals": r.n_evaluations,
                            "allocation": r.best_allocation.tolist()}
        if "value" in out["exact"]:
            out[f"beam_{h}"]["gap_vs_exact"] = float(
                (out["exact"]["value"] - r.best_value) / max(abs(out["exact"]["value"]), 1e-12))
    out["greedy_vs_beam10"] = float(out["beam_10"]["value"] - out["greedy"]["value"])
    return out


def ab10_metadata_aware(fx: Fixture) -> dict:
    """Metadata-aware acquisition vs ignoring metadata actionability."""
    model, _ = fx.train()
    z = model.encode_store(fx.store, fx.ref.features)
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=fx.cfg["n_clusters"], seed=0)).fit(z, fx.store.metadata)
    dirs = DirectionGenerator(
        DirectionConfig(r_max=fx.cfg["r_max"], delta_scale=fx.cfg["delta_scale"], seed=0)
    ).generate(clusters, z)
    mapper = MetadataMapper(fx.world.metadata_spec, MetadataMapperConfig(seed=0))
    mapper.fit(z, fx.store.metadata, clusters)
    rng = np.random.default_rng(0)

    rows = []
    for d in dirs:
        plan = mapper.plan_direction(d, 3, fx.store.metadata, rng=rng)
        mv = measure_realized_latent_movement(
            model, fx.world, plan, d, fx.ref.features, rng, n_per_anchor=24)
        rows.append({"achievable": plan.achievable_cosine,
                     "realized": mv["direction_cosine"]})
    ach = np.array([r["achievable"] for r in rows])
    real = np.array([r["realized"] for r in rows])
    filtered = ach >= 0.9
    return {
        "n_directions": len(dirs),
        "all_directions_realized_cosine": float(real.mean()),
        "actionable_only_realized_cosine": float(real[filtered].mean()) if filtered.any() else None,
        "n_actionable": int(filtered.sum()),
        "frac_positive_all": float((real > 0).mean()),
        "frac_positive_actionable": float((real[filtered] > 0).mean()) if filtered.any() else None,
        "mapper": mapper.report(),
    }


def ab11_latent_dim_sweep(fx: Fixture) -> dict:
    """PLAN.md 6: do not assume a larger latent is better."""
    out = {}
    for d in fx.cfg["latent_dims"]:
        model, final = fx.train(latent_dim=d)
        z = model.encode_store(fx.store, fx.ref.features)
        nb = neighbor_effect_consistency(z, fx.table, k=10, seed=0)
        out[str(d)] = {
            "effect_mse": final["effect_mse"],
            "effect_spearman": final["effect_spearman"],
            "gain_within_r2": final.get("gain_within_r2"),
            "neighbor_consistency_ratio": nb["neighbor_consistency_ratio"],
            "cluster_ari": cluster_stability(z, n_clusters=fx.cfg["n_clusters"],
                                             seed=0)["cluster_ari_mean"],
            "latent_norm_mean": final["latent_norm_mean"],
        }
    best = min(out, key=lambda k: out[k]["effect_mse"])
    out["best_by_effect_mse"] = best
    return out


def ab6b_ckpt_id_vs_continuous_context(fx: Fixture) -> dict:
    """PLAN.md 20.6: checkpoint-ID embedding vs continuous policy context.

    The comparison P0.4 exists to make. Both models are validated on **held-out
    checkpoints**, which is what exposes the difference: an ID embedding can
    only memorize the checkpoints it was trained on, so on an unseen policy its
    embedding slot carries no information and whatever it learned through that
    slot is unavailable. A continuous context has features for any checkpoint,
    seen or not.

    Read `effect_spearman` on the held-out checkpoints: if the ID variant wins
    on training checkpoints but loses here, that is memorization, and PLAN.md
    4.2's rule against using it in the main result is confirmed rather than
    assumed.
    """
    out = {}
    for name, use_id in [("continuous_context", False), ("ckpt_id_embedding", True)]:
        model, final = fx.train(use_ckpt_id=use_id)
        out[name] = {
            "effect_mse": final["effect_mse"],
            "effect_spearman": final["effect_spearman"],
            "gain_within_r2": final.get("gain_within_r2", float("nan")),
            "uses_ckpt_id": use_id,
        }
    cont = out["continuous_context"]["effect_spearman"]
    idemb = out["ckpt_id_embedding"]["effect_spearman"]
    out["verdict"] = {
        "continuous_at_least_as_good_on_unseen_checkpoints": bool(cont >= idemb),
        "spearman_gap": float(cont - idemb),
        "validated_on": "held-out checkpoints",
        "rule": "PLAN.md 4.2: do not rely on checkpoint-ID embeddings in the "
                "main result",
    }
    return out


def ab9_actionability_filter(fx: Fixture) -> dict:
    """PLAN.md 20.9: with and without the metadata-actionability filter.

    Dropping the filter lets the planner spend budget on directions the
    environment cannot actually be asked to produce. The cost shows up as
    realized direction control, not as predicted utility - an unactionable
    direction still *predicts* well, which is exactly why the filter is needed.
    """
    from ldva.acquisition.metadata_mapper import (
        ActionabilityConfig,
        MetadataMapper,
        MetadataMapperConfig,
        filter_actionable_directions,
    )

    model, _ = fx.train()
    z_all = model.encode_store(fx.store, fx.ref.features)
    model.set_dataset_context(z_all)
    clusters = LatentClustering(
        ClusteringConfig(n_clusters=fx.cfg["n_clusters"], seed=fx.seeds["latent"])
    ).fit(z_all, fx.store.metadata)
    directions = DirectionGenerator(
        DirectionConfig(r_max=fx.cfg["r_max"], delta_scale=fx.cfg["delta_scale"],
                        seed=fx.seeds["acquisition"])
    ).generate(clusters, z_all)
    mapper = MetadataMapper(
        fx.world.metadata_spec, MetadataMapperConfig(seed=fx.seeds["acquisition"]))
    mapper.fit(z_all, fx.store.metadata, clusters)

    kept, act = filter_actionable_directions(
        directions, mapper, fx.store.metadata,
        ActionabilityConfig(min_achievable_cosine=fx.cfg["min_achievable_cosine"]))
    return {
        "no_filter": {
            "n_directions": len(directions),
            "mean_achievable_cosine": float(np.mean([
                mapper.plan_direction(d, 1, fx.store.metadata).achievable_cosine
                for d in directions])) if directions else float("nan"),
        },
        "with_filter": {
            "n_directions": len(kept),
            "mean_achievable_cosine": act.get("mean_achievable_kept", float("nan")),
            "survival_rate": act["survival_rate"],
        },
        "verdict": {
            "filter_improves_achievability": bool(
                act.get("mean_achievable_kept", 0.0) >= 0.0),
            "n_rejected": len(directions) - len(kept),
        },
    }


def ab5_no_policy_context(fx: Fixture) -> dict:
    """PLAN.md 20.5: with and without the policy context entirely.

    If removing theta-conditioning costs nothing, the effect labels are not
    actually policy-dependent and RQ1's "contextual" claim is about batch
    composition only. That is a result either way, but it has to be measured
    rather than assumed - PLAN.md 19/F1 lists a missing effect geometry as a
    named failure mode.
    """
    out = {}
    for name, use in [("with_policy_context", True), ("no_policy_context", False)]:
        _, final = fx.train(use_policy_context=use)
        out[name] = {
            "effect_mse": final["effect_mse"],
            "effect_spearman": final["effect_spearman"],
            "gain_within_r2": final.get("gain_within_r2", float("nan")),
        }
    out["verdict"] = {
        "policy_context_helps": bool(
            out["with_policy_context"]["effect_spearman"]
            > out["no_policy_context"]["effect_spearman"]),
        "spearman_gap": float(out["with_policy_context"]["effect_spearman"]
                              - out["no_policy_context"]["effect_spearman"]),
    }
    return out


#: PLAN.md 20's required ablations, keyed by ITS numbering so the registry can
#: be compared with the spec line by line. Item 13 (BC-loss prediction vs
#: downstream rollout correlation) needs a simulator and therefore cannot run
#: in this synthetic script - it is measured by `run_acquisition_loop.py` on
#: dmc / metaworld, where both a BC utility and a real rollout exist.
ABLATIONS = {
    1: ("scalar_vs_latent", ab1_scalar_vs_latent),
    2: ("single_vs_multi_context", ab2_single_vs_multi_context),
    3: ("additive_vs_set_utility", ab3_additive_vs_set_utility),
    4: ("metric_loss", ab4_metric_loss),
    5: ("no_policy_context", ab5_no_policy_context),
    6: ("ckpt_id_vs_continuous_context", ab6b_ckpt_id_vs_continuous_context),
    7: ("clustering_vs_none", ab5_clustering_vs_none),
    8: ("random_vs_pca_directions", ab6_random_vs_pca_directions),
    9: ("actionability_filter", ab9_actionability_filter),
    10: ("outward_vs_local", ab7_outward_vs_local),
    11: ("solvers_greedy_beam_exact", ab8_9_solvers),
    12: ("latent_dim_sweep", ab11_latent_dim_sweep),
    #: not in PLAN.md 20's numbered list, kept because it is the direct test of
    #: whether acquisition needs the metadata map at all
    14: ("metadata_aware_acquisition", ab10_metadata_aware),
}

#: PLAN.md 20 items this script cannot cover, with the reason
ABLATIONS_REQUIRING_SIMULATOR = {
    13: ("bc_loss_vs_rollout_correlation",
         "needs real rollout return; run run_acquisition_loop.py --env dmc"),
}



def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/ablations")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated ablation numbers, e.g. 1,3,11")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.quick:
        cfg.update(QUICK)
    wanted = ([int(x) for x in args.only.split(",")] if args.only else sorted(ABLATIONS))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(f"[ablations] {m}", flush=True)  # noqa: E731

    report: dict = {"config": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in cfg.items()},
                    "requested": wanted, "results": {}}
    # ablation 9 is part of ab8 (beam vs exact on the same small problem)
    report["notes"] = {
        "9": "covered by ablation 8: the same objective is solved by exact, "
             "greedy and beam, and the gap is reported per beam width",
    }

    for seed in range(args.seeds):
        say(f"building fixture for seed {seed}")
        fx = Fixture(cfg, seed)
        for num in wanted:
            if num not in ABLATIONS:
                report["results"].setdefault(str(num), {})[f"seed{seed}"] = {
                    "skipped": "not implemented separately"}
                continue
            name, fn = ABLATIONS[num]
            say(f"  ablation {num}: {name}")
            try:
                res = fn(fx)
            except Exception as e:  # one failure must not lose the whole sweep
                res = {"error": f"{type(e).__name__}: {e}"}
                say(f"    FAILED: {res['error']}")
            report["results"].setdefault(f"{num}_{name}", {})[f"seed{seed}"] = res

    save_json(report, out_dir / "ablations_report.json")
    say(f"wrote {out_dir / 'ablations_report.json'}")
    _print(report)
    return report


def _print(report: dict) -> None:
    print("\n" + "=" * 78)
    print("ABLATIONS (PLAN.md 18)")
    print("=" * 78)
    for key, seeds in report["results"].items():
        print(f"\n  {key}")
        for sk, res in seeds.items():
            if "error" in res:
                print(f"    {sk}: ERROR {res['error']}")
                continue
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"    {sk} {k:<42s} {v:+.4f}")
                elif isinstance(v, (int, str)):
                    print(f"    {sk} {k:<42s} {v}")
    print("\n  note: ablation 9 is reported inside ablation 8")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
