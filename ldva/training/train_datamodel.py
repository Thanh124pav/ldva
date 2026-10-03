"""Train the LDVA data model on stored context records (PLAN.md Phase 2).

The trainer reports, every evaluation, the two baselines that the first-milestone
claim depends on (PLAN.md 22): a constant predictor and the best hindsight
*fixed scalar per sample*. If the contextual readout does not beat the latter,
the model is not using context and there is no reason to continue to clustering
and acquisition - so these numbers are first-class outputs, not diagnostics.

The additivity (F2) check lives in `analysis/latent_geometry.py` instead: it is a
property of the *labels*, not of the model, and its additive fit has one free
parameter per sample, so running it on a small validation split would report
perfect additivity no matter what the data looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ldva.data.context_dataset import ContextDataset
from ldva.data.effect_profiles import EffectProfileTable
from ldva.models.datamodel import LDVADataModel
from ldva.training import losses as L
from ldva.training.metrics import (
    constant_baseline_metrics,
    fit_per_sample_scalar,
    group_variance_decomposition,
    per_sample_scalar_baseline_metrics,
    regression_metrics,
    within_group_metrics,
)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    weights: L.LossWeights = field(default_factory=L.LossWeights)
    #: L_smooth perturbation strength
    smooth_drop_frac: float = 0.25
    smooth_policy_noise: float = 0.05
    max_metric_pairs_per_context: int = 28
    eval_every: int = 5
    seed: int = 0
    device: str = "auto"
    progress: bool = False
    #: keep the parameters with the best validation effect spearman
    select_metric: str = "effect_spearman"
    select_mode: str = "max"
    wandb: bool = False
    wandb_project: str = "ldva"
    wandb_run_name: str | None = None
    #: attach to a run the CALLER already started instead of creating one.
    #: The acquisition loop retrains the data model every round inside a single
    #: per-(method, seed) run; without this, each round would call
    #: `wandb.init(reinit=True)` and then `finish()`, which ends the parent run
    #: and splits one acquisition curve across a dozen orphan runs.
    wandb_attach: bool = False
    #: namespace for the attached metrics, so per-epoch data-model losses do
    #: not collide with the loop's per-round metrics
    wandb_prefix: str = ""


class DataModelTrainer:
    def __init__(
        self,
        model: LDVADataModel,
        train_ds: ContextDataset,
        val_ds: ContextDataset | None = None,
        cfg: TrainConfig | None = None,
        effect_table: EffectProfileTable | None = None,
    ):
        from ldva.utils import get_device, set_seed

        self.cfg = cfg or TrainConfig()
        set_seed(self.cfg.seed)
        self.device = get_device(self.cfg.device)
        self.model = model.to(self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.effect_table = effect_table
        self.rng = np.random.default_rng(self.cfg.seed)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        # the per-sample scalar baseline is fitted on TRAIN contexts so that it
        # is scored out-of-sample, exactly like the model (PLAN.md 14 crit. 1)
        self._scalar_fit = fit_per_sample_scalar(
            np.concatenate([r.batch_sample_ids for r in train_ds.records]),
            np.concatenate(
                [
                    train_ds.normalizer.effect(r.per_sample_effects)
                    if train_ds.normalize
                    else r.per_sample_effects
                    for r in train_ds.records
                ]
            ),
        )
        self.history: list[dict] = []
        self._best_state: dict | None = None
        self._best_value = -np.inf if self.cfg.select_mode == "max" else np.inf
        self._wandb = None
        #: True when we created the run and are therefore responsible for
        #: finishing it; False when attached to a caller's run
        self._owns_wandb = False
        if self.cfg.wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb

            if self.cfg.wandb_attach:
                # use whatever run the caller started; never create or end one
                self._wandb = wandb.run
                self._owns_wandb = False
                return
            self._wandb = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config={"train": self.cfg.__dict__, "weights": self.cfg.weights.as_dict()},
                reinit=True,
            )
            self._owns_wandb = True
        except Exception as e:  # logging must never break training
            print(f"[ldva] wandb disabled: {e}")
            self._wandb = None

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    # ---- loss assembly ---------------------------------------------------
    def _compute_losses(self, batch: dict) -> tuple[torch.Tensor, dict]:
        w = self.cfg.weights
        out = self.model(batch)
        mask = batch["mask"]

        l_effect = L.effect_loss(out["effect_pred"], batch["effects"], mask)
        l_batch = L.batch_gain_loss(out["gain_pred"], batch["gain"])
        total = w.effect * l_effect + w.batch * l_batch
        logs = {"l_effect": float(l_effect.item()), "l_batch": float(l_batch.item())}

        if w.metric > 0 and self.effect_table is not None:
            l_metric, n_pairs = L.metric_loss(
                out["z"],
                batch["sample_ids"],
                mask,
                self.effect_table.lookup_batch,
                max_pairs_per_context=self.cfg.max_metric_pairs_per_context,
                rng=self.rng,
            )
            if n_pairs > 0:
                total = total + w.metric * l_metric
                logs["l_metric"] = float(l_metric.item())
                logs["metric_pairs"] = n_pairs

        if w.smooth > 0:
            l_smooth, shared = self._smoothness(batch, out)
            if shared > 0:
                total = total + w.smooth * l_smooth
                logs["l_smooth"] = float(l_smooth.item())

        logs["loss"] = float(total.item())
        return total, logs

    def _smoothness(self, batch: dict, out: dict) -> tuple[torch.Tensor, int]:
        """Perturb the batch composition and the policy context slightly."""
        mask = batch["mask"]
        n_ctx, n_max = mask.shape
        keep = mask.clone()
        for c in range(n_ctx):
            members = torch.nonzero(mask[c]).flatten()
            n_drop = int(self.cfg.smooth_drop_frac * len(members))
            # always leave at least two members so the context stays meaningful
            n_drop = min(n_drop, max(0, len(members) - 2))
            if n_drop > 0:
                sel = members[torch.randperm(len(members), device=mask.device)[:n_drop]]
                keep[c, sel] = False
        if int(keep.sum().item()) == 0:
            return torch.zeros((), device=mask.device), 0

        pert = dict(batch)
        pert["mask"] = keep
        pert["policy_features"] = batch["policy_features"] + self.cfg.smooth_policy_noise * torch.randn_like(
            batch["policy_features"]
        )
        out_p = self.model(pert)
        shared = keep & mask
        return (
            L.smoothness_loss(out["effect_pred"], out_p["effect_pred"], shared),
            int(shared.sum().item()),
        )

    # ---- loop -------------------------------------------------------------
    def fit(self) -> list[dict]:
        loader = self.train_ds.loader(batch_size=self.cfg.batch_size, shuffle=True)
        epochs = range(1, self.cfg.epochs + 1)
        if self.cfg.progress:
            epochs = tqdm(epochs, desc="datamodel")

        for epoch in epochs:
            self.model.train()
            agg: dict[str, list[float]] = {}
            for raw in loader:
                batch = self._to_device(raw)
                total, logs = self._compute_losses(batch)
                self.opt.zero_grad(set_to_none=True)
                total.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                for k, v in logs.items():
                    agg.setdefault(k, []).append(v)

            row = {"epoch": epoch, **{f"train/{k}": float(np.mean(v)) for k, v in agg.items()}}
            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs:
                if self.val_ds is not None:
                    val = self.evaluate(self.val_ds)
                    row.update({f"val/{k}": v for k, v in val.items()})
                    self._maybe_select(val)
            self.history.append(row)
            self._log_row(row, epoch)

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        if self._wandb is not None and self._owns_wandb:
            self._wandb.finish()
        return self.history

    def _log_row(self, row: dict, epoch: int) -> None:
        """Send one epoch's metrics to wandb, if logging is on.

        An attached run does not pass `step`: its step axis belongs to the
        caller (the acquisition round), and forcing the epoch number onto it
        would rewind the parent's step counter and silently drop the rest of
        the loop's metrics.
        """
        if self._wandb is None:
            return
        try:
            pre = self.cfg.wandb_prefix
            out = {f"{pre}{k}": v for k, v in row.items()} if pre else dict(row)
            if self._owns_wandb:
                self._wandb.log(out, step=epoch)
            else:
                self._wandb.log(out)
        except Exception:
            pass  # logging must never break training

    def _maybe_select(self, val: dict) -> None:
        v = val.get(self.cfg.select_metric)
        if v is None or not np.isfinite(v):
            return
        better = v > self._best_value if self.cfg.select_mode == "max" else v < self._best_value
        if better:
            self._best_value = v
            self._best_state = {
                k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()
            }

    # ---- evaluation --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, ds: ContextDataset) -> dict:
        self.model.eval()
        eff_p, eff_t, eff_ids, gain_p, gain_t, gain_ckpt = [], [], [], [], [], []
        z_norms = []
        for raw in ds.loader(batch_size=self.cfg.batch_size, shuffle=False):
            batch = self._to_device(raw)
            out = self.model(batch)
            m = batch["mask"]
            eff_p.append(out["effect_pred"][m].cpu().numpy())
            eff_t.append(batch["effects"][m].cpu().numpy())
            eff_ids.append(batch["sample_ids"][m].cpu().numpy())
            gain_p.append(out["gain_pred"].cpu().numpy())
            gain_t.append(batch["gain"].cpu().numpy())
            gain_ckpt.append(batch["ckpt_index"].cpu().numpy())
            z_norms.append(out["z"][m].norm(dim=-1).cpu().numpy())

        eff_p, eff_t = np.concatenate(eff_p), np.concatenate(eff_t)
        eff_ids = np.concatenate(eff_ids)
        gain_p, gain_t = np.concatenate(gain_p), np.concatenate(gain_t)
        gain_ckpt = np.concatenate(gain_ckpt)

        out = {}
        out.update(regression_metrics(eff_p, eff_t, prefix="effect_"))
        out.update(regression_metrics(gain_p, gain_t, prefix="gain_"))
        # the composition-dependent part of the gain, with the checkpoint's
        # mean removed; this is what a set-level utility model has to win on
        out.update(within_group_metrics(gain_p, gain_t, gain_ckpt, prefix="gain_within_"))
        out.update(
            {
                f"gain_var_{k}": v
                for k, v in group_variance_decomposition(gain_t, gain_ckpt).items()
            }
        )
        out.update(constant_baseline_metrics(eff_t, prefix="effect_const_"))
        out.update(
            per_sample_scalar_baseline_metrics(
                eff_ids, eff_t, prefix="effect_scalar_", fit=self._scalar_fit
            )
        )
        out.update(
            per_sample_scalar_baseline_metrics(
                eff_ids, eff_t, prefix="effect_scalar_hindsight_"
            )
        )
        out["latent_norm_mean"] = float(np.concatenate(z_norms).mean())
        # headline comparisons: >1 means the contextual model wins
        if out["effect_mse"] > 0:
            out["effect_gain_over_constant"] = float(out["effect_const_mse"] / out["effect_mse"])
            out["effect_gain_over_scalar"] = float(out["effect_scalar_mse"] / out["effect_mse"])
        return out


def train_datamodel(
    model: LDVADataModel,
    train_ds: ContextDataset,
    val_ds: ContextDataset | None = None,
    cfg: TrainConfig | None = None,
    effect_table: EffectProfileTable | None = None,
    save_path: str | Path | None = None,
) -> tuple[LDVADataModel, list[dict]]:
    trainer = DataModelTrainer(model, train_ds, val_ds, cfg, effect_table)
    history = trainer.fit()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.model.save(save_path)
    return trainer.model, history
