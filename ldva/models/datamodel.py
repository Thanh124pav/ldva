"""The LDVA data model: encoder + context encoder + readout + utility.

One object owns the four modules of PLAN.md 4 plus the shared policy-context
encoder, so that "the policy context" is computed in exactly one place and the
encoder, readout and utility model all condition on the same vector.

Two entry points matter downstream and are deliberately separate:

- `forward(batch)` trains on real chunks from `ContextDataset`.
- `utility_from_latents(Z, ...)` scores *hypothetical* latents that have no
  chunks at all. That is what the acquisition planner calls, and it is why the
  utility model is defined over latents rather than over raw data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ldva.data.samples import SampleStore
from ldva.models.batch_utility import BatchUtilityConfig, build_batch_utility
from ldva.models.common import mlp
from ldva.models.context_encoder import ContextEncoderConfig, build_context_encoder
from ldva.models.effect_readout import EffectReadoutConfig, build_effect_readout
from ldva.models.metadata_direction import MetadataDirectionConfig, MetadataDirectionModel
from ldva.models.sample_encoder import SampleEncoder, SampleEncoderConfig


@dataclass
class PolicyContextConfig:
    policy_feat_dim: int = 4
    n_checkpoints: int = 0
    embed_dim: int = 8
    out_dim: int = 16
    enabled: bool = True


class PolicyContextEncoder(nn.Module):
    """Numeric checkpoint features (+ optional embedding) -> policy context.

    `n_checkpoints=0` - the main-result setting of PLAN.md 4.2 - means the
    context is the continuous features alone, so the model has no way to
    memorize which checkpoint it saw and must generalize to an unseen one.
    An embedding is built only for the checkpoint-ID ablation of PLAN.md 12.1.

    When an embedding *is* built, a missing `ckpt_index` contributes zeros
    rather than raising. An unseen future checkpoint has no vocabulary index by
    definition, and `in_dim` is fixed at construction, so without this the
    ablated model could not be evaluated on held-out checkpoints at all - the
    very comparison the ablation exists to make. Zeros, not a learned "unknown"
    row: an unknown row would never appear in training and would stay at its
    random initialization, injecting a fixed random vector into the context.
    """

    def __init__(self, cfg: PolicyContextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = (
            nn.Embedding(cfg.n_checkpoints, cfg.embed_dim) if cfg.n_checkpoints > 0 else None
        )
        in_dim = cfg.policy_feat_dim + (cfg.embed_dim if self.embed else 0)
        self.net = mlp(max(in_dim, 1), [max(2 * cfg.out_dim, 8)], cfg.out_dim)

    @property
    def out_dim(self) -> int:
        return self.cfg.out_dim if self.cfg.enabled else 0

    @property
    def uses_ckpt_id(self) -> bool:
        return self.embed is not None

    def forward(
        self, policy_features: torch.Tensor, ckpt_index: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self.cfg.enabled:
            return None
        parts = []
        if self.cfg.policy_feat_dim > 0:
            parts.append(policy_features[..., : self.cfg.policy_feat_dim])
        if self.embed is not None:
            if ckpt_index is None:
                parts.append(
                    torch.zeros(
                        (*policy_features.shape[:-1], self.cfg.embed_dim),
                        device=policy_features.device,
                        dtype=policy_features.dtype,
                    )
                )
            else:
                parts.append(self.embed(ckpt_index))
        if not parts:
            parts = [torch.zeros(policy_features.shape[0], 1, device=policy_features.device)]
        return self.net(torch.cat(parts, dim=-1))


@dataclass
class LDVAConfig:
    latent_dim: int = 32
    encoder: SampleEncoderConfig = field(default_factory=SampleEncoderConfig)
    context: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)
    readout: EffectReadoutConfig = field(default_factory=EffectReadoutConfig)
    utility: BatchUtilityConfig = field(default_factory=BatchUtilityConfig)
    policy_context: PolicyContextConfig = field(default_factory=PolicyContextConfig)
    metadata_direction: MetadataDirectionConfig | None = None
    context_encoder_kind: str = "deepsets"
    readout_kind: str = "contextual"
    utility_kind: str = "deepsets"
    #: summarize the owned dataset D_t and feed it to the utility model
    use_dataset_context: bool = True

    @classmethod
    def build(
        cls,
        obs_dim: int,
        act_dim: int,
        chunk_len: int,
        meta_dim: int,
        policy_feat_dim: int = 4,
        n_checkpoints: int = 0,
        latent_dim: int = 32,
        hidden: tuple[int, ...] = (128, 128),
        backbone: str = "mlp",
        context_out_dim: int = 64,
        policy_ctx_dim: int = 16,
        use_policy_context: bool = True,
        use_dataset_context: bool = True,
        readout_kind: str = "contextual",
        utility_kind: str = "deepsets",
        context_encoder_kind: str = "deepsets",
        use_metadata_in_encoder: bool = False,
        use_context_in_readout: bool = True,
        learn_metadata_direction: bool = False,
    ) -> "LDVAConfig":
        """Derive a consistent config from dataset shapes.

        Keeping the cross-module dimension bookkeeping here avoids the usual
        failure where the readout is told a context width the encoder does not
        produce.
        """
        pctx = PolicyContextConfig(
            policy_feat_dim=policy_feat_dim,
            n_checkpoints=n_checkpoints,
            out_dim=policy_ctx_dim,
            enabled=use_policy_context,
        )
        eff_policy_dim = policy_ctx_dim if use_policy_context else 0
        # the dataset summary is [mean z, mean z^2] over the owned dataset
        dataset_dim = 2 * latent_dim if use_dataset_context else 0
        return cls(
            latent_dim=latent_dim,
            encoder=SampleEncoderConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                chunk_len=chunk_len,
                latent_dim=latent_dim,
                hidden=hidden,
                backbone=backbone,
                policy_cond="concat" if use_policy_context else "none",
                policy_feat_dim=eff_policy_dim,
                n_checkpoints=0,  # the shared policy encoder owns the embedding
                use_metadata=use_metadata_in_encoder,
                meta_dim=meta_dim,
            ),
            context=ContextEncoderConfig(
                latent_dim=latent_dim, hidden=hidden, out_dim=context_out_dim
            ),
            readout=EffectReadoutConfig(
                latent_dim=latent_dim,
                context_dim=context_out_dim,
                policy_dim=eff_policy_dim,
                hidden=hidden,
                use_context=use_context_in_readout,
            ),
            utility=BatchUtilityConfig(
                latent_dim=latent_dim,
                hidden=hidden,
                policy_dim=eff_policy_dim,
                dataset_dim=dataset_dim,
            ),
            policy_context=pctx,
            metadata_direction=(
                MetadataDirectionConfig(latent_dim=latent_dim, meta_dim=meta_dim, hidden=hidden)
                if learn_metadata_direction
                else None
            ),
            context_encoder_kind=context_encoder_kind,
            readout_kind=readout_kind,
            utility_kind=utility_kind,
            use_dataset_context=use_dataset_context,
        )


class LDVADataModel(nn.Module):
    def __init__(self, cfg: LDVAConfig):
        super().__init__()
        self.cfg = cfg
        self.policy_encoder = PolicyContextEncoder(cfg.policy_context)
        self.encoder = SampleEncoder(cfg.encoder)
        self.context_encoder = build_context_encoder(cfg.context, cfg.context_encoder_kind)
        self.readout = build_effect_readout(cfg.readout, cfg.readout_kind)
        self.batch_utility = build_batch_utility(cfg.utility, cfg.utility_kind)
        self.metadata_direction = (
            MetadataDirectionModel(cfg.metadata_direction)
            if cfg.metadata_direction is not None
            else None
        )
        #: cached summary of the owned dataset D_t, refreshed between rounds
        self.register_buffer(
            "dataset_context",
            torch.zeros(cfg.utility.dataset_dim),
            persistent=True,
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---- encoding -------------------------------------------------------
    def policy_context(self, batch: dict) -> torch.Tensor | None:
        return self.policy_encoder(batch["policy_features"], batch.get("ckpt_index"))

    def encode(self, batch: dict, policy_ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a padded context batch -> (n_contexts, n_samples, latent_dim)."""
        if policy_ctx is None:
            policy_ctx = self.policy_context(batch)
        return self.encoder(
            batch["obs"],
            batch["act"],
            policy_features=policy_ctx,
            meta=batch.get("meta"),
        )

    @torch.no_grad()
    def encode_store(
        self,
        store: SampleStore,
        policy_features: np.ndarray,
        ckpt_index: int | None = None,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Encode a whole `SampleStore` at one reference checkpoint (PLAN.md 7).

        Clustering and direction generation operate on latents from a *single*
        `theta_ref`, so the geometry they see is not smeared across checkpoints.
        """
        self.eval()
        dev = self.device
        pf = torch.as_tensor(np.asarray(policy_features, dtype=np.float32), device=dev)
        out = []
        for lo in range(0, len(store), batch_size):
            hi = min(lo + batch_size, len(store))
            obs = torch.from_numpy(store.obs[lo:hi]).to(dev)
            act = torch.from_numpy(store.act[lo:hi]).to(dev)
            meta = torch.from_numpy(
                store.metadata_norm()[lo:hi].astype(np.float32)
            ).to(dev)
            n = hi - lo
            idx = (
                None
                if ckpt_index is None
                else torch.full((n,), ckpt_index, dtype=torch.long, device=dev)
            )
            pctx = self.policy_encoder(pf.unsqueeze(0).expand(n, -1), idx)
            z = self.encoder(obs, act, policy_features=pctx, meta=meta)
            out.append(z.cpu().numpy())
        return np.concatenate(out, axis=0)

    # ---- dataset context -------------------------------------------------
    def summarize_dataset(self, latents: np.ndarray | torch.Tensor) -> torch.Tensor:
        """[mean z, mean z^2] over the owned dataset D_t."""
        z = torch.as_tensor(np.asarray(latents), dtype=torch.float32, device=self.device)
        return torch.cat([z.mean(0), (z * z).mean(0)])

    def set_dataset_context(self, latents: np.ndarray | torch.Tensor) -> None:
        if self.cfg.utility.dataset_dim == 0:
            return
        s = self.summarize_dataset(latents)
        if s.numel() != self.cfg.utility.dataset_dim:
            raise ValueError(
                f"dataset context has {s.numel()} dims, model expects "
                f"{self.cfg.utility.dataset_dim}"
            )
        self.dataset_context.copy_(s)

    def _dataset_context(self, n: int) -> torch.Tensor | None:
        if self.cfg.utility.dataset_dim == 0:
            return None
        return self.dataset_context.unsqueeze(0).expand(n, -1)

    # ---- forward ---------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """Full training forward pass over a padded context batch."""
        policy_ctx = self.policy_context(batch)
        z = self.encode(batch, policy_ctx)
        mask = batch["mask"]
        h_loo = self.context_encoder.leave_one_out(z, mask)
        h_full = self.context_encoder(z, mask)
        s_hat = self.readout(z, h_loo, policy_ctx)
        v_hat = self.batch_utility(
            z, mask, policy_ctx, self._dataset_context(z.shape[0])
        )
        return {
            "z": z,
            "h_loo": h_loo,
            "h_full": h_full,
            "effect_pred": s_hat,
            "gain_pred": v_hat,
            "policy_ctx": policy_ctx,
            "mask": mask,
        }

    # ---- planner interface ------------------------------------------------
    def utility_from_latents(
        self,
        z: torch.Tensor | np.ndarray,
        policy_features: np.ndarray | torch.Tensor | None = None,
        ckpt_index: int | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score hypothetical latent batches -> (n_batches,).

        `z` is (n_batches, n_samples, latent_dim). No chunks are required, which
        is what lets the planner evaluate data that does not exist yet.
        """
        dev = self.device
        z = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev)
        if z.ndim == 2:
            z = z.unsqueeze(0)
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=dev)
        policy_ctx = None
        if self.cfg.utility.policy_dim > 0:
            if policy_features is None:
                raise ValueError("this model conditions on the policy context")
            pf = torch.as_tensor(
                np.asarray(policy_features, dtype=np.float32), device=dev
            )
            if pf.ndim == 1:
                pf = pf.unsqueeze(0).expand(z.shape[0], -1)
            idx = (
                None
                if ckpt_index is None
                else torch.full((z.shape[0],), ckpt_index, dtype=torch.long, device=dev)
            )
            policy_ctx = self.policy_encoder(pf, idx)
        return self.batch_utility(
            z, mask, policy_ctx, self._dataset_context(z.shape[0])
        )

    def effect_from_latents(
        self,
        z: torch.Tensor | np.ndarray,
        policy_features: np.ndarray | torch.Tensor | None = None,
        ckpt_index: int | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Contextual effects for hypothetical latent batches -> (n_batches, n)."""
        dev = self.device
        z = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev)
        if z.ndim == 2:
            z = z.unsqueeze(0)
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=dev)
        policy_ctx = None
        if self.cfg.readout.policy_dim > 0:
            pf = torch.as_tensor(np.asarray(policy_features, dtype=np.float32), device=dev)
            if pf.ndim == 1:
                pf = pf.unsqueeze(0).expand(z.shape[0], -1)
            idx = (
                None
                if ckpt_index is None
                else torch.full((z.shape[0],), ckpt_index, dtype=torch.long, device=dev)
            )
            policy_ctx = self.policy_encoder(pf, idx)
        h_loo = self.context_encoder.leave_one_out(z, mask)
        return self.readout(z, h_loo, policy_ctx)

    # ---- io ----------------------------------------------------------------
    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.cfg}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "LDVADataModel":
        d = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(d["config"])
        model.load_state_dict(d["state_dict"])
        return model
