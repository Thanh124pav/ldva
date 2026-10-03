"""Sample encoder E_phi(x, theta_context) -> z (PLAN.md 4.1; SETUP.md 13).

Deliberately *not* given the acquisition metadata. If the encoder saw `m`, the
effect latent would collapse onto metadata geometry, and PLAN.md 5.5 is explicit
that the two geometries must not be forced to coincide - metadata enters only
later, through the local directional model in `acquisition/metadata_mapper.py`.
`use_metadata=True` exists so the ablation can be run, not as a default.

The policy context matters because the same chunk has different effects at
different checkpoints; `encode` therefore always takes an explicit context, and
`PLAN.md 7` uses a fixed `theta_ref` when producing latents for clustering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ldva.models.common import FiLM, mlp


@dataclass
class SampleEncoderConfig:
    obs_dim: int = 6
    act_dim: int = 2
    chunk_len: int = 8
    latent_dim: int = 32
    hidden: Sequence[int] = field(default_factory=lambda: (128, 128))
    backbone: str = "mlp"  # "mlp" | "gru" | "transformer"
    gru_hidden: int = 64
    n_heads: int = 4
    n_layers: int = 2
    policy_cond: str = "concat"  # "none" | "concat" | "film"
    policy_feat_dim: int = 4
    n_checkpoints: int = 0  # >0 adds a learned checkpoint embedding
    ckpt_embed_dim: int = 8
    use_metadata: bool = False
    meta_dim: int = 0
    final_norm: str = "none"  # "none" | "layernorm" | "l2"
    activation: str = "relu"
    dropout: float = 0.0


class SampleEncoder(nn.Module):
    def __init__(self, cfg: SampleEncoderConfig):
        super().__init__()
        self.cfg = cfg
        step_dim = cfg.obs_dim + cfg.act_dim

        if cfg.backbone == "mlp":
            self.backbone = mlp(
                step_dim * cfg.chunk_len,
                cfg.hidden,
                cfg.hidden[-1],
                activation=cfg.activation,
                dropout=cfg.dropout,
                final_activation=True,
            )
            feat_dim = cfg.hidden[-1]
        elif cfg.backbone == "gru":
            self.backbone = nn.GRU(
                step_dim, cfg.gru_hidden, num_layers=cfg.n_layers, batch_first=True
            )
            feat_dim = cfg.gru_hidden
        elif cfg.backbone == "transformer":
            self.in_proj = nn.Linear(step_dim, cfg.gru_hidden)
            self.pos = nn.Parameter(torch.zeros(1, cfg.chunk_len, cfg.gru_hidden))
            layer = nn.TransformerEncoderLayer(
                cfg.gru_hidden,
                cfg.n_heads,
                dim_feedforward=4 * cfg.gru_hidden,
                batch_first=True,
                dropout=cfg.dropout,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            feat_dim = cfg.gru_hidden
        else:
            raise ValueError(f"unknown backbone {cfg.backbone!r}")

        self.ckpt_embed = (
            nn.Embedding(cfg.n_checkpoints, cfg.ckpt_embed_dim)
            if cfg.n_checkpoints > 0
            else None
        )
        cond_dim = cfg.policy_feat_dim + (cfg.ckpt_embed_dim if self.ckpt_embed else 0)
        self.cond_dim = cond_dim if cfg.policy_cond != "none" else 0

        head_in = feat_dim
        if cfg.policy_cond == "concat":
            head_in += self.cond_dim
        elif cfg.policy_cond == "film":
            self.film = FiLM(feat_dim, max(cond_dim, 1))
        if cfg.use_metadata:
            head_in += cfg.meta_dim

        self.head = mlp(head_in, cfg.hidden, cfg.latent_dim, activation=cfg.activation)
        self.final_norm = (
            nn.LayerNorm(cfg.latent_dim) if cfg.final_norm == "layernorm" else None
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    def _policy_context(
        self, policy_features: torch.Tensor | None, ckpt_index: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.cfg.policy_cond == "none":
            return None
        parts = []
        if policy_features is not None and self.cfg.policy_feat_dim > 0:
            parts.append(policy_features)
        if self.ckpt_embed is not None and ckpt_index is not None:
            parts.append(self.ckpt_embed(ckpt_index))
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        policy_features: torch.Tensor | None = None,
        ckpt_index: torch.Tensor | None = None,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`obs`/`act` are (..., chunk_len, dim); leading dims are flattened.

        Accepts either (n, T, d) or (n_contexts, n_samples, T, d), so the same
        module serves per-sample encoding and whole-context encoding.
        """
        lead = obs.shape[:-2]
        T = obs.shape[-2]
        x = torch.cat([obs, act], dim=-1).reshape(-1, T, obs.shape[-1] + act.shape[-1])

        if self.cfg.backbone == "mlp":
            feat = self.backbone(x.reshape(x.shape[0], -1))
        elif self.cfg.backbone == "gru":
            out, _ = self.backbone(x)
            feat = out[:, -1]
        else:
            feat = self.backbone(self.in_proj(x) + self.pos).mean(dim=1)

        cond = self._policy_context(policy_features, ckpt_index)
        if cond is not None:
            # broadcast one context vector per leading group over its samples
            cond = _broadcast_to_lead(cond, lead)
            if self.cfg.policy_cond == "concat":
                feat = torch.cat([feat, cond], dim=-1)
            else:
                feat = self.film(feat, cond)

        if self.cfg.use_metadata:
            if meta is None:
                raise ValueError("use_metadata=True but no meta tensor was passed")
            feat = torch.cat([feat, meta.reshape(-1, meta.shape[-1])], dim=-1)

        z = self.head(feat)
        if self.final_norm is not None:
            z = self.final_norm(z)
        elif self.cfg.final_norm == "l2":
            z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return z.reshape(*lead, self.cfg.latent_dim)


def _broadcast_to_lead(cond: torch.Tensor, lead: torch.Size) -> torch.Tensor:
    """Expand a per-context condition vector to one row per flattened sample."""
    if len(lead) == 1:
        if cond.shape[0] != lead[0]:
            raise ValueError(
                f"policy context has {cond.shape[0]} rows but {lead[0]} samples"
            )
        return cond
    n_ctx, n_samp = lead[0], int(torch.tensor(lead[1:]).prod().item())
    if cond.shape[0] != n_ctx:
        raise ValueError(
            f"policy context has {cond.shape[0]} rows but {n_ctx} contexts"
        )
    return cond.unsqueeze(1).expand(n_ctx, n_samp, cond.shape[-1]).reshape(-1, cond.shape[-1])
