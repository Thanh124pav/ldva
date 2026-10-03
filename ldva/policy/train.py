"""BC training that *emits checkpoints* (PLAN.md 6 step 1).

Policy training is not the contribution; its job here is to produce a diverse
trajectory of checkpoints `theta_1..theta_T` for multi-context supervision. Two
extra knobs exist for that purpose only:

- `n_restarts`: independent runs from different inits, so checkpoints are not
  all on one optimization path (which would make "policy context" a proxy for
  "training step").
- `snapshot_every`: how densely we sample along each path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ldva.data.samples import SampleStore
from ldva.policy.bc import MLPPolicy
from ldva.policy.checkpoints import Checkpoint, CheckpointStore


@dataclass
class BCTrainConfig:
    steps: int = 300
    batch_size: int = 64
    lr: float = 1e-2
    snapshot_every: int = 50
    n_restarts: int = 2
    #: deliberate init perturbation, which spreads the checkpoint cloud so the
    #: supervision contexts are not all one optimization path. `None` leaves
    #: the module's own initialization alone, which is what an *evaluation*
    #: policy wants: perturbing the init of a policy whose rollout return is
    #: the headline metric trades away the thing being measured.
    init_scale: float | None = 0.5
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    #: skip the first snapshot of each run if you do not want random-init contexts
    include_init: bool = True


def _make_optimizer(policy: MLPPolicy, cfg: BCTrainConfig):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def train_bc(
    store: SampleStore,
    val_obs: torch.Tensor,
    val_act: torch.Tensor,
    cfg: BCTrainConfig | None = None,
    hidden: tuple[int, ...] = (),
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[MLPPolicy, CheckpointStore]:
    """Train BC policies and return the final policy plus all checkpoints."""
    cfg = cfg or BCTrainConfig()
    device = torch.device(device)
    obs = torch.from_numpy(store.obs).to(device)
    act = torch.from_numpy(store.act).to(device)
    val_obs = val_obs.to(device)
    val_act = val_act.to(device)

    ckpts = CheckpointStore()
    policy = None
    for restart in range(max(1, cfg.n_restarts)):
        torch.manual_seed(seed + 7919 * restart)
        policy = MLPPolicy(store.obs_dim, store.act_dim, hidden=hidden).to(device)
        policy.fit_obs_normalizer(obs)
        if cfg.init_scale is not None:
            with torch.no_grad():
                # Spread the checkpoint cloud, but *relative to each tensor's own
                # scale*. An absolute perturbation is fine for a linear policy and
                # destroys a deep one: adding N(0, 0.5) to a 128-wide layer whose
                # init std is ~0.09 is a 5-sigma kick that SGD cannot recover from.
                for p in policy.parameters():
                    scale = (
                        p.detach().std().clamp(min=1e-8)
                        if p.numel() > 1
                        else torch.tensor(1.0)
                    )
                    p.mul_(cfg.init_scale).add_(cfg.init_scale * scale * torch.randn_like(p))
        opt = _make_optimizer(policy, cfg)
        rng = np.random.default_rng(seed + 104729 * restart)

        for step in range(cfg.steps + 1):
            take_snapshot = step % cfg.snapshot_every == 0 and (
                cfg.include_init or step > 0
            )
            if take_snapshot:
                ckpts.add(
                    _snapshot(
                        policy,
                        obs,
                        act,
                        val_obs,
                        val_act,
                        step=step,
                        total=cfg.steps,
                        ckpt_id=f"r{restart}_s{step}",
                        rng=rng,
                    )
                )
            if step == cfg.steps:
                break
            idx = rng.choice(len(store), size=min(cfg.batch_size, len(store)), replace=False)
            loss = policy.bc_loss(obs[idx], act[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    assert policy is not None
    return policy, ckpts


def _snapshot(
    policy: MLPPolicy,
    obs: torch.Tensor,
    act: torch.Tensor,
    val_obs: torch.Tensor,
    val_act: torch.Tensor,
    step: int,
    total: int,
    ckpt_id: str,
    rng: np.random.Generator,
) -> Checkpoint:
    sub = rng.choice(obs.shape[0], size=min(256, obs.shape[0]), replace=False)
    train_loss = policy.bc_loss(obs[sub], act[sub])
    grads = torch.autograd.grad(train_loss, list(policy.parameters()), allow_unused=True)
    gnorm = float(
        torch.sqrt(sum((g**2).sum() for g in grads if g is not None)).item()
    )
    with torch.no_grad():
        val_loss = float(policy.bc_loss(val_obs, val_act).item())
    features = np.array(
        [step / max(total, 1), float(train_loss.item()), val_loss, gnorm],
        dtype=np.float32,
    )
    return Checkpoint(
        ckpt_id=ckpt_id,
        flat_params=policy.flat_params().cpu().numpy(),
        step=step,
        features=features,
        info={"train_loss": float(train_loss.item()), "val_loss": val_loss},
    )
