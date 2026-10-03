"""Realized acquisition gain for any environment (PLAN.md 17.2).

The planner predicts `V_hat(n)` before anything is collected. This module
actually collects an allocation and measures what it does to the policy, so the
prediction can be scored instead of trusted:

    realized gain = U(Update(theta, collected(Q))) - U(theta)

It is environment-agnostic - it only needs `adapter.collect` - so the same
calibration runs on the synthetic world, DMC and MetaWorld. `n_repeats` averages
over whatever stochasticity the environment has; for a pinned reset state and a
scripted expert that is zero, and one repeat suffices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ldva.acquisition.metadata_mapper import MetadataPlan
from ldva.envs.base import EnvAdapter
from ldva.policy.bc import MLPPolicy
from ldva.supervision.bc_task import BCSupervisionTask
from ldva.supervision.leave_one_out import LeaveOneOutEstimator


@dataclass
class OracleConfig:
    lr: float = 0.3
    n_steps: int = 4
    n_repeats: int = 1
    seed: int = 0


class AcquisitionOracle:
    def __init__(
        self,
        adapter: EnvAdapter,
        policy: MLPPolicy,
        val_obs: torch.Tensor,
        val_act: torch.Tensor,
        cfg: OracleConfig | None = None,
    ):
        self.adapter = adapter
        self.policy = policy
        self.val_obs = val_obs
        self.val_act = val_act
        self.cfg = cfg or OracleConfig()
        self.estimator = LeaveOneOutEstimator(lr=self.cfg.lr, n_steps=self.cfg.n_steps)

    def realized_batch_gain(
        self,
        metadata: np.ndarray,
        checkpoint_flat: np.ndarray | None = None,
        policy_features: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Collect at `metadata`, then measure the gain of training on it."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        gains, sizes = [], []
        for _ in range(self.cfg.n_repeats):
            store = self.adapter.collect(np.atleast_2d(metadata), rng, round_id=1)
            if len(store) == 0:
                continue
            task = BCSupervisionTask(
                self.policy, store, self.val_obs, self.val_act,
                policy_features=policy_features)
            if checkpoint_flat is not None:
                task.set_checkpoint(
                    "oracle", checkpoint_flat,
                    np.zeros(4) if policy_features is None else policy_features)
            gains.append(self.estimator.batch_gain(task, np.arange(len(store))))
            sizes.append(len(store))
        if not gains:
            return {"realized_gain": 0.0, "realized_gain_std": 0.0, "n_chunks": 0}
        return {
            "realized_gain": float(np.mean(gains)),
            "realized_gain_std": float(np.std(gains)),
            "n_requests": int(len(np.atleast_2d(metadata))),
            "n_chunks": int(np.mean(sizes)),
            "n_repeats": len(gains),
        }

    def realized_allocation_gain(
        self,
        plans: list[MetadataPlan],
        checkpoint_flat: np.ndarray | None = None,
        policy_features: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        if not plans:
            return {"realized_gain": 0.0, "realized_gain_std": 0.0, "n_chunks": 0}
        metadata = np.concatenate([p.metadata for p in plans], axis=0)
        out = self.realized_batch_gain(metadata, checkpoint_flat, policy_features, rng)
        out["n_directions_used"] = len(plans)
        return out
