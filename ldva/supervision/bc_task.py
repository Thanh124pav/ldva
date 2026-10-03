"""`SupervisionTask` for behaviour cloning.

This is the single adapter the effect estimators talk to, so the same
gradient-alignment / influence / leave-one-out code runs on the Stage 0 linear
policy and on a PushT or MetaWorld BC policy with no changes. The only
environment-specific inputs are a `SampleStore` and a fixed validation set.

`utility` is `-validation BC loss`, so larger is better and
`U(Update(theta, B)) - U(theta)` is a gain, matching the sign convention in
`supervision/base.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ldva.data.samples import SampleStore
from ldva.policy.bc import MLPPolicy


class BCSupervisionTask:
    def __init__(
        self,
        policy: MLPPolicy,
        store: SampleStore,
        val_obs: torch.Tensor,
        val_act: torch.Tensor,
        checkpoint_id: str = "ckpt",
        policy_features: np.ndarray | None = None,
        device: torch.device | str = "cpu",
    ):
        self.policy = policy.to(device)
        self.store = store
        self.device = torch.device(device)
        self._obs = torch.from_numpy(store.obs).to(self.device)
        self._act = torch.from_numpy(store.act).to(self.device)
        self._val_obs = val_obs.to(self.device)
        self._val_act = val_act.to(self.device)
        self._ckpt_id = checkpoint_id
        self._policy_features = (
            np.zeros(4, dtype=np.float32)
            if policy_features is None
            else np.asarray(policy_features, dtype=np.float32)
        )

    # ---- SupervisionTask protocol --------------------------------------
    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return list(self.policy.parameters())

    def train_loss(self, sample_ids: np.ndarray) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(sample_ids, dtype=np.int64), device=self.device)
        return self.policy.bc_loss(self._obs[idx], self._act[idx])

    def val_loss(self, params: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        if params is None:
            return self.policy.bc_loss(self._val_obs, self._val_act)
        with _temporary_params(self.policy, params):
            return self.policy.bc_loss(self._val_obs, self._val_act)

    def utility(self, params: Sequence[torch.Tensor] | None = None) -> float:
        with torch.no_grad():
            return float(-self.val_loss(params).item())

    @property
    def checkpoint_id(self) -> str:
        return self._ckpt_id

    def policy_features(self) -> np.ndarray:
        return self._policy_features

    # ---- convenience ----------------------------------------------------
    def set_checkpoint(self, ckpt_id: str, flat_params, features: np.ndarray) -> None:
        """Point the task at a different checkpoint, reusing all loaded tensors."""
        self.policy.load_flat_params(flat_params)
        self._ckpt_id = ckpt_id
        self._policy_features = np.asarray(features, dtype=np.float32)

    def set_store(self, store: SampleStore) -> None:
        """Swap in a grown dataset between acquisition rounds."""
        self.store = store
        self._obs = torch.from_numpy(store.obs).to(self.device)
        self._act = torch.from_numpy(store.act).to(self.device)


class _temporary_params:
    """Write `params` into a module for the duration of a `with` block."""

    def __init__(self, module: torch.nn.Module, params: Sequence[torch.Tensor]):
        self.module = module
        self.params = list(params)
        self.backup: list[torch.Tensor] = []

    def __enter__(self):
        live = list(self.module.parameters())
        if len(live) != len(self.params):
            raise ValueError(
                f"expected {len(live)} parameter tensors, got {len(self.params)}"
            )
        self.backup = [p.detach().clone() for p in live]
        with torch.no_grad():
            for p, new in zip(live, self.params):
                p.copy_(new)
        return self.module

    def __exit__(self, *exc):
        with torch.no_grad():
            for p, old in zip(self.module.parameters(), self.backup):
                p.copy_(old)
        return False
