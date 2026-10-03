"""Sample units and their storage (SETUP.md 10).

The default sample unit is a *trajectory chunk*: it keeps local temporal context
that an isolated transition loses, while staying far cheaper to label than a
full demonstration. `SampleStore` keeps chunks in dense arrays because the
effect estimators and the data model both want whole batches at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldva.data.metadata import MetadataSpec


@dataclass
class Sample:
    """A single trajectory chunk with the fields SETUP.md 10 requires."""

    sample_id: int
    trajectory_id: int
    start_t: int
    end_t: int
    obs: np.ndarray  # (chunk_len, obs_dim)
    act: np.ndarray  # (chunk_len, act_dim)
    reward: float
    success: bool
    policy_ckpt_id: str
    task_id: int
    metadata: np.ndarray  # (meta_dim,) raw acquisition metadata
    #: acquisition round that produced this chunk (0 = initial dataset)
    round_id: int = 0
    #: monetary cost of acquiring it (SETUP.md 31); 0 for free simulation data
    cost: float = 0.0


class SampleStore:
    """Columnar store of trajectory chunks.

    Arrays are kept 2-D/3-D and contiguous so that `batch_tensors` is a cheap
    gather rather than a per-sample loop.
    """

    def __init__(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        metadata: np.ndarray,
        metadata_spec: MetadataSpec,
        trajectory_id: np.ndarray | None = None,
        start_t: np.ndarray | None = None,
        task_id: np.ndarray | None = None,
        reward: np.ndarray | None = None,
        success: np.ndarray | None = None,
        policy_ckpt_id: np.ndarray | None = None,
        round_id: np.ndarray | None = None,
        cost: np.ndarray | None = None,
        latent_true: np.ndarray | None = None,
    ):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.act = np.asarray(act, dtype=np.float32)
        if self.obs.ndim != 3 or self.act.ndim != 3:
            raise ValueError("obs/act must be (n_samples, chunk_len, dim)")
        if self.obs.shape[:2] != self.act.shape[:2]:
            raise ValueError("obs/act must agree on (n_samples, chunk_len)")
        n = self.obs.shape[0]
        self.metadata = np.asarray(metadata, dtype=np.float64).reshape(n, -1)
        if self.metadata.shape[1] != len(metadata_spec):
            raise ValueError(
                f"metadata has {self.metadata.shape[1]} columns but spec declares "
                f"{len(metadata_spec)}"
            )
        self.metadata_spec = metadata_spec

        self.trajectory_id = _default(trajectory_id, np.arange(n), np.int64)
        self.start_t = _default(start_t, np.zeros(n), np.int64)
        self.task_id = _default(task_id, np.zeros(n), np.int64)
        self.reward = _default(reward, np.zeros(n), np.float32)
        self.success = _default(success, np.zeros(n), bool)
        self.round_id = _default(round_id, np.zeros(n), np.int64)
        self.cost = _default(cost, np.zeros(n), np.float64)
        self.policy_ckpt_id = (
            np.asarray(policy_ckpt_id, dtype=object)
            if policy_ckpt_id is not None
            else np.array(["init"] * n, dtype=object)
        )
        #: ground-truth latent factors, available only in synthetic settings
        self.latent_true = (
            None if latent_true is None else np.asarray(latent_true, dtype=np.float32)
        )

    # ---- shape helpers -------------------------------------------------
    def __len__(self) -> int:
        return self.obs.shape[0]

    @property
    def n_samples(self) -> int:
        return self.obs.shape[0]

    @property
    def chunk_len(self) -> int:
        return self.obs.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[2]

    @property
    def act_dim(self) -> int:
        return self.act.shape[2]

    @property
    def meta_dim(self) -> int:
        return self.metadata.shape[1]

    @property
    def sample_ids(self) -> np.ndarray:
        return np.arange(self.n_samples)

    def metadata_norm(self) -> np.ndarray:
        return self.metadata_spec.normalize(self.metadata)

    # ---- access --------------------------------------------------------
    def get(self, i: int) -> Sample:
        return Sample(
            sample_id=int(i),
            trajectory_id=int(self.trajectory_id[i]),
            start_t=int(self.start_t[i]),
            end_t=int(self.start_t[i]) + self.chunk_len,
            obs=self.obs[i],
            act=self.act[i],
            reward=float(self.reward[i]),
            success=bool(self.success[i]),
            policy_ckpt_id=str(self.policy_ckpt_id[i]),
            task_id=int(self.task_id[i]),
            metadata=self.metadata[i],
            round_id=int(self.round_id[i]),
            cost=float(self.cost[i]),
        )

    def features(self, ids: np.ndarray | None = None) -> np.ndarray:
        """Flat (n, chunk_len * (obs_dim + act_dim)) view used by linear probes."""
        ids = self.sample_ids if ids is None else np.asarray(ids)
        x = np.concatenate([self.obs[ids], self.act[ids]], axis=-1)
        return x.reshape(len(ids), -1)

    def concat(self, other: "SampleStore") -> "SampleStore":
        """Append another store (used by the closed-loop acquisition rounds)."""
        if other.chunk_len != self.chunk_len:
            raise ValueError("cannot concatenate stores with different chunk_len")
        lt = None
        if self.latent_true is not None and other.latent_true is not None:
            lt = np.concatenate([self.latent_true, other.latent_true])
        return SampleStore(
            obs=np.concatenate([self.obs, other.obs]),
            act=np.concatenate([self.act, other.act]),
            metadata=np.concatenate([self.metadata, other.metadata]),
            metadata_spec=self.metadata_spec,
            trajectory_id=np.concatenate([self.trajectory_id, other.trajectory_id]),
            start_t=np.concatenate([self.start_t, other.start_t]),
            task_id=np.concatenate([self.task_id, other.task_id]),
            reward=np.concatenate([self.reward, other.reward]),
            success=np.concatenate([self.success, other.success]),
            policy_ckpt_id=np.concatenate([self.policy_ckpt_id, other.policy_ckpt_id]),
            round_id=np.concatenate([self.round_id, other.round_id]),
            cost=np.concatenate([self.cost, other.cost]),
            latent_true=lt,
        )

    def subset(self, ids: np.ndarray) -> "SampleStore":
        ids = np.asarray(ids)
        return SampleStore(
            obs=self.obs[ids],
            act=self.act[ids],
            metadata=self.metadata[ids],
            metadata_spec=self.metadata_spec,
            trajectory_id=self.trajectory_id[ids],
            start_t=self.start_t[ids],
            task_id=self.task_id[ids],
            reward=self.reward[ids],
            success=self.success[ids],
            policy_ckpt_id=self.policy_ckpt_id[ids],
            round_id=self.round_id[ids],
            cost=self.cost[ids],
            latent_true=None if self.latent_true is None else self.latent_true[ids],
        )

    # ---- io ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        np.savez_compressed(
            path,
            obs=self.obs,
            act=self.act,
            metadata=self.metadata,
            trajectory_id=self.trajectory_id,
            start_t=self.start_t,
            task_id=self.task_id,
            reward=self.reward,
            success=self.success,
            policy_ckpt_id=self.policy_ckpt_id.astype(str),
            round_id=self.round_id,
            cost=self.cost,
            latent_true=(
                np.zeros(0) if self.latent_true is None else self.latent_true
            ),
            has_latent_true=np.array([self.latent_true is not None]),
            metadata_spec=np.array([json.dumps(self.metadata_spec.to_dict())]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SampleStore":
        import json

        d = np.load(path, allow_pickle=True)
        spec = MetadataSpec.from_dict(json.loads(str(d["metadata_spec"][0])))
        return cls(
            obs=d["obs"],
            act=d["act"],
            metadata=d["metadata"],
            metadata_spec=spec,
            trajectory_id=d["trajectory_id"],
            start_t=d["start_t"],
            task_id=d["task_id"],
            reward=d["reward"],
            success=d["success"],
            policy_ckpt_id=d["policy_ckpt_id"],
            round_id=d["round_id"],
            cost=d["cost"],
            latent_true=d["latent_true"] if bool(d["has_latent_true"][0]) else None,
        )


def _default(arr, fallback, dtype):
    return np.asarray(fallback if arr is None else arr, dtype=dtype)
