"""Policy checkpoints as first-class objects (PLAN.md 6; SETUP.md 11).

Effect labels are only meaningful relative to a checkpoint, so every checkpoint
is stored together with the numeric features that the data model uses as its
policy context. Checkpoint *diversity* is a supervision-quality metric in its
own right: if all contexts come from one theta, the model cannot learn a
policy-conditioned readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class Checkpoint:
    ckpt_id: str
    flat_params: np.ndarray
    step: int
    #: numeric policy context: [step_frac, train_loss, val_loss, grad_norm]
    features: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.flat_params = np.asarray(self.flat_params, dtype=np.float32)
        self.features = np.asarray(self.features, dtype=np.float32)


class CheckpointStore:
    FEATURE_NAMES = ("step_frac", "train_loss", "val_loss", "grad_norm")

    def __init__(self, checkpoints: list[Checkpoint] | None = None):
        self.checkpoints: list[Checkpoint] = list(checkpoints or [])

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __iter__(self):
        return iter(self.checkpoints)

    def __getitem__(self, i: int) -> Checkpoint:
        return self.checkpoints[i]

    def add(self, ckpt: Checkpoint) -> None:
        self.checkpoints.append(ckpt)

    def by_id(self, ckpt_id: str) -> Checkpoint:
        for c in self.checkpoints:
            if c.ckpt_id == ckpt_id:
                return c
        raise KeyError(ckpt_id)

    @property
    def ids(self) -> list[str]:
        return [c.ckpt_id for c in self.checkpoints]

    def feature_matrix(self) -> np.ndarray:
        if not self.checkpoints:
            return np.zeros((0, 0), dtype=np.float32)
        d = max(c.features.shape[0] for c in self.checkpoints)
        out = np.zeros((len(self.checkpoints), d), dtype=np.float32)
        for i, c in enumerate(self.checkpoints):
            out[i, : c.features.shape[0]] = c.features
        return out

    def diversity_report(self) -> dict:
        """How different are these checkpoints?

        Near-zero parameter spread means the contexts are effectively one
        policy, and any "policy-conditioned" claim would be unsupported.
        """
        if len(self.checkpoints) < 2:
            return {"n_checkpoints": len(self.checkpoints), "param_spread": 0.0}
        P = np.stack([c.flat_params for c in self.checkpoints])
        centred = P - P.mean(0, keepdims=True)
        norms = np.linalg.norm(centred, axis=1)
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        iu = np.triu_indices(len(P), k=1)
        return {
            "n_checkpoints": len(P),
            "param_spread": float(norms.mean()),
            "pairwise_dist_mean": float(d[iu].mean()),
            "pairwise_dist_min": float(d[iu].min()),
            "param_norm_mean": float(np.linalg.norm(P, axis=1).mean()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ckpt_id": [c.ckpt_id for c in self.checkpoints],
                "flat_params": [c.flat_params for c in self.checkpoints],
                "step": [c.step for c in self.checkpoints],
                "features": [c.features for c in self.checkpoints],
                "info": [c.info for c in self.checkpoints],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointStore":
        d = torch.load(path, weights_only=False)
        return cls(
            [
                Checkpoint(
                    ckpt_id=i, flat_params=p, step=s, features=f, info=n
                )
                for i, p, s, f, n in zip(
                    d["ckpt_id"], d["flat_params"], d["step"], d["features"], d["info"]
                )
            ]
        )
