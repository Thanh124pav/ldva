"""Real rollout evaluation of a *learned* policy (PLAN.md 15, P0.1).

BC validation loss is supervision, not robot performance. A policy can shave its
per-chunk action error while still failing the task, and PLAN.md 18 asks for
"rollout success rate" and "return" as the robotics outcome - so the acquisition
curve has to be plotted against something the simulator reports, not against the
regression loss the data model was trained on.

This module holds the two pieces that are the same for every simulator:

- `RolloutMetrics`  what an evaluation reports, with the per-episode arrays kept
                    so a later seed-aggregation can recompute a standard error
                    instead of averaging already-averaged numbers.
- `EvalConditions`  the *fixed* set of initial conditions. PLAN.md 15 requires
                    "fixed evaluation conditions across all methods and rounds";
                    conditions are therefore drawn once from a dedicated RNG and
                    carried by value through the whole run, and `fingerprint()`
                    makes an accidental re-draw visible in the report rather
                    than silently shifting the comparison.

The per-environment part - how to reset to a condition and step the policy -
stays in each adapter, because that is the only part that differs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutMetrics:
    """Outcome of rolling a policy out from a fixed set of conditions."""

    n_episodes: int
    mean_return: float
    std_return: float
    success_rate: float
    mean_length: float
    #: per-episode values, kept for honest cross-seed aggregation
    returns: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @classmethod
    def from_episodes(
        cls,
        returns: list[float],
        successes: list[bool],
        lengths: list[int],
        **info,
    ) -> "RolloutMetrics":
        r = np.asarray(returns, dtype=np.float64)
        s = np.asarray(successes, dtype=bool)
        n = len(r)
        return cls(
            n_episodes=int(n),
            mean_return=float(r.mean()) if n else float("nan"),
            std_return=float(r.std()) if n else float("nan"),
            success_rate=float(s.mean()) if n else float("nan"),
            mean_length=float(np.mean(lengths)) if n else float("nan"),
            returns=[float(x) for x in r],
            successes=[bool(x) for x in s],
            lengths=[int(x) for x in lengths],
            info=dict(info),
        )

    def to_dict(self, with_episodes: bool = False) -> dict:
        d = {
            "n_episodes": self.n_episodes,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "success_rate": self.success_rate,
            "mean_length": self.mean_length,
            **self.info,
        }
        if with_episodes:
            d["returns"] = self.returns
            d["successes"] = self.successes
        return d

    @property
    def standard_error(self) -> float:
        """s.e. of the mean return over episodes."""
        n = max(self.n_episodes, 1)
        return float(self.std_return / np.sqrt(n))


@dataclass
class EvalConditions:
    """Initial conditions for rollout evaluation, drawn once per run.

    `metadata` rows are environment-specific requests in exactly the format
    `EnvAdapter.collect` accepts, so an adapter needs no second code path to
    reset into them. `seed` is recorded so the draw is reproducible, and
    `fingerprint` is checked against the report to catch a re-draw.
    """

    metadata: np.ndarray
    seed: int
    env_name: str = "unknown"

    def __post_init__(self) -> None:
        self.metadata = np.atleast_2d(np.asarray(self.metadata, dtype=np.float64))

    def __len__(self) -> int:
        return int(self.metadata.shape[0])

    def fingerprint(self) -> str:
        """Stable hash of the conditions; identical across methods and rounds."""
        h = hashlib.sha256()
        h.update(self.env_name.encode())
        h.update(np.ascontiguousarray(np.round(self.metadata, 9)).tobytes())
        return h.hexdigest()[:16]

    def subset(self, n: int) -> "EvalConditions":
        """First `n` conditions - a prefix, never a resample, so a cheaper
        evaluation stays a subset of the expensive one."""
        return EvalConditions(self.metadata[:n], self.seed, self.env_name)

    def to_dict(self) -> dict:
        return {
            "n_conditions": len(self),
            "seed": self.seed,
            "env": self.env_name,
            "fingerprint": self.fingerprint(),
        }


class PolicyActor:
    """Wraps a BC policy as a `obs -> action` callable for simulator stepping.

    The policy is chunk-trained but applied per step, which is what the stored
    chunks already assume (`MLPPolicy` maps (..., obs_dim) -> (..., act_dim)).
    Observation normalization lives inside the policy, so nothing here has to
    know about it.
    """

    def __init__(self, policy, device: torch.device | str | None = None):
        self.policy = policy
        self.device = torch.device(device) if device is not None else next(
            policy.parameters()
        ).device
        self._was_training = policy.training
        policy.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        a = self.policy(x.reshape(1, -1)).reshape(-1)
        return a.detach().cpu().numpy().astype(np.float64)

    def close(self) -> None:
        if self._was_training:
            self.policy.train()

    def __enter__(self) -> "PolicyActor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
