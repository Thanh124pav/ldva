"""On-disk caches for the expensive parts of a run (PLAN.md 15, P1).

Three things in the loop cost real time and are *identical* across acquisition
methods: the initial dataset D_0, the fixed BC evaluation chunks, and the fixed
rollout evaluation conditions. All three depend only on the environment, its
config and the seed - never on which method is being run - so computing them
once per seed instead of once per (method, seed) pair saves a factor of the
number of methods on simulator time. With the six methods of PLAN.md 17 E1 that
is most of the collection cost.

The key is a hash of the environment name, its config and the seed. A changed
config therefore misses the cache rather than silently returning data collected
under different settings, which is the failure mode that makes caching
dangerous in an experiment.

Correctness notes, because a cache that quietly changes an experiment is worse
than no cache:

- `enabled=False` bypasses everything, so a suspicious result can always be
  re-derived from scratch.
- the cache is keyed on the config *as passed*, so two adapters that differ in
  any declared field get different keys.
- nothing method-dependent or round-dependent is ever cached. Acquired data
  depends on the planner's choices, and caching it would make the methods share
  data they did not choose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ldva.data.samples import SampleStore
from ldva.envs.rollout import EvalConditions
from ldva.utils import config_hash


@dataclass
class RunCache:
    """Cache for per-seed, method-independent environment products."""

    root: Path | None = None
    enabled: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(self.root)
            if self.enabled:
                self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.enabled = False
        self.hits = 0
        self.misses = 0

    def _key(self, kind: str, env_name: str, cfg: dict, seed: int, n: int) -> Path:
        h = config_hash(cfg, env_name, seed, n, kind)
        return self.root / f"{env_name}_{kind}_{h}"

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[cache] {msg}", flush=True)

    # ---- initial dataset --------------------------------------------------
    def initial_dataset(self, env, cfg: dict, seed: int, n: int, rng) -> SampleStore:
        if not self.enabled:
            return env.initial_dataset(n, rng)
        path = self._key("d0", env.name, cfg, seed, n).with_suffix(".npz")
        if path.exists():
            try:
                self.hits += 1
                self._say(f"hit  D_0 {path.name}")
                return SampleStore.load(path)
            except Exception as e:  # a corrupt file must not kill the run
                self._say(f"unreadable ({e}); recollecting")
        self.misses += 1
        store = env.initial_dataset(n, rng)
        try:
            store.save(path)
            self._say(f"miss D_0 -> {path.name}")
        except Exception as e:
            self._say(f"could not write ({e})")
        return store

    # ---- fixed BC evaluation chunks --------------------------------------
    def evaluation_set(self, env, cfg: dict, seed: int, n: int, rng):
        if not self.enabled:
            return env.evaluation_set(n, rng)
        path = self._key("eval", env.name, cfg, seed, n).with_suffix(".pt")
        if path.exists():
            try:
                self.hits += 1
                d = torch.load(path, weights_only=True)
                self._say(f"hit  eval set {path.name}")
                return d["obs"], d["act"]
            except Exception as e:
                self._say(f"unreadable ({e}); regenerating")
        self.misses += 1
        obs, act = env.evaluation_set(n, rng)
        try:
            torch.save({"obs": obs, "act": act}, path)
            self._say(f"miss eval set -> {path.name}")
        except Exception as e:
            self._say(f"could not write ({e})")
        return obs, act

    # ---- fixed rollout conditions ----------------------------------------
    def eval_conditions(self, env, cfg: dict, seed: int, n: int, rng) -> EvalConditions:
        """Cached because PLAN.md 15 requires these to be *identical* across
        methods and rounds; reading them from one file is also the strongest
        guarantee of that, stronger than re-deriving them from a seed."""
        if not self.enabled:
            return env.eval_conditions(n, rng)
        path = self._key("cond", env.name, cfg, seed, n).with_suffix(".npz")
        if path.exists():
            try:
                self.hits += 1
                d = np.load(path, allow_pickle=False)
                self._say(f"hit  conditions {path.name}")
                return EvalConditions(
                    metadata=d["metadata"], seed=int(d["seed"]), env_name=str(env.name)
                )
            except Exception as e:
                self._say(f"unreadable ({e}); redrawing")
        self.misses += 1
        cond = env.eval_conditions(n, rng)
        try:
            np.savez(path, metadata=cond.metadata, seed=cond.seed)
            self._say(f"miss conditions -> {path.name}")
        except Exception as e:
            self._say(f"could not write ({e})")
        return cond

    def report(self) -> dict:
        return {
            "enabled": self.enabled,
            "root": str(self.root) if self.root else None,
            "hits": self.hits,
            "misses": self.misses,
        }
