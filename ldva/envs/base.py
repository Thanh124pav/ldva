"""The one seam between LDVA and an environment (SETUP.md 37).

Everything above this interface - supervision, the data model, clustering,
directions, the planners, the metadata mapper - is environment-agnostic. Moving
up the benchmark ladder means implementing `EnvAdapter` and nothing else:

    Stage 0  synthetic        implemented (`ldva/envs/synthetic`)
    Stage 1  PushT            adapter stub
    Stage 2  MetaWorld        adapter stub
    Stage 3  ManiSkill        adapter stub
    Stage 4  static datasets  adapter stub (collect() is a lookup, not a rollout)
    Stage 5  paid data        adapter stub (collect() issues a purchase request)
    Stage 6  real robot       adapter stub (collect() is a teleop session)

Three methods carry all of it:

- `metadata_spec`    what the acquisition interface can actually control, and
                     within which bounds. Directional acquisition is only
                     executable through these variables.
- `collect`          turn requested metadata into real samples. In simulation
                     this resets to a state and rolls out; for a static dataset
                     it is nearest-neighbour retrieval; for paid data it is an
                     order. The planner never needs to know which.
- `evaluation_set`   the *fixed* evaluation distribution, drawn before any
                     acquisition and never re-drawn (SETUP.md 33).

Note that `collect` is where the stages differ most in cost, and nothing else
in the codebase assumes it is cheap: the planner decides what to request before
anything is collected, which is the entire point of predicting utility for data
that does not exist yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from ldva.data.metadata import MetadataSpec
from ldva.data.samples import SampleStore


class EnvAdapter(ABC):
    """What an environment must provide for LDVA to run on it."""

    #: short identifier recorded in run reports
    name: str = "unnamed"

    @property
    @abstractmethod
    def metadata_spec(self) -> MetadataSpec:
        """Controllable acquisition variables and their feasible ranges."""

    @abstractmethod
    def collect(
        self,
        metadata: np.ndarray,
        rng: np.random.Generator,
        round_id: int = 0,
    ) -> SampleStore:
        """Acquire data for one acquisition *request* per row of `metadata`.

        A request is one episode, which is also the unit the budget and the
        cost model count (SETUP.md 31 prices per trajectory). One episode
        usually yields several chunks, so `len(store)` may exceed
        `len(metadata)`; what must hold is that `store.trajectory_id` contains
        exactly one distinct value per requested row.

        Implementations record the **realized** metadata, not the requested
        metadata, whenever a simulator or vendor cannot honour a request
        exactly - the metadata mapper is fitted on what actually arrived, so
        recording the request would teach it a map that does not exist.
        """

    @abstractmethod
    def evaluation_set(
        self, n: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed held-out evaluation chunks as `(obs, act)` tensors."""

    # ---- optional hooks --------------------------------------------------
    def acquisition_cost(self, metadata: np.ndarray) -> np.ndarray:
        """Monetary cost per requested sample (SETUP.md 31, 36).

        Free in simulation; a real price for the paid-data track. Returning
        zeros means only the count budget applies.
        """
        return np.zeros(len(np.atleast_2d(metadata)), dtype=np.float64)

    def initial_dataset(
        self, n: int, rng: np.random.Generator
    ) -> SampleStore:
        """D_0. Deliberately *incomplete* coverage, or acquisition has nowhere
        to expand into and the whole experiment is vacuous."""
        return self.collect(self.metadata_spec.sample(n, rng), rng, round_id=0)

    def policy_defaults(self) -> dict:
        """Suggested policy family for this stage (SETUP.md 9)."""
        return {"kind": "mlp_bc", "hidden": ()}


class NotImplementedAdapter(EnvAdapter):
    """Base for the stages that are not wired up yet.

    It raises with the specific steps required instead of failing somewhere deep
    in the pipeline, so the integration point is unambiguous.
    """

    stage: str = "?"
    setup_section: str = "?"
    install_hint: str = ""
    todo: tuple[str, ...] = ()

    def _not_ready(self, what: str):
        lines = [
            f"{self.name} ({self.stage}) is not implemented yet: {what}.",
            f"See SETUP.md section {self.setup_section}.",
        ]
        if self.install_hint:
            lines.append(f"Install into its own conda env (SETUP.md 3): {self.install_hint}")
        if self.todo:
            lines.append("To implement this adapter:")
            lines += [f"  {i + 1}. {t}" for i, t in enumerate(self.todo)]
        raise NotImplementedError("\n".join(lines))

    @property
    def metadata_spec(self) -> MetadataSpec:
        self._not_ready("metadata_spec")

    def collect(self, metadata, rng, round_id: int = 0) -> SampleStore:
        self._not_ready("collect")

    def evaluation_set(self, n: int, rng):
        self._not_ready("evaluation_set")


#: adapter registry, so experiment scripts can take `--env <name>`
_REGISTRY: dict[str, type[EnvAdapter]] = {}


def register_adapter(name: str, cls: type[EnvAdapter]) -> None:
    _REGISTRY[name] = cls


def get_adapter(name: str, **kwargs) -> EnvAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown env {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_adapters() -> dict[str, bool]:
    """Registered adapters and whether each is actually implemented."""
    return {
        n: not issubclass(c, NotImplementedAdapter) for n, c in sorted(_REGISTRY.items())
    }
