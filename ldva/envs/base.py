"""The one seam between LDVA and an environment (PLAN.md 11).

Everything above this interface - supervision, the data model, clustering,
directions, the planners, the metadata mapper - is environment-agnostic. Moving
up the benchmark ladder means implementing `EnvAdapter` and nothing else:

    synthetic     preflight gate      implemented (`ldva/envs/synthetic`)
    DMC           smoke / debugging   implemented (`ldva/envs/dmc`)
    MetaWorld     first real result   implemented (`ldva/envs/metaworld`)
    PushT         optional            adapter stub (PLAN.md 16: not a blocker)
    ManiSkill     optional            adapter stub (PLAN.md 16: not a blocker)
    robosuite     Stage 2             not written
    paid data     Stage 3             not written (collect() issues an order)
    real robot    Stage 3             not written (collect() is a teleop session)

Four methods carry all of it:

- `metadata_spec`    what the acquisition interface can actually control, and
                     within which bounds. Directional acquisition is only
                     executable through these variables.
- `collect`          turn requested metadata into real samples. In simulation
                     this resets to a state and rolls out; for a static dataset
                     it is nearest-neighbour retrieval; for paid data it is an
                     order. The planner never needs to know which.
- `evaluation_set`   the *fixed* BC evaluation chunks, drawn before any
                     acquisition and never re-drawn.
- `evaluate_policy`  the *robot* metric: roll a learned policy out from fixed
                     conditions and report return / success (PLAN.md 15, P0.1).
                     Only simulators implement it; see `supports_rollout_eval`.

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
from ldva.envs.rollout import EvalConditions, RolloutMetrics


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
        cost model count (PLAN.md 8 prices per trajectory). One episode
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

    # ---- rollout evaluation (PLAN.md 15, P0.1) ---------------------------
    #: does `evaluate_policy` actually drive a simulator? False means the only
    #: available utility is the BC proxy, which PLAN.md 18 does not accept as a
    #: robotics outcome - experiment scripts check this and say so in the report
    #: rather than quietly plotting a regression loss as "performance".
    supports_rollout_eval: bool = False

    def eval_conditions(self, n: int, rng: np.random.Generator) -> EvalConditions:
        """The *fixed* rollout initial conditions (PLAN.md 15: "keep fixed
        evaluation conditions across all methods and rounds").

        Drawn uniformly over the whole metadata box by default, which is the
        distribution `initial_dataset` deliberately fails to cover.
        """
        return EvalConditions(
            metadata=self.metadata_spec.sample(n, rng),
            seed=int(rng.bit_generator.seed_seq.entropy or 0) if hasattr(
                rng.bit_generator, "seed_seq") else 0,
            env_name=self.name,
        )

    def evaluate_policy(
        self, policy, conditions: EvalConditions
    ) -> RolloutMetrics:
        """Roll `policy` out from each condition and report return / success.

        This is the robot metric: PLAN.md 15 asks for environment return on DMC
        and actual rollout success plus return on MetaWorld. Adapters without a
        steppable simulator raise, instead of returning a BC proxy dressed up as
        a rollout, so a missing number can never be mistaken for a real one.
        """
        raise NotImplementedError(
            f"{self.name} has no rollout evaluation: it cannot report real "
            "policy return or success (PLAN.md 15, P0.1). Either use an adapter "
            "with supports_rollout_eval = True (dmc, metaworld) or read the BC "
            "proxy utility, which PLAN.md 18 does not count as a robotics "
            "outcome."
        )

    # ---- optional hooks --------------------------------------------------
    def acquisition_cost(self, metadata: np.ndarray) -> np.ndarray:
        """Monetary cost per requested sample (PLAN.md 10).

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
        """Suggested policy family for this stage (PLAN.md 13)."""
        return {"kind": "mlp_bc", "hidden": ()}


class NotImplementedAdapter(EnvAdapter):
    """Base for the stages that are not wired up yet.

    It raises with the specific steps required instead of failing somewhere deep
    in the pipeline, so the integration point is unambiguous.
    """

    stage: str = "?"
    setup_section: str = "16"
    install_hint: str = ""
    todo: tuple[str, ...] = ()

    def _not_ready(self, what: str):
        lines = [
            f"{self.name} ({self.stage}) is not implemented yet: {what}.",
            f"See PLAN.md section {self.setup_section}.",
        ]
        if self.install_hint:
            lines.append(f"Install into its own conda env (PLAN.md 15, P1): {self.install_hint}")
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
