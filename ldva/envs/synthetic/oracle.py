"""Ground truth for Stage 0 (SETUP.md 4: "optimal acquisition allocation").

Because the synthetic world can be queried at any metadata, we can *actually
collect* a proposed acquisition batch and measure what it does to the policy.
That turns every acquisition claim into a checkable one:

- `realized_batch_gain`      - what a specific collected batch really achieves
- `realized_allocation_gain` - the same for a planner's allocation, via its
                               metadata plans, so the whole chain
                               latent direction -> metadata -> data -> gain
                               is exercised rather than just the latent part
- `oracle_best_allocation`   - brute force over allocations using *realized*
                               gain, i.e. the true optimum rather than the
                               optimum of the learned utility model

Only the synthetic stage can afford the last one; it is what makes the Stage 0
gate meaningful before anything runs in a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from ldva.acquisition.directions import AcquisitionDirection
from ldva.acquisition.metadata_mapper import MetadataMapper, MetadataPlan
from ldva.acquisition.objective import BudgetSpec, enumerate_allocations
from ldva.data.samples import SampleStore
from ldva.envs.synthetic.generator import SyntheticWorld
from ldva.policy.bc import MLPPolicy
from ldva.supervision.bc_task import BCSupervisionTask
from ldva.supervision.leave_one_out import LeaveOneOutEstimator


@dataclass
class OracleConfig:
    lr: float = 0.1
    n_steps: int = 4
    #: Monte Carlo repeats over the environment's sampling noise
    n_repeats: int = 3
    seed: int = 0


class SyntheticAcquisitionOracle:
    def __init__(
        self,
        world: SyntheticWorld,
        policy: MLPPolicy,
        val_obs: torch.Tensor,
        val_act: torch.Tensor,
        cfg: OracleConfig | None = None,
    ):
        self.world = world
        self.policy = policy
        self.val_obs = val_obs
        self.val_act = val_act
        self.cfg = cfg or OracleConfig()
        self.estimator = LeaveOneOutEstimator(lr=self.cfg.lr, n_steps=self.cfg.n_steps)

    def collect(self, metadata: np.ndarray, rng: np.random.Generator, round_id: int = 1) -> SampleStore:
        """Actually acquire data at the requested metadata."""
        return self.world.build_store(
            len(metadata), rng, metadata=np.atleast_2d(metadata), round_id=round_id
        )

    def realized_batch_gain(
        self,
        metadata: np.ndarray,
        checkpoint_flat: np.ndarray | None = None,
        policy_features: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """U(Update(theta, collected(m))) - U(theta), averaged over env noise."""
        rng = rng or np.random.default_rng(self.cfg.seed)
        gains = []
        for _ in range(self.cfg.n_repeats):
            store = self.collect(metadata, rng)
            task = BCSupervisionTask(
                self.policy, store, self.val_obs, self.val_act,
                policy_features=policy_features,
            )
            if checkpoint_flat is not None:
                task.set_checkpoint(
                    "oracle", checkpoint_flat,
                    np.zeros(4) if policy_features is None else policy_features,
                )
            gains.append(self.estimator.batch_gain(task, np.arange(len(store))))
        return {
            "realized_gain": float(np.mean(gains)),
            "realized_gain_std": float(np.std(gains)),
            "n_samples": int(len(metadata)),
            "n_repeats": self.cfg.n_repeats,
        }

    def realized_allocation_gain(
        self,
        plans: list[MetadataPlan],
        checkpoint_flat: np.ndarray | None = None,
        policy_features: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Realized gain of a planner's allocation, collected via its plans."""
        if not plans:
            return {"realized_gain": 0.0, "realized_gain_std": 0.0, "n_samples": 0}
        metadata = np.concatenate([p.metadata for p in plans], axis=0)
        out = self.realized_batch_gain(metadata, checkpoint_flat, policy_features, rng)
        out["n_directions_used"] = len(plans)
        return out

    def oracle_best_allocation(
        self,
        directions: list[AcquisitionDirection],
        budget: BudgetSpec,
        mapper: MetadataMapper,
        metadata_all: np.ndarray,
        checkpoint_flat: np.ndarray | None = None,
        policy_features: np.ndarray | None = None,
        max_allocations: int = 2000,
        seed: int = 0,
        progress: bool = False,
    ) -> dict:
        """True optimum by collecting and measuring every feasible allocation.

        This is the ground truth that the learned utility model is judged
        against. It is only affordable because the synthetic environment can be
        sampled at will; the same call on a simulator would be the entire
        compute budget.
        """
        rng = np.random.default_rng(seed)
        allocs = []
        for a in enumerate_allocations(len(directions), budget, max_count=max_allocations):
            allocs.append(a.copy())
        it = tqdm(allocs, desc="oracle alloc", leave=False) if progress else allocs

        rows = []
        for alloc in it:
            plans = mapper.plan_allocation(directions, alloc, metadata_all, rng=rng)
            r = self.realized_allocation_gain(plans, checkpoint_flat, policy_features, rng)
            rows.append({"allocation": alloc, **r})
        if not rows:
            raise RuntimeError("no feasible allocation to evaluate")
        best = max(rows, key=lambda r: r["realized_gain"])
        return {
            "best_allocation": best["allocation"],
            "best_realized_gain": best["realized_gain"],
            "n_evaluated": len(rows),
            "all": rows,
        }


def measure_realized_latent_movement(
    model,
    world: SyntheticWorld,
    plan: MetadataPlan,
    direction: AcquisitionDirection,
    policy_features: np.ndarray,
    rng: np.random.Generator,
    n_per_anchor: int = 24,
    paired: bool = True,
) -> dict:
    """Collect at the plan's metadata and check where the latents actually land.

    This closes the loop of PLAN.md 24: desired latent direction -> predicted
    metadata perturbation -> collected sample -> realized latent movement.

    The measurement is deliberately **anchor-matched and resampled on both
    sides**. Each planned row was derived from one specific anchor `m_a` by its
    own `delta_m`, so the displacement to measure is

        E[z | m_a + delta_m] - E[z | m_a]

    with both expectations estimated from fresh chunks. Comparing instead
    against the stored anchor chunk's latent, or against the mean of all
    anchors, mixes in two artifacts that can each dominate and even flip the
    sign: a single chunk's encoder noise, and the offset between the anchors a
    plan happened to use and the anchor set as a whole.

    Both sides are also drawn with **common random numbers** (`paired=True`):
    on the Stage 0 world a single chunk's latent noise has norm ~0.50 while a
    planned displacement is ~0.22, so an unpaired 12-vs-12 comparison has noise
    as large as the signal. Sharing the noise stream makes 12 paired samples as
    accurate as 256 unpaired ones.
    """
    from ldva.acquisition.metadata_mapper import evaluate_realized_direction

    deltas = []
    for row_m, anchor_m in zip(plan.metadata, plan.anchor_metadata):
        shared = int(rng.integers(0, 2**31 - 1))
        rng_base = np.random.default_rng(shared)
        rng_new = np.random.default_rng(shared) if paired else rng
        base_store = world.build_store(
            0, rng_base, metadata=np.tile(anchor_m, (n_per_anchor, 1)), round_id=1
        )
        new_store = world.build_store(
            0, rng_new, metadata=np.tile(row_m, (n_per_anchor, 1)), round_id=1
        )
        z_base = model.encode_store(base_store, policy_features).mean(0)
        z_new = model.encode_store(new_store, policy_features).mean(0)
        deltas.append(z_new - z_base)
    deltas = np.asarray(deltas)

    out = evaluate_realized_direction(
        np.zeros((1, deltas.shape[1])), deltas, direction.vector
    )
    out.update(
        {
            "direction_id": direction.direction_id,
            "cluster_id": direction.cluster_id,
            "achievable_cosine": plan.achievable_cosine,
            "reachability_cosine": plan.reachability_cosine,
            "jacobian_r2_heldout": plan.jacobian_r2_heldout,
            "n_anchors": int(len(deltas)),
            "n_per_anchor": int(n_per_anchor),
            "paired_sampling": bool(paired),
        }
    )
    return out
