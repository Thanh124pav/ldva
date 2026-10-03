"""DMC adapter: dm_control for debugging (PLAN.md 15, item 3).

PLAN.md lists DMC/MuJoCo as the debugging rung of the MVP ladder, and that is
exactly what it is used for here: a real MuJoCo simulator with *cleanly
settable* state, so the whole acquisition loop can be exercised on physics
without MetaWorld's 39-dimensional observations and task multiplexing.

Two tasks are wired up, chosen because each gives a different shape of
acquisition metadata and each admits a genuine hand-written expert:

- `reacher/easy`   - metadata is the **goal**, parameterized in *polar*
                     coordinates (radius, angle). The expert is analytic 2-link
                     inverse kinematics plus a joint-space PD controller; it
                     reaches the target at step ~14 and scores 186/200. This is
                     the cleaner acquisition story: a latent direction maps to
                     "collect data for targets over there".
- `point_mass/easy` - metadata is the **initial state** (start x, y), with the
                     goal fixed at the origin. The expert is a PD controller
                     (~94/200; the actuator is slow, so the episode spends real
                     time travelling, which is what makes early chunks differ
                     strongly by starting position).

Other DMC tasks need both a state setter and a controller, so they raise rather
than silently collecting random-action data that no BC policy could learn from.

**Why the reacher target is polar, not Cartesian.** A `MetadataSpec` is a box,
so the declared box has to *be* the environment's feasible set. With Cartesian
`(target_x, target_y)` in a box it is not: the arm can only reach an annulus, so
the setter had to clip the radius into `[0.05, 0.20]`. That clip is a
non-local, angle-dependent kink that no local linear Jacobian can represent, and
it broke directional acquisition in a way that looked like a model failure -
measured on this task, plans whose latent displacement stayed small tracked the
intended direction well (cosine +0.83, +0.79, +0.86) while plans that hit the
clip were thrown elsewhere entirely (|displacement| ~2.0, cosine -0.98 to
-0.36), dragging the mean to +0.01. In polar coordinates the annulus *is* a box,
nothing clips, and the map stays locally linear. The angle range also stops
short of a full turn so the planner never has to cross the -pi/+pi seam, which
would be a discontinuity for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ldva.data.metadata import MetadataField, MetadataSpec
from ldva.data.samples import SampleStore
from ldva.envs.base import EnvAdapter, register_adapter

#: reacher link lengths, read off the model (body `hand` and `finger` offsets)
_REACHER_L1 = 0.12
_REACHER_L2 = 0.12


def _ik_2link(x: float, y: float, l1: float = _REACHER_L1, l2: float = _REACHER_L2) -> np.ndarray:
    """Analytic inverse kinematics for a planar 2-link arm.

    The radius is clamped inside the arm's reach, so an unreachable request
    becomes the closest reachable pose instead of a NaN.
    """
    r = float(np.hypot(x, y))
    r = float(np.clip(r, 1e-4, l1 + l2 - 1e-3))
    c2 = float(np.clip((r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0))
    q2 = float(np.arccos(c2))
    q1 = float(np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2)))
    return np.array([q1, q2], dtype=np.float64)


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


# ---- per-task plumbing ---------------------------------------------------


@dataclass
class DMCTaskSpec:
    domain: str
    task: str
    fields: list[MetadataField]
    #: write the requested metadata into the physics state at reset
    setter: Callable[[object, np.ndarray], None]
    #: read back what the simulator actually realized
    reader: Callable[[object], np.ndarray]
    #: hand-written expert: physics -> action
    expert: Callable[[object], np.ndarray]


def _reacher_setter(physics, meta: np.ndarray) -> None:
    """Place the target from (radius, angle). No clipping: the box is feasible."""
    r, th = float(meta[0]), float(meta[1])
    physics.named.model.geom_pos["target", "x"] = r * np.cos(th)
    physics.named.model.geom_pos["target", "y"] = r * np.sin(th)


def _reacher_reader(physics) -> np.ndarray:
    """Read the realized target back as (radius, angle)."""
    p = physics.named.model.geom_pos["target"]
    x, y = float(p[0]), float(p[1])
    return np.array([float(np.hypot(x, y)), float(np.arctan2(y, x))], dtype=np.float64)


def _reacher_expert(physics, kp: float = 12.0, kd: float = 1.0) -> np.ndarray:
    tgt = physics.named.data.geom_xpos["target"][:2]
    q_des = _ik_2link(float(tgt[0]), float(tgt[1]))
    q = np.asarray(physics.named.data.qpos[:], dtype=np.float64)
    dq = np.asarray(physics.named.data.qvel[:], dtype=np.float64)
    return np.clip(kp * _wrap(q_des - q) - kd * dq, -1.0, 1.0)


def _point_mass_setter(physics, meta: np.ndarray) -> None:
    physics.named.data.qpos["root_x"] = float(meta[0])
    physics.named.data.qpos["root_y"] = float(meta[1])


def _point_mass_reader(physics) -> np.ndarray:
    return np.array(
        [float(physics.named.data.qpos["root_x"]), float(physics.named.data.qpos["root_y"])],
        dtype=np.float64,
    )


def _point_mass_expert(physics, kp: float = 25.0, kd: float = 5.0) -> np.ndarray:
    pos = np.asarray(physics.named.data.qpos[:], dtype=np.float64)
    vel = np.asarray(physics.named.data.qvel[:], dtype=np.float64)
    return np.clip(-kp * pos - kd * vel, -1.0, 1.0)


TASK_SPECS: dict[str, DMCTaskSpec] = {
    "reacher-easy": DMCTaskSpec(
        domain="reacher",
        task="easy",
        # radius stays inside the arm's reach (L1 + L2 = 0.24) and away from the
        # singular centre; the angle stops short of a full turn so there is no
        # -pi/+pi seam inside the box. Both bounds are therefore honoured
        # exactly, with nothing to clip.
        fields=[
            MetadataField("target_radius", 0.06, 0.19),
            MetadataField("target_angle", -2.8, 2.8),
        ],
        setter=_reacher_setter,
        reader=_reacher_reader,
        expert=_reacher_expert,
    ),
    "point_mass-easy": DMCTaskSpec(
        domain="point_mass",
        task="easy",
        fields=[MetadataField("start_x", -0.25, 0.25), MetadataField("start_y", -0.25, 0.25)],
        setter=_point_mass_setter,
        reader=_point_mass_reader,
        expert=_point_mass_expert,
    ),
}


@dataclass
class DMCConfig:
    task: str = "reacher-easy"
    chunk_len: int = 16
    chunk_stride: int | None = None
    max_steps: int = 120
    max_chunks_per_episode: int = 4
    #: fraction of the episode tail used to decide success
    success_tail: float = 0.25
    success_threshold: float = 0.5
    seed: int = 0
    cost_per_episode: float = 0.0
    #: restrict the initial dataset to this fraction of each metadata range
    initial_corner_frac: float = 0.3
    extra: dict = field(default_factory=dict)


class DMCAdapter(EnvAdapter):
    name = "dmc"

    def __init__(self, cfg: DMCConfig | None = None, **kw):
        self.cfg = cfg or DMCConfig(**kw)
        if self.cfg.task not in TASK_SPECS:
            raise NotImplementedError(
                f"DMC task {self.cfg.task!r} has no state setter or expert controller.\n"
                f"Wired up: {sorted(TASK_SPECS)}.\n"
                "To add one, register a DMCTaskSpec with:\n"
                "  setter(physics, metadata) - write the request into the physics state\n"
                "  reader(physics)           - read back what was realized\n"
                "  expert(physics)           - a controller good enough to clone\n"
                "Collecting random-action rollouts instead would produce data no BC "
                "policy can learn from, so this is refused rather than defaulted."
            )
        from dm_control import suite  # imported lazily: only this stage needs it

        self.spec_ = TASK_SPECS[self.cfg.task]
        self.env = suite.load(
            self.spec_.domain, self.spec_.task, task_kwargs={"random": self.cfg.seed}
        )
        ts = self.env.reset()
        self._obs_keys = sorted(ts.observation)
        self._obs_dim = int(sum(np.asarray(ts.observation[k]).size for k in self._obs_keys))
        self._act_dim = int(self.env.action_spec().shape[0])
        self._spec = MetadataSpec(list(self.spec_.fields))

    # ---- observations -----------------------------------------------------
    def _flat_obs(self, observation) -> np.ndarray:
        return np.concatenate(
            [np.asarray(observation[k], dtype=np.float32).ravel() for k in self._obs_keys]
        )

    @property
    def metadata_spec(self) -> MetadataSpec:
        return self._spec

    # ---- collection -------------------------------------------------------
    def _rollout(self, meta_row: np.ndarray):
        cfg = self.cfg
        self.env.reset()
        with self.env.physics.reset_context():
            self.spec_.setter(self.env.physics, meta_row)
        realized = self.spec_.reader(self.env.physics)

        obs_list, act_list, rewards = [], [], []
        spec = self.env.action_spec()
        for _ in range(cfg.max_steps):
            a = np.clip(
                np.asarray(self.spec_.expert(self.env.physics), dtype=np.float64),
                spec.minimum,
                spec.maximum,
            )
            obs_list.append(self._flat_obs(self._observe()))
            act_list.append(a.astype(np.float32))
            ts = self.env.step(a)
            rewards.append(float(ts.reward or 0.0))
            if ts.last():
                break

        tail = max(1, int(cfg.success_tail * len(rewards)))
        success = bool(np.mean(rewards[-tail:]) > cfg.success_threshold) if rewards else False
        return (
            np.asarray(obs_list, dtype=np.float32),
            np.asarray(act_list, dtype=np.float32),
            float(np.sum(rewards)),
            success,
            realized,
        )

    def _observe(self):
        """Current observation dict, via the task's own observation function."""
        return self.env.task.get_observation(self.env.physics)

    def collect(self, metadata, rng, round_id: int = 0) -> SampleStore:
        cfg = self.cfg
        metadata = np.atleast_2d(np.asarray(metadata, dtype=np.float64))
        stride = cfg.chunk_stride or cfg.chunk_len

        obs_c, act_c, meta_c = [], [], []
        traj, start, rew, succ, cost = [], [], [], [], []
        for req_i, row in enumerate(metadata):
            obs, act, total_r, success, realized = self._rollout(row)
            n = 0
            for s in range(0, max(len(obs) - cfg.chunk_len + 1, 0), stride):
                if n >= cfg.max_chunks_per_episode:
                    break
                obs_c.append(obs[s : s + cfg.chunk_len])
                act_c.append(act[s : s + cfg.chunk_len])
                meta_c.append(realized)
                traj.append(req_i)
                start.append(s)
                rew.append(total_r)
                succ.append(success)
                cost.append(cfg.cost_per_episode)
                n += 1
            if n == 0:
                o = np.zeros((cfg.chunk_len, self._obs_dim), dtype=np.float32)
                a = np.zeros((cfg.chunk_len, self._act_dim), dtype=np.float32)
                o[: len(obs)], a[: len(act)] = obs, act
                obs_c.append(o)
                act_c.append(a)
                meta_c.append(realized)
                traj.append(req_i)
                start.append(0)
                rew.append(total_r)
                succ.append(success)
                cost.append(cfg.cost_per_episode)

        return SampleStore(
            obs=np.asarray(obs_c, dtype=np.float32),
            act=np.asarray(act_c, dtype=np.float32),
            metadata=np.asarray(meta_c, dtype=np.float64),
            metadata_spec=self._spec,
            trajectory_id=np.asarray(traj, dtype=np.int64),
            start_t=np.asarray(start, dtype=np.int64),
            task_id=np.zeros(len(obs_c), dtype=np.int64),
            reward=np.asarray(rew, dtype=np.float32),
            success=np.asarray(succ, dtype=bool),
            policy_ckpt_id=np.array(["scripted_expert"] * len(obs_c), dtype=object),
            round_id=np.full(len(obs_c), round_id, dtype=np.int64),
            cost=np.asarray(cost, dtype=np.float64),
        )

    # ---- evaluation -------------------------------------------------------
    def evaluation_set(self, n: int, rng):
        """Uniform over the whole metadata box, drawn once (SETUP.md 33)."""
        rows = self._spec.sample(max(1, n // max(self.cfg.max_chunks_per_episode, 1)) + 1, rng)
        store = self.collect(rows, rng, round_id=-1)
        idx = rng.permutation(len(store))[:n]
        return torch.from_numpy(store.obs[idx]), torch.from_numpy(store.act[idx])

    def initial_dataset(self, n: int, rng) -> SampleStore:
        """D_0 confined to one corner of the metadata box.

        The evaluation distribution spans the whole box, so coverage can only be
        completed by expanding support - which is the behaviour LDVA is meant to
        produce and which resampling the existing data cannot.
        """
        lo, hi = self._spec.low, self._spec.high
        frac = self.cfg.initial_corner_frac
        rows = rng.uniform(lo, lo + frac * (hi - lo), size=(max(n, 1), len(self._spec)))
        return self.collect(self._spec.clip(rows), rng, round_id=0)

    def acquisition_cost(self, metadata) -> np.ndarray:
        n = len(np.atleast_2d(metadata))
        return np.full(n, self.cfg.cost_per_episode, dtype=np.float64)

    def policy_defaults(self) -> dict:
        return {"kind": "mlp_bc", "hidden": (128, 128)}

    def report(self) -> dict:
        return {
            "task": self.cfg.task,
            "domain": self.spec_.domain,
            "obs_keys": self._obs_keys,
            "obs_dim": self._obs_dim,
            "act_dim": self._act_dim,
            "chunk_len": self.cfg.chunk_len,
            "metadata_fields": self._spec.names,
            "available_tasks": sorted(TASK_SPECS),
        }


register_adapter("dmc", DMCAdapter)
