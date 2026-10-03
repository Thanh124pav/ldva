"""MetaWorld adapter: the first meaningful result (PLAN.md 13, 14 Stage 1C).

MetaWorld is the first stage where acquisition is *genuinely* controllable, and
that rests on one mechanism: setting `_freeze_rand_vec = True` and writing
`_last_rand_vec` pins the episode's randomized object and goal placement, so a
latent direction can be turned into an actual collection request. Verified on
this install: shifting `object_x` by +0.08 moves the object in the observation
from -0.012 to 0.068 while the goal stays put.

Three deviations from PLAN.md 13 are forced by the installed package and are
recorded here rather than hidden:

1. **v3, not v2.** metaworld 3.1.1 ships only `*-v3` environments. The five
   tasks PLAN.md 14 names all exist: `reach-v3`, `push-v3`, `pick-place-v3`,
   `drawer-open-v3`, `button-press-v3`.
2. **The controllable metadata width differs per task.** `reach`, `push` and
   `pick-place` randomize a 6-vector (object xyz + goal xyz); `drawer-open` and
   `button-press` randomize only a 3-vector. A `MetadataSpec` has one global
   box, so the spec uses the union of the per-task ranges and `collect` clips
   each request to the task's own feasible set. The **realized** metadata is
   what gets recorded, so the metadata mapper learns the map that actually
   holds - including the fact that some coordinates do not move for some tasks.
3. **Data comes from the scripted experts**, not from a learned policy. All
   five `Sawyer*V3Policy` controllers are available and solve their task (push
   succeeds in ~63 steps), which is what makes BC data generation cheap enough
   to run many acquisition rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ldva.data.metadata import MetadataField, MetadataSpec
from ldva.data.samples import SampleStore
from ldva.envs.base import EnvAdapter, register_adapter
from ldva.envs.rollout import EvalConditions, PolicyActor, RolloutMetrics

#: PLAN.md 14 (Stage 1C) starter tasks, as v3 names
DEFAULT_TASKS = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "drawer-open-v3",
    "button-press-v3",
)

#: the 6 slots a task's random-reset vector can occupy
_SLOT_NAMES = ("object_x", "object_y", "object_z", "goal_x", "goal_y", "goal_z")

#: Minimum object-goal xy separation MetaWorld itself enforces, read off
#: `reset_model` in sawyer_reach_v3 / sawyer_push_v3 / sawyer_pick_place_v3:
#:
#:     goal_pos = self._get_state_rand_vec()
#:     while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
#:         goal_pos = self._get_state_rand_vec()
#:
#: Normally that loop resamples until the constraint holds. Under
#: `_freeze_rand_vec = True` - the mechanism that makes acquisition
#: controllable at all - `_get_state_rand_vec` returns `_last_rand_vec`
#: unchanged, so the loop **never terminates** and `env.reset()` hangs forever
#: on any pinned vector that violates the constraint. No exception, no timeout:
#: the experiment simply stops. Requests near the violating set are perfectly
#: reasonable ones for a planner to make, so every pinned vector is repaired
#: before it reaches the simulator.
_MIN_OBJ_GOAL_SEP = 0.15

#: tasks whose `reset_model` contains that rejection loop
_SEPARATION_TASKS = frozenset({"reach-v3", "push-v3", "pick-place-v3"})


def _policy_class_name(task: str) -> str:
    """`push-v3` -> `SawyerPushV3Policy`."""
    stem = task.replace("-v3", "")
    return "Sawyer" + "".join(w.capitalize() for w in stem.split("-")) + "V3Policy"


@dataclass
class MetaWorldConfig:
    tasks: tuple[str, ...] = DEFAULT_TASKS
    chunk_len: int = 16
    #: stride between chunks within one episode; `None` means non-overlapping
    chunk_stride: int | None = None
    #: cap the rollout; the experts finish well inside this
    max_steps: int = 160
    #: stop the rollout a few steps after success instead of running the horizon
    stop_after_success: int = 5
    #: at most this many chunks kept per episode, taken from the start
    max_chunks_per_episode: int = 4
    #: observation noise added to the stored chunks, as a fraction of obs std
    obs_noise: float = 0.0
    seed: int = 0
    #: monetary cost per episode (PLAN.md 10); free in simulation
    cost_per_episode: float = 0.0


class MetaWorldAdapter(EnvAdapter):
    name = "metaworld"

    def __init__(self, cfg: MetaWorldConfig | None = None, **kw):
        self.cfg = cfg or MetaWorldConfig(**kw)
        import metaworld  # imported lazily: only Stage 2 needs it
        import metaworld.policies as policies

        self._mw = metaworld
        self._envs: dict[str, object] = {}
        self._policies: dict[str, object] = {}
        self._rand_dim: dict[str, int] = {}
        self._task_box: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for t in self.cfg.tasks:
            key = f"{t}-goal-observable"
            if key not in metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
                raise KeyError(
                    f"task {t!r} not in this metaworld build; available v3 tasks "
                    f"include {sorted(metaworld.ALL_V3_ENVIRONMENTS)[:5]}..."
                )
            env = metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[key](seed=self.cfg.seed)
            env.reset()
            space = env._random_reset_space
            self._envs[t] = env
            self._rand_dim[t] = int(space.shape[0])
            self._task_box[t] = (
                np.asarray(space.low, dtype=np.float64),
                np.asarray(space.high, dtype=np.float64),
            )
            pol_name = _policy_class_name(t)
            if not hasattr(policies, pol_name):
                raise KeyError(f"no scripted expert {pol_name} for task {t!r}")
            self._policies[t] = getattr(policies, pol_name)()

        self._obs_dim = int(self._envs[self.cfg.tasks[0]].observation_space.shape[0])
        self._act_dim = int(self._envs[self.cfg.tasks[0]].action_space.shape[0])
        for t, e in self._envs.items():
            if e.observation_space.shape[0] != self._obs_dim or e.action_space.shape[0] != self._act_dim:
                raise ValueError(
                    f"task {t} has obs/act dims {e.observation_space.shape[0]}/"
                    f"{e.action_space.shape[0]}, expected {self._obs_dim}/{self._act_dim}; "
                    "a single SampleStore needs one shared shape"
                )
        self._spec = self._build_spec()

    # ---- metadata ---------------------------------------------------------
    @property
    def multi_task(self) -> bool:
        return len(self.cfg.tasks) > 1

    def _build_spec(self) -> MetadataSpec:
        """Union of per-task reset boxes, with task id first when multi-task."""
        lows = np.full((len(self.cfg.tasks), 6), np.nan)
        highs = np.full((len(self.cfg.tasks), 6), np.nan)
        for i, t in enumerate(self.cfg.tasks):
            lo, hi = self._task_box[t]
            n = self._rand_dim[t]
            lows[i, :n], highs[i, :n] = lo[:n], hi[:n]
        low = np.nanmin(lows, axis=0)
        high = np.nanmax(highs, axis=0)

        fields = []
        if self.multi_task:
            fields.append(
                MetadataField("task_id", 0.0, float(len(self.cfg.tasks) - 1), discrete=True)
            )
        for k, nm in enumerate(_SLOT_NAMES):
            lo, hi = float(low[k]), float(high[k])
            if not np.isfinite(lo):
                # no task randomizes this slot; keep it but pin it
                fields.append(MetadataField(nm, 0.0, 0.0, controllable=False))
                continue
            if hi <= lo:
                hi = lo + 1e-6  # a degenerate slot stays declared but unmovable
                fields.append(MetadataField(nm, lo, hi, controllable=False))
                continue
            fields.append(MetadataField(nm, lo, hi, cost=0.0))
        return MetadataSpec(fields)

    @property
    def metadata_spec(self) -> MetadataSpec:
        return self._spec

    def _split_row(self, row: np.ndarray) -> tuple[str, np.ndarray]:
        """Split a metadata row into (task name, 6-slot reset vector)."""
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        if self.multi_task:
            idx = int(np.clip(round(float(row[0])), 0, len(self.cfg.tasks) - 1))
            return self.cfg.tasks[idx], row[1:7].copy()
        return self.cfg.tasks[0], row[0:6].copy()

    def _merge_row(self, task: str, slots: np.ndarray) -> np.ndarray:
        out = []
        if self.multi_task:
            out.append(float(self.cfg.tasks.index(task)))
        out.extend(np.asarray(slots, dtype=np.float64).reshape(-1)[:6].tolist())
        return np.array(out, dtype=np.float64)

    # ---- pinning the reset vector ----------------------------------------
    def _repair_separation(self, task: str, rv: np.ndarray) -> np.ndarray:
        """Push the goal away from the object until MetaWorld will accept it.

        Returns a vector satisfying `||obj_xy - goal_xy|| >= _MIN_OBJ_GOAL_SEP`,
        so `reset_model`'s rejection loop exits on its first test. The move is
        the smallest one that works: slide the goal radially outward from the
        object to the required radius, then clip back into the goal box. If the
        clip pulls it back inside the forbidden disc, fall back to the corner of
        the goal box furthest from the object, which is the most separated point
        the box contains.

        For push-v3 the boxes are object xy in [-0.1,0.1]x[0.6,0.7] and goal xy
        in [-0.1,0.1]x[0.8,0.9], so |dy| >= 0.1 always and the corner fallback
        reaches 0.36 - the constraint is always satisfiable here.
        """
        if task not in _SEPARATION_TASKS or len(rv) < 5:
            return rv
        lo, hi = self._task_box[task]
        obj = rv[0:2]
        goal = rv[3:5]
        d = goal - obj
        dist = float(np.linalg.norm(d))
        if dist >= _MIN_OBJ_GOAL_SEP:
            return rv
        need = _MIN_OBJ_GOAL_SEP + 1e-3  # a hair over, never exactly on it
        # degenerate overlap has no outward direction; +y is the one axis with
        # guaranteed headroom in every affected task's goal box
        u = d / dist if dist > 1e-9 else np.array([0.0, 1.0])
        out = rv.copy()
        cand = np.clip(obj + need * u, lo[3:5], hi[3:5])
        if float(np.linalg.norm(cand - obj)) >= _MIN_OBJ_GOAL_SEP:
            out[3:5] = cand
            return out
        corners = np.array(
            [[x, y] for x in (lo[3], hi[3]) for y in (lo[4], hi[4])], dtype=np.float64
        )
        best = corners[int(np.argmax(np.linalg.norm(corners - obj, axis=1)))]
        if float(np.linalg.norm(best - obj)) < _MIN_OBJ_GOAL_SEP:
            raise RuntimeError(
                f"task {task!r} cannot satisfy the object-goal separation of "
                f"{_MIN_OBJ_GOAL_SEP} for object at {obj}: the whole goal box "
                f"[{lo[3:5]}, {hi[3:5]}] lies inside the forbidden disc. "
                "env.reset() would hang, so this is refused instead."
            )
        out[3:5] = best
        return out

    def _pin(self, task: str, slots: np.ndarray) -> np.ndarray:
        """Clip a request into the task's box, repair it, and pin it.

        The single place where `_freeze_rand_vec` / `_last_rand_vec` are set, so
        collection and rollout evaluation cannot drift apart in how they
        interpret a request - and so the hang guard cannot be bypassed by one
        of them. Returns the **realized** vector, which is what gets recorded
        as metadata: the mapper has to learn the map that actually holds,
        including the repair.
        """
        env = self._envs[task]
        n = self._rand_dim[task]
        lo, hi = self._task_box[task]
        rv = np.clip(np.asarray(slots, dtype=np.float64)[:n], lo[:n], hi[:n])
        rv = self._repair_separation(task, rv)
        env._freeze_rand_vec = True
        env._last_rand_vec = rv
        return rv

    # ---- collection -------------------------------------------------------
    def _rollout(self, task: str, slots: np.ndarray, rng: np.random.Generator):
        """Pin the reset vector, run the scripted expert, return the episode."""
        env = self._envs[task]
        pol = self._policies[task]
        n = self._rand_dim[task]

        rv = self._pin(task, slots)
        obs, _ = env.reset()

        obs_list, act_list = [], []
        total_r, success, t_success = 0.0, False, None
        lo_a, hi_a = env.action_space.low, env.action_space.high
        for t in range(self.cfg.max_steps):
            a = np.clip(np.asarray(pol.get_action(obs), dtype=np.float64), lo_a, hi_a)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(a.astype(np.float32))
            obs, r, term, trunc, info = env.step(a)
            total_r += float(r)
            if not success and float(info.get("success", 0.0)) > 0.5:
                success, t_success = True, t
            if success and t - t_success >= self.cfg.stop_after_success:
                break
            if term or trunc:
                break

        # realized metadata: the clipped reset vector, padded with the goal the
        # simulator actually used, so short-rand_vec tasks still report a goal
        realized = np.zeros(6, dtype=np.float64)
        realized[:n] = rv
        if n < 6:
            tgt = np.asarray(getattr(env, "_target_pos", np.zeros(3)), dtype=np.float64)
            realized[3:6] = tgt[:3]
        return (
            np.asarray(obs_list, dtype=np.float32),
            np.asarray(act_list, dtype=np.float32),
            total_r,
            success,
            realized,
        )

    def collect(self, metadata, rng, round_id: int = 0) -> SampleStore:
        cfg = self.cfg
        metadata = np.atleast_2d(np.asarray(metadata, dtype=np.float64))
        stride = cfg.chunk_stride or cfg.chunk_len

        obs_c, act_c, meta_c = [], [], []
        traj_id, start_t, task_id, rew, succ, cost = [], [], [], [], [], []

        for req_i, row in enumerate(metadata):
            task, slots = self._split_row(row)
            obs, act, total_r, success, realized = self._rollout(task, slots, rng)
            row_meta = self._merge_row(task, realized)

            n_chunks = 0
            for s in range(0, max(len(obs) - cfg.chunk_len + 1, 0), stride):
                if n_chunks >= cfg.max_chunks_per_episode:
                    break
                o = obs[s : s + cfg.chunk_len]
                a = act[s : s + cfg.chunk_len]
                if cfg.obs_noise > 0:
                    o = o + cfg.obs_noise * o.std(0, keepdims=True) * rng.normal(size=o.shape).astype(np.float32)
                obs_c.append(o)
                act_c.append(a)
                meta_c.append(row_meta)
                traj_id.append(req_i)
                start_t.append(s)
                task_id.append(self.cfg.tasks.index(task))
                rew.append(total_r)
                succ.append(success)
                cost.append(cfg.cost_per_episode)
                n_chunks += 1

            if n_chunks == 0:
                # episode shorter than one chunk: keep a zero-padded chunk so a
                # request never silently vanishes from the budget accounting
                o = np.zeros((cfg.chunk_len, self._obs_dim), dtype=np.float32)
                a = np.zeros((cfg.chunk_len, self._act_dim), dtype=np.float32)
                o[: len(obs)] = obs
                a[: len(act)] = act
                obs_c.append(o)
                act_c.append(a)
                meta_c.append(row_meta)
                traj_id.append(req_i)
                start_t.append(0)
                task_id.append(self.cfg.tasks.index(task))
                rew.append(total_r)
                succ.append(success)
                cost.append(cfg.cost_per_episode)

        return SampleStore(
            obs=np.asarray(obs_c, dtype=np.float32),
            act=np.asarray(act_c, dtype=np.float32),
            metadata=np.asarray(meta_c, dtype=np.float64),
            metadata_spec=self._spec,
            trajectory_id=np.asarray(traj_id, dtype=np.int64),
            start_t=np.asarray(start_t, dtype=np.int64),
            task_id=np.asarray(task_id, dtype=np.int64),
            reward=np.asarray(rew, dtype=np.float32),
            success=np.asarray(succ, dtype=bool),
            policy_ckpt_id=np.array(["scripted_expert"] * len(obs_c), dtype=object),
            round_id=np.full(len(obs_c), round_id, dtype=np.int64),
            cost=np.asarray(cost, dtype=np.float64),
        )

    # ---- rollout evaluation (PLAN.md 15, P0.1) ---------------------------
    supports_rollout_eval = True

    def eval_conditions(self, n: int, rng) -> EvalConditions:
        """Fixed object/goal placements spread evenly over the tasks.

        Drawn from each task's *full* reset box, while `initial_dataset` draws
        from the lower quarter - so the evaluation distribution is only
        reachable by expanding support, which is the point of the experiment.
        """
        per_task = max(1, n // len(self.cfg.tasks))
        rows = []
        for t in self.cfg.tasks:
            lo, hi = self._task_box[t]
            k = self._rand_dim[t]
            for _ in range(per_task):
                slots = np.zeros(6)
                slots[:k] = rng.uniform(lo[:k], hi[:k])
                slots[:k] = self._repair_separation(t, slots[:k])
                rows.append(self._merge_row(t, slots))
        return EvalConditions(
            metadata=np.array(rows[:n] if len(rows) >= n else rows),
            seed=int(self.cfg.seed),
            env_name=self.name,
        )

    def evaluate_policy(self, policy, conditions: EvalConditions) -> RolloutMetrics:
        """Run `policy` in MetaWorld from each pinned reset vector.

        PLAN.md 15 asks for "actual rollout success and return" here, and
        success is MetaWorld's own `info["success"]` flag - the benchmark's
        definition, not a threshold we chose. The reset vector is pinned the
        same way `collect` pins it, so an evaluation condition is reproducible
        across methods and rounds.
        """
        returns, successes, lengths = [], [], []
        per_task_success: dict[str, list[bool]] = {t: [] for t in self.cfg.tasks}
        with PolicyActor(policy) as actor:
            for row in conditions.metadata:
                task, slots = self._split_row(row)
                env = self._envs[task]
                self._pin(task, slots)
                obs, _ = env.reset()
                lo_a, hi_a = env.action_space.low, env.action_space.high
                total_r, success, steps = 0.0, False, 0
                for t in range(self.cfg.max_steps):
                    a = np.clip(actor(obs), lo_a, hi_a)
                    obs, r, term, trunc, info = env.step(a)
                    total_r += float(r)
                    steps = t + 1
                    if float(info.get("success", 0.0)) > 0.5:
                        success = True
                        break
                    if term or trunc:
                        break
                returns.append(total_r)
                successes.append(success)
                lengths.append(steps)
                per_task_success[task].append(success)
        return RolloutMetrics.from_episodes(
            returns, successes, lengths,
            metric="metaworld_success",
            success_rule="metaworld info['success'] at any step",
            per_task_success={
                t: float(np.mean(v)) for t, v in per_task_success.items() if v
            },
        )

    def expert_reference(self, conditions: EvalConditions) -> RolloutMetrics:
        """The scripted experts' score on the *same* conditions (the ceiling)."""
        returns, successes, lengths = [], [], []
        rng = np.random.default_rng(0)
        for row in conditions.metadata:
            task, slots = self._split_row(row)
            _, _, total_r, success, _ = self._rollout(task, slots, rng)
            returns.append(total_r)
            successes.append(success)
            lengths.append(self.cfg.max_steps)
        return RolloutMetrics.from_episodes(
            returns, successes, lengths,
            metric="metaworld_success", policy="scripted_expert",
        )

    # ---- evaluation -------------------------------------------------------
    def evaluation_set(self, n: int, rng):
        """Fixed held-out chunks, spread evenly over the tasks.

        Drawn once before any acquisition and never re-drawn.
        """
        per_task = max(1, n // len(self.cfg.tasks))
        rows = []
        for t in self.cfg.tasks:
            lo, hi = self._task_box[t]
            k = self._rand_dim[t]
            for _ in range(per_task):
                slots = np.zeros(6)
                slots[:k] = rng.uniform(lo[:k], hi[:k])
                slots[:k] = self._repair_separation(t, slots[:k])
                rows.append(self._merge_row(t, slots))
        store = self.collect(np.array(rows), rng, round_id=-1)
        idx = rng.permutation(len(store))[:n]
        return (
            torch.from_numpy(store.obs[idx]),
            torch.from_numpy(store.act[idx]),
        )

    def initial_dataset(self, n: int, rng) -> SampleStore:
        """D_0 with deliberately incomplete coverage (PLAN.md 10).

        Requests are drawn from one corner of each task's reset box, so the
        evaluation distribution - which spans the whole box - can only be
        covered by *expanding* support. A uniform D_0 would make directional
        acquisition pointless.
        """
        rows = []
        for i in range(n):
            t = self.cfg.tasks[i % len(self.cfg.tasks)]
            lo, hi = self._task_box[t]
            k = self._rand_dim[t]
            slots = np.zeros(6)
            # lower quarter of each controllable range
            slots[:k] = rng.uniform(lo[:k], lo[:k] + 0.25 * (hi[:k] - lo[:k]))
            slots[:k] = self._repair_separation(t, slots[:k])
            rows.append(self._merge_row(t, slots))
        return self.collect(np.array(rows), rng, round_id=0)

    def acquisition_cost(self, metadata) -> np.ndarray:
        n = len(np.atleast_2d(metadata))
        return np.full(n, self.cfg.cost_per_episode, dtype=np.float64)

    def policy_defaults(self) -> dict:
        # PLAN.md 13 moves to Diffusion Policy / ACT at Stage 2; Stage 1 trains BC
        # on scripted-expert chunks, which is what makes many rounds affordable
        return {"kind": "mlp_bc", "hidden": (256, 256)}

    def report(self) -> dict:
        return {
            "tasks": list(self.cfg.tasks),
            "obs_dim": self._obs_dim,
            "act_dim": self._act_dim,
            "chunk_len": self.cfg.chunk_len,
            "rand_vec_dims": {t: self._rand_dim[t] for t in self.cfg.tasks},
            "metadata_fields": self._spec.names,
            "controllable": self._spec.controllable_mask.tolist(),
            "note": "v3 task names; metaworld 3.1.1 has no v2 environments",
        }


register_adapter("metaworld", MetaWorldAdapter)
