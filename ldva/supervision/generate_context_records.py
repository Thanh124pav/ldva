"""Generate multi-context supervision (PLAN.md 6, 3.3).

Two requirements fight each other here. Every sample needs *many* contexts
(>= 20) so the model cannot memorize one scalar per sample, and the batch
compositions must be *diverse* so that co-occurrence structure does not become
a confound - if samples i and j always appear together, no readout can separate
their contributions.

`ContextBatchSampler` handles both: it draws from the least-covered samples
first (coverage), jitters batch sizes, and rejects compositions too similar to
ones already emitted for the same checkpoint (diversity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ldva.data.context_dataset import ContextRecord
from ldva.policy.checkpoints import CheckpointStore
from ldva.supervision.base import EffectEstimator, SupervisionTask


@dataclass
class ContextGenConfig:
    #: target number of contexts each sample should appear in
    contexts_per_sample: int = 24
    batch_size: int = 8
    #: batch sizes are drawn from batch_size +- jitter
    batch_size_jitter: int = 2
    #: reject a composition sharing more than this fraction with a previous one
    max_jaccard: float = 0.6
    max_rejections: int = 20
    #: hard cap on contexts, as a multiple of the ideal count. The coverage
    #: target is met by *continuing* rather than by running a fixed number of
    #: batches: the composer draws stochastically, so the ideal count
    #: `contexts_per_sample * n_samples / batch_size` leaves the least-covered
    #: samples short of the target.
    max_overshoot: float = 3.0
    #: label a random subset with an expensive estimator to calibrate the cheap one
    calibration_fraction: float = 0.0
    seed: int = 0
    progress: bool = False


@dataclass
class ContextGenReport:
    n_records: int = 0
    n_rejected: int = 0
    estimator_id: str = ""
    calibration: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    #: did every sample reach `contexts_per_sample`?
    target_met: bool = False
    min_contexts_per_sample: int = 0
    n_contexts_cap: int = 0


class ContextBatchSampler:
    """Coverage-driven, diversity-filtered batch composer."""

    def __init__(self, n_samples: int, cfg: ContextGenConfig, rng: np.random.Generator):
        self.n = int(n_samples)
        self.cfg = cfg
        self.rng = rng
        self.counts = np.zeros(self.n, dtype=np.int64)
        self._emitted: dict[str, list[set[int]]] = {}
        self.n_rejected = 0

    def _batch_size(self) -> int:
        j = self.cfg.batch_size_jitter
        lo = max(2, self.cfg.batch_size - j)
        hi = min(self.n, self.cfg.batch_size + j)
        return int(self.rng.integers(lo, hi + 1)) if hi > lo else int(lo)

    def _draw(self, size: int) -> np.ndarray:
        """Sample `size` ids, biased toward the least-covered ones.

        Softmax over negative coverage keeps it stochastic: a hard "take the
        `size` least covered" rule would emit near-identical batches.
        """
        deficit = self.counts.max() - self.counts + 1.0
        p = deficit / deficit.sum()
        return self.rng.choice(self.n, size=size, replace=False, p=p)

    def propose(self, ckpt_id: str) -> np.ndarray:
        seen = self._emitted.setdefault(ckpt_id, [])
        size = self._batch_size()
        best = None
        for _ in range(self.cfg.max_rejections):
            ids = self._draw(size)
            s = set(ids.tolist())
            if all(_jaccard(s, prev) <= self.cfg.max_jaccard for prev in seen[-64:]):
                best = ids
                break
            self.n_rejected += 1
        if best is None:  # give up on diversity rather than stall
            best = ids
        seen.append(set(best.tolist()))
        self.counts[best] += 1
        return np.sort(best)

    @property
    def done(self) -> bool:
        return bool((self.counts >= self.cfg.contexts_per_sample).all())


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def generate_context_records(
    task: SupervisionTask,
    checkpoints: CheckpointStore,
    estimator: EffectEstimator,
    n_samples: int,
    cfg: ContextGenConfig | None = None,
    calibration_estimator: EffectEstimator | None = None,
    utilization_rule_id: str = "uniform_minibatch",
) -> tuple[list[ContextRecord], ContextGenReport]:
    """Label batches across checkpoints until the coverage target is met.

    `task` is mutated as we walk checkpoints (via `set_checkpoint`), so pass a
    task you own.
    """
    cfg = cfg or ContextGenConfig()
    if len(checkpoints) == 0:
        raise ValueError("need at least one checkpoint to generate contexts")
    rng = np.random.default_rng(cfg.seed)
    sampler = ContextBatchSampler(n_samples, cfg, rng)

    #: the ideal count if coverage were perfectly even
    ideal = int(np.ceil(cfg.contexts_per_sample * n_samples / max(cfg.batch_size, 1)))
    cap = int(np.ceil(ideal * max(cfg.max_overshoot, 1.0)))
    records: list[ContextRecord] = []
    calib: list[tuple[np.ndarray, np.ndarray]] = []

    pbar = tqdm(total=cap, desc="contexts", leave=False) if cfg.progress else None
    ci = 0
    # keep going until every sample has met the target, not for a fixed number
    # of batches: the composer is stochastic, so stopping at `ideal` leaves the
    # least-covered samples short and silently breaks the coverage guarantee
    while ci < cap and not (sampler.done and len(records) >= ideal):
        ckpt = checkpoints[ci % len(checkpoints)]
        if hasattr(task, "set_checkpoint"):
            task.set_checkpoint(ckpt.ckpt_id, ckpt.flat_params, ckpt.features)
        ids = sampler.propose(ckpt.ckpt_id)
        labels = estimator.label(task, ids)
        records.append(
            ContextRecord(
                context_id=ci,
                policy_ckpt_id=ckpt.ckpt_id,
                batch_sample_ids=ids,
                per_sample_effects=labels.per_sample_effects,
                batch_gain=labels.batch_gain,
                utilization_rule_id=utilization_rule_id,
                policy_features=ckpt.features,
                estimator_id=labels.estimator_id,
            )
        )
        if (
            calibration_estimator is not None
            and cfg.calibration_fraction > 0
            and rng.random() < cfg.calibration_fraction
        ):
            ref = calibration_estimator.label(task, ids)
            calib.append((labels.per_sample_effects, ref.per_sample_effects))
        ci += 1
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    report = ContextGenReport(
        n_records=len(records),
        n_rejected=sampler.n_rejected,
        estimator_id=estimator.estimator_id,
        calibration=_calibration_report(calib, calibration_estimator),
        target_met=bool(sampler.done),
        min_contexts_per_sample=int(sampler.counts.min()),
        n_contexts_cap=cap,
    )
    if not sampler.done:
        print(
            f"[ldva] coverage target not met: min "
            f"{sampler.counts.min()}/{cfg.contexts_per_sample} contexts per sample "
            f"after {len(records)} contexts (cap {cap}). Raise max_overshoot or "
            f"lower contexts_per_sample."
        )
    return records, report


def _calibration_report(
    calib: list[tuple[np.ndarray, np.ndarray]], ref_estimator: EffectEstimator | None
) -> dict:
    """Correlate the cheap target against the expensive one (PLAN.md 3.2).

    If this is weak, the cheap labels are mostly estimator noise and latent
    learning will collapse - failure mode F3 in PLAN.md 19.
    """
    if not calib:
        return {}
    from scipy.stats import pearsonr, spearmanr

    cheap = np.concatenate([c for c, _ in calib])
    exp = np.concatenate([r for _, r in calib])
    out = {
        "n_calibration_samples": int(len(cheap)),
        "reference_estimator": getattr(ref_estimator, "estimator_id", "unknown"),
    }
    if len(cheap) > 2 and np.std(cheap) > 0 and np.std(exp) > 0:
        out["pearson"] = float(pearsonr(cheap, exp)[0])
        out["spearman"] = float(spearmanr(cheap, exp)[0])
    return out
