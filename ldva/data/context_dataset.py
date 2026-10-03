"""Multi-context supervision records (PLAN.md 3.2, 6; SETUP.md 11).

The central constraint of LDVA is that a sample must *not* carry one historical
scalar. Every sample appears under many `(batch, policy)` contexts, and the data
model is only ever supervised through those contexts. This module defines the
record format, the coverage diagnostics that tell us whether the supervision is
actually multi-context, and the torch plumbing that pads ragged batches.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ldva.data.samples import SampleStore


@dataclass
class ContextRecord:
    """One optimization context (SETUP.md 11).

    `per_sample_effects[k]` is the contextual effect of `batch_sample_ids[k]`
    *inside this batch under this checkpoint* - the same sample in another
    record will generally have a different value, which is the whole point.
    """

    context_id: int
    policy_ckpt_id: str
    batch_sample_ids: np.ndarray  # (n,) int
    per_sample_effects: np.ndarray  # (n,) float
    batch_gain: float
    utilization_rule_id: str = "uniform_minibatch"
    #: numeric features describing the checkpoint (step, loss, grad norm, ...)
    policy_features: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: which estimator produced the labels, e.g. "gradient_alignment"
    estimator_id: str = "unknown"

    def __post_init__(self) -> None:
        self.batch_sample_ids = np.asarray(self.batch_sample_ids, dtype=np.int64)
        self.per_sample_effects = np.asarray(self.per_sample_effects, dtype=np.float64)
        self.policy_features = np.asarray(self.policy_features, dtype=np.float32)
        if self.batch_sample_ids.shape != self.per_sample_effects.shape:
            raise ValueError(
                "batch_sample_ids and per_sample_effects must have the same shape, "
                f"got {self.batch_sample_ids.shape} vs {self.per_sample_effects.shape}"
            )

    def __len__(self) -> int:
        return int(self.batch_sample_ids.shape[0])


@dataclass
class EffectNormalizer:
    """Standardizes effect / gain labels.

    Raw gradient-alignment and leave-one-out targets are tiny (1e-4 and below)
    and their scale drifts across checkpoints, so regression on raw values is
    badly conditioned. Metrics are reported in normalized space; rank
    correlations are unaffected by the affine map.
    """

    effect_mean: float = 0.0
    effect_std: float = 1.0
    gain_mean: float = 0.0
    gain_std: float = 1.0

    @classmethod
    def fit(cls, records: list[ContextRecord], eps: float = 1e-12) -> "EffectNormalizer":
        effects = np.concatenate([r.per_sample_effects for r in records])
        gains = np.array([r.batch_gain for r in records], dtype=np.float64)
        return cls(
            effect_mean=float(effects.mean()),
            effect_std=float(max(effects.std(), eps)),
            gain_mean=float(gains.mean()),
            gain_std=float(max(gains.std(), eps)),
        )

    def effect(self, x):
        return (x - self.effect_mean) / self.effect_std

    def gain(self, x):
        return (x - self.gain_mean) / self.gain_std

    def inverse_gain(self, x):
        return x * self.gain_std + self.gain_mean

    def to_dict(self) -> dict:
        return {
            "effect_mean": self.effect_mean,
            "effect_std": self.effect_std,
            "gain_mean": self.gain_mean,
            "gain_std": self.gain_std,
        }


class ContextDataset(Dataset):
    """Context records joined against a `SampleStore`.

    Item `i` is context `i` with its chunks gathered; `collate` pads the ragged
    batch dimension and returns a mask, so a single forward pass can mix
    contexts of different sizes.
    """

    def __init__(
        self,
        store: SampleStore,
        records: list[ContextRecord],
        normalizer: EffectNormalizer | None = None,
        normalize: bool = True,
    ):
        self.store = store
        self.records = list(records)
        if not self.records:
            raise ValueError("ContextDataset needs at least one record")
        max_id = max(int(r.batch_sample_ids.max()) for r in self.records)
        if max_id >= len(store):
            raise ValueError(
                f"record references sample {max_id} but store holds {len(store)}"
            )
        self.normalize = normalize
        self.normalizer = normalizer or EffectNormalizer.fit(self.records)
        self._ckpt_ids = sorted({r.policy_ckpt_id for r in self.records})
        self._ckpt_index = {c: i for i, c in enumerate(self._ckpt_ids)}
        self._policy_feat_dim = int(
            max((r.policy_features.shape[0] for r in self.records), default=0)
        )

    # ---- shape / vocabulary -------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    @property
    def n_checkpoints(self) -> int:
        return len(self._ckpt_ids)

    @property
    def policy_feat_dim(self) -> int:
        return self._policy_feat_dim

    @property
    def checkpoint_ids(self) -> list[str]:
        return list(self._ckpt_ids)

    def checkpoint_index(self, ckpt_id: str) -> int:
        return self._ckpt_index[ckpt_id]

    # ---- torch interface ----------------------------------------------
    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        ids = r.batch_sample_ids
        effects = r.per_sample_effects
        gain = r.batch_gain
        if self.normalize:
            effects = self.normalizer.effect(effects)
            gain = self.normalizer.gain(gain)
        pf = r.policy_features
        if pf.shape[0] < self._policy_feat_dim:
            pf = np.pad(pf, (0, self._policy_feat_dim - pf.shape[0]))
        return {
            "obs": torch.from_numpy(self.store.obs[ids]),
            "act": torch.from_numpy(self.store.act[ids]),
            "meta": torch.from_numpy(
                self.store.metadata_norm()[ids].astype(np.float32)
            ),
            "effects": torch.from_numpy(effects.astype(np.float32)),
            "gain": torch.tensor(float(gain), dtype=torch.float32),
            "sample_ids": torch.from_numpy(ids.astype(np.int64)),
            "policy_features": torch.from_numpy(pf.astype(np.float32)),
            "ckpt_index": torch.tensor(
                self._ckpt_index[r.policy_ckpt_id], dtype=torch.long
            ),
            "context_id": torch.tensor(int(r.context_id), dtype=torch.long),
        }

    @staticmethod
    def collate(items: list[dict]) -> dict:
        """Pad contexts to the max batch size and return a validity mask."""
        n_ctx = len(items)
        sizes = [it["obs"].shape[0] for it in items]
        n_max = max(sizes)
        chunk_len, obs_dim = items[0]["obs"].shape[1:]
        act_dim = items[0]["act"].shape[2]
        meta_dim = items[0]["meta"].shape[1]

        obs = torch.zeros(n_ctx, n_max, chunk_len, obs_dim)
        act = torch.zeros(n_ctx, n_max, chunk_len, act_dim)
        meta = torch.zeros(n_ctx, n_max, meta_dim)
        effects = torch.zeros(n_ctx, n_max)
        sample_ids = torch.full((n_ctx, n_max), -1, dtype=torch.long)
        mask = torch.zeros(n_ctx, n_max, dtype=torch.bool)

        for i, it in enumerate(items):
            n = sizes[i]
            obs[i, :n] = it["obs"]
            act[i, :n] = it["act"]
            meta[i, :n] = it["meta"]
            effects[i, :n] = it["effects"]
            sample_ids[i, :n] = it["sample_ids"]
            mask[i, :n] = True

        return {
            "obs": obs,
            "act": act,
            "meta": meta,
            "effects": effects,
            "sample_ids": sample_ids,
            "mask": mask,
            "gain": torch.stack([it["gain"] for it in items]),
            "policy_features": torch.stack([it["policy_features"] for it in items]),
            "ckpt_index": torch.stack([it["ckpt_index"] for it in items]),
            "context_id": torch.stack([it["context_id"] for it in items]),
        }

    def loader(self, batch_size: int = 16, shuffle: bool = True, **kw) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            **kw,
        )

    def split(
        self, val_frac: float = 0.2, seed: int = 0, by: str = "context"
    ) -> tuple["ContextDataset", "ContextDataset"]:
        """Split into train/val.

        `by="context"` holds out whole contexts (tests generalization to new
        batch compositions). `by="checkpoint"` holds out whole checkpoints,
        which is the harder and more honest test of policy-context transfer.
        """
        rng = np.random.default_rng(seed)
        if by == "context":
            idx = rng.permutation(len(self.records))
            n_val = max(1, int(round(val_frac * len(idx))))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
        elif by == "checkpoint":
            ckpts = np.array(self._ckpt_ids)
            n_val = max(1, int(round(val_frac * len(ckpts))))
            val_ckpts = set(rng.permutation(ckpts)[:n_val].tolist())
            val_idx = np.array(
                [i for i, r in enumerate(self.records) if r.policy_ckpt_id in val_ckpts]
            )
            train_idx = np.array(
                [
                    i
                    for i, r in enumerate(self.records)
                    if r.policy_ckpt_id not in val_ckpts
                ]
            )
        else:
            raise ValueError(f"unknown split mode {by!r}")

        make = lambda ix: ContextDataset(  # noqa: E731
            self.store,
            [self.records[i] for i in ix],
            normalizer=self.normalizer,
            normalize=self.normalize,
        )
        return make(train_idx), make(val_idx)

    # ---- coverage diagnostics (SETUP.md 11) ---------------------------
    def contexts_per_sample(self) -> np.ndarray:
        counts = np.zeros(len(self.store), dtype=np.int64)
        for r in self.records:
            np.add.at(counts, r.batch_sample_ids, 1)
        return counts

    def checkpoints_per_sample(self) -> np.ndarray:
        seen: dict[int, set[str]] = defaultdict(set)
        for r in self.records:
            for sid in r.batch_sample_ids:
                seen[int(sid)].add(r.policy_ckpt_id)
        return np.array([len(seen[i]) for i in range(len(self.store))], dtype=np.int64)

    def cooccurrence_stats(self, max_pairs: int = 200_000) -> dict:
        """How often do two samples share a batch?

        A near-constant co-occurrence of 1 means batches are effectively
        disjoint; very high values mean the same composition is being resampled.
        Both break the multi-context premise, so we surface them.
        """
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        truncated = False
        for r in self.records:
            ids = np.sort(r.batch_sample_ids)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    pair_counts[(int(ids[a]), int(ids[b]))] += 1
                    if len(pair_counts) > max_pairs:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                break
        vals = np.array(list(pair_counts.values()) or [0])
        return {
            "n_distinct_pairs": int(len(pair_counts)),
            "cooccurrence_mean": float(vals.mean()),
            "cooccurrence_max": int(vals.max()),
            "duplicate_pair_frac": float((vals > 1).mean()),
            "truncated": truncated,
        }

    def coverage_report(self, min_contexts: int = 20) -> dict:
        cps = self.contexts_per_sample()
        kps = self.checkpoints_per_sample()
        covered = cps > 0
        report = {
            "n_samples": int(len(self.store)),
            "n_contexts": int(len(self.records)),
            "n_checkpoints": int(self.n_checkpoints),
            "mean_context_size": float(np.mean([len(r) for r in self.records])),
            "contexts_per_sample_mean": float(cps[covered].mean() if covered.any() else 0),
            "contexts_per_sample_min": int(cps[covered].min() if covered.any() else 0),
            "contexts_per_sample_max": int(cps.max()),
            "uncovered_samples": int((~covered).sum()),
            "frac_meeting_min_contexts": float((cps >= min_contexts).mean()),
            "checkpoints_per_sample_mean": float(kps[covered].mean() if covered.any() else 0),
            "min_contexts_target": int(min_contexts),
        }
        report.update(self.cooccurrence_stats())
        return report

    def histogram_contexts_per_sample(self, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        return np.histogram(self.contexts_per_sample(), bins=bins)

    # ---- io ------------------------------------------------------------
    def save_records(self, path: str | Path) -> None:
        save_records(self.records, path, normalizer=self.normalizer)


def save_records(
    records: list[ContextRecord],
    path: str | Path,
    normalizer: EffectNormalizer | None = None,
) -> None:
    """Store records in a ragged-safe flat layout (offsets + concatenated ids)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = np.array([len(r) for r in records], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    pf_dim = max((r.policy_features.shape[0] for r in records), default=0)
    pf = np.zeros((len(records), pf_dim), dtype=np.float32)
    for i, r in enumerate(records):
        pf[i, : r.policy_features.shape[0]] = r.policy_features
    np.savez_compressed(
        path,
        context_id=np.array([r.context_id for r in records], dtype=np.int64),
        policy_ckpt_id=np.array([r.policy_ckpt_id for r in records], dtype=str),
        utilization_rule_id=np.array([r.utilization_rule_id for r in records], dtype=str),
        estimator_id=np.array([r.estimator_id for r in records], dtype=str),
        batch_gain=np.array([r.batch_gain for r in records], dtype=np.float64),
        offsets=offsets,
        sample_ids=np.concatenate([r.batch_sample_ids for r in records]),
        effects=np.concatenate([r.per_sample_effects for r in records]),
        policy_features=pf,
        normalizer=np.array(
            [json.dumps(normalizer.to_dict() if normalizer else {})], dtype=str
        ),
    )


def load_records(path: str | Path) -> tuple[list[ContextRecord], EffectNormalizer | None]:
    d = np.load(path, allow_pickle=True)
    offsets = d["offsets"]
    records = []
    for i in range(len(d["context_id"])):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        records.append(
            ContextRecord(
                context_id=int(d["context_id"][i]),
                policy_ckpt_id=str(d["policy_ckpt_id"][i]),
                batch_sample_ids=d["sample_ids"][lo:hi],
                per_sample_effects=d["effects"][lo:hi],
                batch_gain=float(d["batch_gain"][i]),
                utilization_rule_id=str(d["utilization_rule_id"][i]),
                policy_features=d["policy_features"][i],
                estimator_id=str(d["estimator_id"][i]),
            )
        )
    nd = json.loads(str(d["normalizer"][0]))
    return records, (EffectNormalizer(**nd) if nd else None)
