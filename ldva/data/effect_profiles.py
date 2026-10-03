"""Effect-distance table for the metric loss (PLAN.md 5.3).

`d_effect(i, j) = average over shared contexts c of |e_i(c) - e_j(c)|`.

Only *shared* contexts are used: comparing a sample's effect in one batch with
another sample's effect in a different batch would mostly measure the batch, not
the pair. That makes the table sparse - a pair is usable only if the two samples
co-occur often enough - so the sampler below returns pairs with at least
`min_shared` shared contexts, and `L_metric` is simply skipped when no such
pair exists in a minibatch.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ldva.data.context_dataset import ContextRecord


class EffectProfileTable:
    def __init__(self, records: list[ContextRecord], n_samples: int, min_shared: int = 2):
        self.n_samples = int(n_samples)
        self.min_shared = int(min_shared)

        #: sample id -> {context index -> effect}
        profiles: list[dict[int, float]] = [dict() for _ in range(self.n_samples)]
        for ci, r in enumerate(records):
            for sid, eff in zip(r.batch_sample_ids, r.per_sample_effects):
                profiles[int(sid)][ci] = float(eff)
        self.profiles = profiles

        # accumulate |e_i(c) - e_j(c)| over every context the pair shares
        acc: dict[tuple[int, int], list[float]] = defaultdict(list)
        for ci, r in enumerate(records):
            ids = r.batch_sample_ids
            effs = r.per_sample_effects
            order = np.argsort(ids)
            ids, effs = ids[order], effs[order]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    acc[(int(ids[a]), int(ids[b]))].append(abs(float(effs[a] - effs[b])))

        self.pairs = np.array(
            [k for k, v in acc.items() if len(v) >= self.min_shared], dtype=np.int64
        ).reshape(-1, 2)
        self.distances = np.array(
            [float(np.mean(v)) for v in acc.values() if len(v) >= self.min_shared],
            dtype=np.float64,
        )
        self.shared_counts = np.array(
            [len(v) for v in acc.values() if len(v) >= self.min_shared], dtype=np.int64
        )
        self._index = {
            (int(i), int(j)): k for k, (i, j) in enumerate(self.pairs)
        }
        #: normalization constant so target distances live on a comparable scale
        self.scale = float(np.median(self.distances)) if len(self.distances) else 1.0
        if self.scale <= 0:
            self.scale = 1.0

    def __len__(self) -> int:
        return len(self.pairs)

    def distance(self, i: int, j: int) -> float | None:
        key = (i, j) if i < j else (j, i)
        k = self._index.get(key)
        return None if k is None else float(self.distances[k])

    def normalized_distance(self, i: int, j: int) -> float | None:
        d = self.distance(i, j)
        return None if d is None else d / self.scale

    def lookup_batch(self, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized lookup; returns (normalized distances, found mask)."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        lo = np.minimum(pairs[:, 0], pairs[:, 1])
        hi = np.maximum(pairs[:, 0], pairs[:, 1])
        out = np.zeros(len(pairs), dtype=np.float64)
        found = np.zeros(len(pairs), dtype=bool)
        for k, (i, j) in enumerate(zip(lo, hi)):
            idx = self._index.get((int(i), int(j)))
            if idx is not None:
                out[k] = self.distances[idx] / self.scale
                found[k] = True
        return out, found

    def sample_pairs(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw `n` usable pairs with their normalized effect distances."""
        if len(self.pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
        idx = rng.integers(0, len(self.pairs), size=n)
        return self.pairs[idx], self.distances[idx] / self.scale

    def effect_profile_matrix(self, ids: np.ndarray, context_ids: np.ndarray):
        """Dense (len(ids), len(context_ids)) matrix with NaN where unobserved.

        Used by the latent-geometry diagnostics for neighbor-consistency, which
        needs whole profiles rather than pairwise distances.
        """
        ids = np.asarray(ids)
        cmap = {int(c): k for k, c in enumerate(context_ids)}
        out = np.full((len(ids), len(context_ids)), np.nan)
        for r, sid in enumerate(ids):
            for ci, eff in self.profiles[int(sid)].items():
                k = cmap.get(int(ci))
                if k is not None:
                    out[r, k] = eff
        return out

    def report(self) -> dict:
        return {
            "n_usable_pairs": int(len(self.pairs)),
            "min_shared_contexts": self.min_shared,
            "shared_contexts_mean": float(
                self.shared_counts.mean() if len(self.shared_counts) else 0.0
            ),
            "effect_distance_median": self.scale,
        }
