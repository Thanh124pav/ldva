"""Stage 0 synthetic world (SETUP.md 4; PLAN.md 20 Phase 0).

The generative chain is deliberately the same one LDVA assumes:

    metadata m  ->  true latent factors u = f(m)  ->  observable chunk x

`f` is a fixed random tanh network: globally nonlinear (so a single global
metadata->latent map cannot work) but locally smooth, which is precisely
assumption A2 in PLAN.md 14 - small controllable metadata changes produce
locally predictable latent changes.

A chunk's observations are centred on `C u`, so the metadata controls *which
direction of policy space the sample carries information about*. Redundancy and
complementarity are then not hand-coded utility terms: they fall out of the
spectrum of the batch design matrix in the behaviour-cloning task next door
(`bc_task.py`). That keeps the Stage 0 gate honest - we are testing the real
mechanism, not a bespoke formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ldva.data.metadata import MetadataField, MetadataSpec
from ldva.data.samples import SampleStore


@dataclass
class SyntheticConfig:
    meta_dim: int = 3
    latent_dim: int = 4
    obs_dim: int = 6
    act_dim: int = 2
    chunk_len: int = 8
    hidden_dim: int = 16
    #: spread of observations around `C u` within a chunk
    obs_noise: float = 0.35
    #: expert action noise; sets the irreducible validation loss
    act_noise: float = 0.05
    #: scale of the metadata->latent network's first layer (locality knob)
    map_scale: float = 1.2
    seed: int = 0


@dataclass
class Region:
    """A named mode of the metadata distribution.

    Used both to build an intentionally *incomplete* initial dataset and by the
    Re-Mix-style domain-mixture baseline, which allocates across source regions
    rather than across learned latent directions.
    """

    name: str
    center: np.ndarray
    scale: np.ndarray
    weight: float = 1.0
    #: monetary cost per trajectory from this region (SETUP.md 31)
    cost: float = 1.0
    extra: dict = field(default_factory=dict)


class SyntheticWorld:
    """Fixed generative process plus the sampling helpers Stage 0 needs."""

    def __init__(self, cfg: SyntheticConfig | None = None):
        self.cfg = cfg or SyntheticConfig()
        c = self.cfg
        rng = np.random.default_rng(c.seed)

        self.metadata_spec = MetadataSpec(
            [
                MetadataField("object_x", -1.0, 1.0, cost=0.0),
                MetadataField("object_y", -1.0, 1.0, cost=0.0),
                MetadataField("difficulty", 0.0, 1.0, cost=0.0),
            ][: c.meta_dim]
        )
        if len(self.metadata_spec) != c.meta_dim:
            raise ValueError("meta_dim > number of declared metadata fields")

        # metadata -> latent: u = W2 tanh(W1 m_norm + b1)
        self.W1 = rng.normal(scale=c.map_scale, size=(c.hidden_dim, c.meta_dim))
        self.b1 = rng.normal(scale=0.3, size=c.hidden_dim)
        self.W2 = rng.normal(scale=1.0 / np.sqrt(c.hidden_dim), size=(c.latent_dim, c.hidden_dim))

        # latent -> observation mean, kept well conditioned so every latent
        # direction maps to a distinguishable policy-space direction
        C = rng.normal(size=(c.obs_dim, c.latent_dim))
        q, _ = np.linalg.qr(C)
        self.C = q[:, : c.latent_dim] * 2.0

        #: the expert whose actions the policy clones
        self.theta_star = rng.normal(scale=0.8, size=(c.act_dim, c.obs_dim))
        self._rng_seed = c.seed

    # ---- generative chain ---------------------------------------------
    def latent_from_metadata(self, m: np.ndarray) -> np.ndarray:
        """True latent factors u = f(m); accepts (d,) or (n, d)."""
        m = np.atleast_2d(np.asarray(m, dtype=np.float64))
        m_norm = self.metadata_spec.normalize(m) * 2.0 - 1.0  # centre on 0
        h = np.tanh(m_norm @ self.W1.T + self.b1)
        return h @ self.W2.T

    def latent_jacobian(self, m: np.ndarray) -> np.ndarray:
        """Analytic d u / d m at a single `m`; the ground truth the mapper fits."""
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        m_norm = self.metadata_spec.normalize(m) * 2.0 - 1.0
        pre = self.W1 @ m_norm + self.b1
        d_tanh = 1.0 - np.tanh(pre) ** 2
        # chain through the normalization: d m_norm / d m = 2 / span
        d_norm = 2.0 / self.metadata_spec.span
        return (self.W2 * d_tanh) @ self.W1 * d_norm

    def generate_chunks(
        self, m: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Metadata -> (obs, act, true latent) for `n` chunks."""
        c = self.cfg
        m = np.atleast_2d(np.asarray(m, dtype=np.float64))
        n = m.shape[0]
        u = self.latent_from_metadata(m)  # (n, d_u)

        # observations spread around C u; `difficulty` widens the spread, which
        # is what makes some regions intrinsically noisier to learn from
        center = u @ self.C.T  # (n, obs_dim)
        if c.meta_dim >= 3:
            diff = self.metadata_spec.normalize(m)[:, 2]
        else:
            diff = np.zeros(n)
        spread = c.obs_noise * (1.0 + diff)[:, None, None]
        obs = center[:, None, :] + spread * rng.normal(size=(n, c.chunk_len, c.obs_dim))

        act = obs @ self.theta_star.T + c.act_noise * rng.normal(
            size=(n, c.chunk_len, c.act_dim)
        )
        return obs.astype(np.float32), act.astype(np.float32), u.astype(np.float32)

    # ---- dataset construction -----------------------------------------
    def sample_metadata(
        self,
        n: int,
        rng: np.random.Generator,
        regions: list[Region] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw metadata, optionally from a mixture of regions.

        Returns `(metadata, region_index)`; region index is -1 for uniform draws.
        """
        if not regions:
            return self.metadata_spec.sample(n, rng), np.full(n, -1, dtype=np.int64)
        w = np.array([r.weight for r in regions], dtype=np.float64)
        w = w / w.sum()
        which = rng.choice(len(regions), size=n, p=w)
        m = np.empty((n, len(self.metadata_spec)))
        for k, r in enumerate(regions):
            sel = which == k
            if not sel.any():
                continue
            m[sel] = r.center + r.scale * rng.normal(size=(int(sel.sum()), len(self.metadata_spec)))
        return self.metadata_spec.clip(m), which

    def build_store(
        self,
        n: int,
        rng: np.random.Generator,
        regions: list[Region] | None = None,
        round_id: int = 0,
        metadata: np.ndarray | None = None,
    ) -> SampleStore:
        """Create a `SampleStore` of `n` chunks (or from explicit `metadata`)."""
        if metadata is None:
            metadata, which = self.sample_metadata(n, rng, regions)
        else:
            metadata = self.metadata_spec.clip(np.atleast_2d(metadata))
            n = metadata.shape[0]
            which = np.full(n, -1, dtype=np.int64)
        obs, act, u = self.generate_chunks(metadata, rng)
        cost = np.array(
            [regions[i].cost if (regions and i >= 0) else 1.0 for i in which],
            dtype=np.float64,
        )
        return SampleStore(
            obs=obs,
            act=act,
            metadata=metadata,
            metadata_spec=self.metadata_spec,
            trajectory_id=np.arange(n),
            start_t=np.zeros(n, dtype=np.int64),
            task_id=np.maximum(which, 0),
            reward=np.zeros(n, dtype=np.float32),
            success=np.zeros(n, dtype=bool),
            policy_ckpt_id=np.array([f"gen_r{round_id}"] * n, dtype=object),
            round_id=np.full(n, round_id, dtype=np.int64),
            cost=cost,
            latent_true=u,
        )

    # ---- canonical Stage 0 setups --------------------------------------
    def default_initial_regions(self) -> list[Region]:
        """An intentionally incomplete initial distribution.

        Two tight modes in one corner of metadata space. The evaluation
        distribution (below) is uniform over the whole box, so the only way to
        improve the policy is to *expand support outward* - which is the
        behaviour LDVA is supposed to produce, and which pure resampling of the
        existing data cannot.
        """
        d = len(self.metadata_spec)
        lo, hi = self.metadata_spec.low, self.metadata_spec.high
        mid = (lo + hi) / 2
        span = (hi - lo) / 2
        c1 = mid - 0.55 * span
        c2 = mid.copy()
        c2[0] = mid[0] - 0.5 * span[0]
        c2[1] = mid[1] + 0.45 * span[1]
        s = 0.12 * span
        return [
            Region("mode_a", c1, s, weight=0.6, cost=1.0),
            Region("mode_b", c2, s, weight=0.4, cost=1.0),
        ]

    def evaluation_metadata(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Fixed evaluation distribution, declared *before* acquisition.

        SETUP.md 33 is explicit that the test distribution must not move after
        seeing what the planner acquires, so this is uniform over the full
        metadata box and never parameterized by the acquired data.
        """
        return self.metadata_spec.sample(n, rng)
