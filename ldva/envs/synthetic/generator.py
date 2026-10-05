"""Stage 0 synthetic world (PLAN.md 14, 20 Phase 0).

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
    #: Scale of the metadata->latent network's first layer: the locality knob,
    #: and also what controls how much tanh saturates.
    #:
    #: Lowered from 1.2 after measuring that saturation was compressing the
    #: generative map. Mean |tanh| rises 0.31 -> 0.78 as this goes 0.3 -> 2.4,
    #: and the true latents' effective dimensionality falls with it, so the
    #: world generated latents *poorer* than the metadata they came from (1.31
    #: vs 2.26) - inverting the premise the benchmark exists to test.
    #:
    #: 0.3 with `well_conditioned_map` gives latent effective dimensionality
    #: 1.94 against metadata's 2.26 (ratio 0.86). Locality survives: one global
    #: linear map still explains only R^2 = 0.737 of the variation over the box,
    #: and the local Jacobian at D_0 differs from the global one by 1.34 in
    #: relative Frobenius norm, so the metadata mapper still faces a genuinely
    #: curved map rather than a linear one.
    map_scale: float = 0.3
    #: Make the metadata->latent map locally isometric at the centre of the box.
    #:
    #: With a random `W2` the composition `W2 tanh(W1 m)` is badly conditioned:
    #: its singular values are spread, so one latent direction carries most of
    #: the variance. Measured consequence - the world's own true latents came
    #: out at 1.38 effective dimensions from metadata carrying 2.27, i.e. the
    #: generative map *compressed*. On such a world latent-space directions are
    #: intrinsically less distinguishable than metadata-space ones, which
    #: inverts the premise the benchmark exists to test: direction specificity
    #: scored +2.22 in metadata space and +1.87 on the true latents.
    #:
    #: Note this is NOT about `latent_dim`: the image of a `meta_dim`
    #: dimensional metadata space is at most `meta_dim` dimensional whatever
    #: the latent width is, so raising `latent_dim` cannot fix it. The same
    #: reasoning is already applied one step later, where `C` is orthonormalised
    #: "so every latent direction maps to a distinguishable policy-space
    #: direction" - this applies it to the step before.
    well_conditioned_map: bool = True
    #: Form of the metadata->latent map.
    #:
    #: `"tanh"` is `W2 tanh(W1 m + b1)`, the original. Measured limitation: its
    #: effective dimensionality caps at ~2.0 whatever `map_scale` is, because
    #: raising the scale *saturates* tanh rather than curving it, and a flat map
    #: loses variation instead of spreading it. The world then generates latents
    #: poorer than the metadata they came from, which inverts the premise the
    #: benchmark exists to test.
    #:
    #: `"rff"` is random Fourier features, `[cos(m W), sin(m W)]` with
    #: `W ~ N(0, rff_omega^2)`. By Bochner's theorem this approximates a
    #: shift-invariant kernel whose bandwidth is `1/rff_omega`, so a single
    #: parameter controls both how rich the geometry is and how local it is -
    #: which is why those two properties trade off rather than being
    #: independently tunable. Measured from 3-d metadata: effective
    #: dimensionality 2.0 at omega=1, 3.2 at omega=2, 14.4 at omega=8.
    #:
    #: Note the intrinsic dimension stays 3 whatever the map (Whitney): the
    #: extra dimensions are *extrinsic*, produced by curvature. That is also
    #: why `latent_dim` alone cannot buy them - the map has to be curved AND
    #: the output wide enough to hold the result.
    map_kind: str = "tanh"
    #: inverse kernel bandwidth for `map_kind="rff"`; higher = richer but less
    #: locally linear, so the mapper's usable step shrinks with it
    rff_omega: float = 2.0
    #: number of random frequencies; the feature count is twice this
    rff_features: int = 16
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
    #: monetary cost per trajectory from this region (PLAN.md 10)
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

        if c.map_kind not in ("tanh", "rff"):
            raise ValueError(
                f"unknown map_kind {c.map_kind!r}; expected 'tanh' or 'rff'")
        if c.obs_dim < c.latent_dim:
            raise ValueError(
                f"obs_dim ({c.obs_dim}) must be >= latent_dim ({c.latent_dim}): "
                "the latent->observation map is orthonormalised, so a latent "
                "wider than the observation cannot be embedded without "
                "collapsing directions that the planner would then be unable "
                "to distinguish")

        # metadata -> latent
        if c.map_kind == "rff":
            # [cos(m Omega), sin(m Omega)] -> W2 -> latent. Curvature comes from
            # the frequencies, not from a saturating activation.
            self.Omega = rng.normal(
                scale=c.rff_omega, size=(c.meta_dim, c.rff_features))
            self.W1, self.b1 = None, None
            self.W2 = rng.normal(
                scale=1.0 / np.sqrt(2 * c.rff_features),
                size=(c.latent_dim, 2 * c.rff_features))
        else:
            # u = W2 tanh(W1 m_norm + b1)
            self.Omega = None
            self.W1 = rng.normal(scale=c.map_scale, size=(c.hidden_dim, c.meta_dim))
            self.b1 = rng.normal(scale=0.3, size=c.hidden_dim)
            self.W2 = rng.normal(
                scale=1.0 / np.sqrt(c.hidden_dim), size=(c.latent_dim, c.hidden_dim))
        if c.well_conditioned_map:
            self.W2 = self._condition_W2()

        # latent -> observation mean, kept well conditioned so every latent
        # direction maps to a distinguishable policy-space direction
        C = rng.normal(size=(c.obs_dim, c.latent_dim))
        q, _ = np.linalg.qr(C)
        self.C = q[:, : c.latent_dim] * 2.0

        #: the expert whose actions the policy clones
        self.theta_star = rng.normal(scale=0.8, size=(c.act_dim, c.obs_dim))
        self._rng_seed = c.seed

    # ---- generative chain ---------------------------------------------
    def _condition_W2(self, n_mc: int = 4096) -> np.ndarray:
        """Rescale `W2` so latent variance is spread evenly over the box.

        Whitens `u` against its **empirical covariance over the metadata box**,
        estimated by Monte Carlo:  `M = U S^-1/2 U^T` from the eigendecomposition
        of `cov(u)`, plus the identity off `span(U)` so `W2` stays full rank.
        After this, `cov(u)` is isotropic on the box, so no single latent
        direction carries most of the variation.

        **A first attempt whitened the Jacobian at the box centre and did
        nothing** (latent effective dimensionality 1.15 -> 1.11), because D_0
        occupies a *corner* sub-box: conditioning the map at the centre leaves
        it ill-conditioned where the data actually is. Matching the covariance
        over the region that gets sampled is what the property requires.

        Only conditioning changes. The tanh nonlinearity is untouched, so the
        map stays curved away from any given point and the local Jacobian
        remains an approximation - which is the thing the metadata mapper is
        supposed to cope with.
        """
        c = self.cfg
        rng = np.random.default_rng(c.seed + 9176)
        lo, hi = self.metadata_spec.low, self.metadata_spec.high
        m = rng.uniform(lo, hi, size=(n_mc, len(self.metadata_spec)))
        u = self._features(m) @ self.W2.T
        cov = np.cov(u - u.mean(0), rowvar=False)
        w, v = np.linalg.eigh(np.atleast_2d(cov))
        keep = w > 1e-12 * max(w.max(), 1e-30)
        if not np.any(keep):
            return self.W2
        vk, wk = v[:, keep], w[keep]
        trans = vk @ np.diag(wk ** -0.5) @ vk.T + (
            np.eye(c.latent_dim) - vk @ vk.T)
        return trans @ self.W2

    def _features(self, m: np.ndarray) -> np.ndarray:
        """The map's hidden features, before the final linear layer."""
        m = np.atleast_2d(np.asarray(m, dtype=np.float64))
        m_norm = self.metadata_spec.normalize(m) * 2.0 - 1.0  # centre on 0
        if self.cfg.map_kind == "rff":
            proj = m_norm @ self.Omega
            return np.concatenate([np.cos(proj), np.sin(proj)], axis=1)
        return np.tanh(m_norm @ self.W1.T + self.b1)

    def latent_from_metadata(self, m: np.ndarray) -> np.ndarray:
        """True latent factors u = f(m); accepts (d,) or (n, d)."""
        return self._features(m) @ self.W2.T

    def latent_jacobian(self, m: np.ndarray) -> np.ndarray:
        """Analytic d u / d m at a single `m`; the ground truth the mapper fits."""
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        m_norm = self.metadata_spec.normalize(m) * 2.0 - 1.0
        # chain through the normalization: d m_norm / d m = 2 / span
        d_norm = 2.0 / self.metadata_spec.span
        if self.cfg.map_kind == "rff":
            # features are [cos(m O), sin(m O)], so d/dm is
            # [-sin(m O) * O, cos(m O) * O] stacked the same way
            proj = m_norm @ self.Omega
            d_feat = np.concatenate(
                [-np.sin(proj)[:, None] * self.Omega.T,
                 np.cos(proj)[:, None] * self.Omega.T],
                axis=0,
            )
            return (self.W2 @ d_feat) * d_norm
        pre = self.W1 @ m_norm + self.b1
        d_tanh = 1.0 - np.tanh(pre) ** 2
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
        """An intentionally incomplete initial distribution - of limited
        *extent*, but **full rank**.

        D_0 has to leave somewhere to expand into: the evaluation distribution
        is uniform over the whole box, so coverage must be partial or the
        experiment is vacuous. The previous version achieved that with two
        tight modes in one corner, and that turned out to be degenerate in a
        way that silently capped criterion 6.

        Measured: two tight modes put the metadata of D_0 on a manifold of
        1.16 effective dimensions out of 3. Local PCA inside such a support
        yields candidate directions spanning only ~2.5 effective dimensions
        with a mean pairwise |cos| of ~0.5, so many candidates are near
        duplicates of each other. Direction *specificity* - whether collection
        moved the latents along the direction that was asked for rather than
        one that was not - then cannot exceed about 1.9 standard deviations
        even when execution is perfect: in raw metadata space the realized
        cosine is 1.000 and the z-score is still only 1.79-1.91 across seeds.
        The ceiling came from the geometry of D_0, not from the model, the
        effect labels (effective dimensionality 38-60) or the metadata mapper.

        So the modes are now offset along *different* axes and given anisotropic
        widths, which keeps the support inside a corner of the box while
        spanning all metadata dimensions. Extent stays incomplete; rank does
        not collapse.
        """
        lo, hi = self.metadata_spec.low, self.metadata_spec.high
        d = len(self.metadata_spec)
        span = hi - lo

        # The support is confined to a CORNER SUB-BOX covering `frac` of each
        # axis, so coverage stays genuinely incomplete: the evaluation
        # distribution is uniform over the whole box and most of it is
        # unreachable without expanding support. Within that sub-box the modes
        # are offset along *different* axes with anisotropic widths, so the
        # support has full rank.
        #
        # Both properties are needed and they pull against each other. Two
        # tight modes in one corner gave incomplete extent but a metadata
        # manifold of 1.16 effective dimensions out of 3, and that alone caps
        # direction specificity at ~1.9 standard deviations even when
        # execution is perfect (realized cosine 1.000 in raw metadata space).
        # Spreading the modes over the whole box fixes the rank but covers 43%
        # of the box volume with a near-central centroid, which leaves
        # acquisition nothing to expand into and makes the experiment vacuous.
        frac = 0.45
        base = lo + 0.02 * span          # just inside the low corner
        sub = frac * span                # extent of the sub-box
        regions: list[Region] = []
        n_modes = max(min(d, 3), 2)
        for k in range(n_modes):
            c = base + 0.5 * sub         # centre of the sub-box
            ax = k % d
            c[ax] = base[ax] + 0.18 * sub[ax]
            c[(ax + 1) % d] = base[(ax + 1) % d] + 0.82 * sub[(ax + 1) % d]
            w = 0.10 * sub
            w[ax] = 0.20 * sub[ax]
            regions.append(
                Region(f"mode_{chr(ord('a') + k)}", c, w,
                       weight=1.0 / n_modes, cost=1.0)
            )
        return regions

    def evaluation_metadata(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Fixed evaluation distribution, declared *before* acquisition.

        PLAN.md 14 is explicit that the test distribution must not move after
        seeing what the planner acquires, so this is uniform over the full
        metadata box and never parameterized by the acquired data.
        """
        return self.metadata_spec.sample(n, rng)
