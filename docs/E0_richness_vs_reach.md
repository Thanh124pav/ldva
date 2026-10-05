# Richness versus reach: why the two conditions LDVA needs exclude each other

Follow-up to [`E0_root_cause.md`](E0_root_cause.md), which ended with the
finding that the synthetic world's generative map *compresses* — the true
latents carried 1.38 effective dimensions against the metadata's 2.27 — so
acting in latent space was intrinsically less controllable than acting in
metadata space, inverting the premise the benchmark exists to test.

This pass asked whether that can be fixed, found that it can, and found that
fixing it breaks something else. The trade-off appears to be structural rather
than an implementation detail.

## 1. A reasoning error, corrected

I claimed a 3-dimensional metadata space cannot produce more than 3 latent
dimensions. That is true of the **intrinsic** dimension — the image of a smooth
map from a 3-manifold is a 3-manifold (Whitney) — but false of the
**participation ratio**, which is a *linear* second-moment measure. A curved
manifold need not lie in any low-dimensional linear subspace: the helix
`t -> (cos t, sin t, t)` is one-dimensional yet spans three linear dimensions.

Measured, from the same 3-dimensional metadata (D_0 at 2.26 effective
dimensions):

| map | effective dimensions |
|---|---|
| linear | 2.24 |
| tanh MLP, `map_scale` 0.3 / 1.2 / 3.0 | 1.90 / 2.01 / 1.99 |
| degree-3 polynomial | 2.57 |
| random Fourier, 8 freqs, omega=8 | **11.18** |
| random Fourier, 32 freqs, omega=8 | **19.68** |

So richness *is* reachable; the original map simply could not produce it. A
tanh MLP caps near 2.0 whatever its scale, because raising the scale
**saturates** the activation rather than curving it, and a saturated map loses
variation instead of spreading it. Saturation is not curvature.

## 2. The theory that applies

- **Whitney / Nash embedding** — the intrinsic dimension is invariant. Every
  extra dimension a map produces is *extrinsic*, bought with curvature. This is
  what makes `latent_dim` alone useless: the map must be curved *and* the output
  wide enough to hold the result.
- **Bochner / random Fourier features** (Rahimi–Recht) — the expansion above
  approximates a shift-invariant kernel whose bandwidth is `1/omega`. Richness
  and locality are therefore governed by **one** parameter, which is why they
  trade off as a matter of mathematics and not of tuning.
- **Reach** (Federer; Niyogi–Smale–Weinberger) — the radius within which a
  curved manifold is well approximated by its tangent space. This is exactly
  the quantity that should set `MetadataMapper`'s step, since the mapper solves
  `J delta_m ~ alpha v` with a local linear `J`.
- **Johnson–Lindenstrauss** — random projection preserves angles for
  `k >~ log(n)/eps^2`. Relevant to the "expand then compress" idea below, and
  the reason that idea is safe but unhelpful.

## 3. The trade-off, measured

Random Fourier map from 3-dimensional metadata, step in normalised metadata
units:

| omega | effective dim | local-linear R² at step 0.2 | reach (R² > 0.95) |
|---|---|---|---|
| 0.5 | 2.05 | 0.996 | <= 0.4 |
| 1.0 | 1.95 | 0.983 | <= 0.2 |
| 2.0 | 3.19 | 0.906 | <= 0.1 |
| 4.0 | 6.45 | 0.713 | <= 0.05 |
| 8.0 | 14.40 | 0.397 | <= 0.02 |
| 16.0 | 23.86 | 0.103 | <= 0.02 |

LDVA needs both:

1. latent geometry **richer** than metadata, or planning in it buys nothing —
   needs high `omega`;
2. the metadata→latent map **locally linear**, or directions cannot be turned
   into collection requests — needs low `omega`.

At `omega <= 1` the latent is *poorer* than the metadata (≈2.0 against 2.26),
which is the state the benchmark was in and under which the premise cannot be
tested at all. At `omega >= 8` the latent is rich but the local Jacobian is
valid only out to ~0.02.

The crossing point is near `omega = 2`, where effective dimensionality first
exceeds the metadata's (3.19 against 2.26) while local R² is still 0.906 —
provided the step stays under 0.1. The existing `delta_scale = 0.4` produces
longer steps than that, so the step has to be chosen from the measured reach
rather than fixed by hand.

## 4. "Expand then compress" does not recover reach

Tested directly, since it is the natural idea: expand with a high-frequency map
for richness, then project back down so the mapper has fewer dimensions to
invert.

| omega=8, compressed to | effective dim | local R² at step 0.2 |
|---|---|---|
| no compression (32) | **18.16** | 0.266 |
| PCA → 8 | 7.51 | 0.281 |
| PCA → 4 | 3.94 | 0.325 |
| PCA → 3 | 2.98 | **0.350** |

Compressing 32 dimensions down to 3 moved local R² only from 0.27 to 0.35
while destroying the richness it was meant to preserve. The theory says why:
`J_total = J_compress · J_expand`, and a **fixed** linear projection of the
output cannot change how fast the Jacobian turns with the *input* — which is
what reach depends on. JL guarantees the compression is geometrically safe; it
simply is not the binding constraint.

What the same table does show is that at `omega=8` local R² reaches 0.903 once
the step is **0.05** rather than 0.2. The map is not too curved to use; the
step was too long for it.

## 5. What was coded

- `SyntheticConfig.map_kind = "rff"` with `rff_omega`, `rff_features`: a
  genuinely curved metadata→latent map, alongside the original `"tanh"`.
- `latent_jacobian` extended to the Fourier map; verified against numerical
  differentiation to a relative error of 1e-11 for both map kinds.
- `obs_dim >= latent_dim` is now enforced with an explanatory error: the
  latent→observation map is orthonormalised, so a latent wider than the
  observation would silently collapse directions the planner then could not
  distinguish.
- `estimate_reach()` in `ldva/analysis/direction_validation.py`, and
  `--auto-step` in the ceiling diagnostic, which sets the mapper's trust region
  from the measured reach instead of the hand-picked 0.35.

### The specificity statistic, corrected twice

Criterion 6 is scored against a permutation null — the other candidate
directions — rather than a raw cosine. Choosing the right statistic over that
null took two attempts, both caught by measurement:

1. **An isotropic null** (random unit vectors in the full space) inflated the
   score threefold, because both the realized displacement and every candidate
   direction live in the same collapsed subspace while an isotropic draw does
   not. It measured the collapse, not the control.
2. **A z-score** over the permutation null broke at both extremes. When the
   candidates are near-identical its denominator vanishes and the score
   explodes — a sweep produced 2e8. When they are orthogonal the denominator
   also vanishes, but that is the *best* case. The null's spread is information
   about how diverse the direction set is; it is not measurement noise and does
   not belong in a denominator.

The statistic is now the **gap** in cosine units:

    gap_i = cos(delta_i, v_i) - mean_j |cos(delta_i, v_j)|

which handles every case on one scale, verified by unit test:

| case | cosine | null | **gap** |
|---|---|---|---|
| orthogonal directions, executed perfectly | +1.000 | 0.000 | **+1.000** |
| collapsed space, all near-parallel | +0.994 | 0.996 | **−0.001** |
| direction realized worse than the alternatives | +0.000 | 0.200 | **−0.200** |

Criterion 6 now requires `gap >= 0.3`, which is about 1.5x the measured
replicate noise of the realized cosine (~0.2). That calibration is a judgement
call and is stated as one.

## 6. A third constraint, not yet confirmed

A single quick run at `omega = 4` showed the expected pattern — latent richer
than metadata for the first time (4.49 against 2.22), candidate directions far
less redundant (mean pairwise |cos| 0.242 against 0.53) — but also two things
that were not expected: the realized cosine fell from 0.94 to 0.385, and the
**effect labels' effective dimensionality fell to 1.50** from the 38–64
measured on the tanh world.

If that holds, the problem is three-way rather than two-way: latent richness,
reach, **and the learnability of the effect structure**. A high-frequency map
makes the latent oscillate quickly in the metadata, which may make per-sample
effects nearly indistinguishable and leave nothing for the data model to learn.
A sweep over `omega` at full config is running to confirm it; the numbers above
are from one `--quick` seed and should not be quoted.

## 7. Consequences for the hypothesis

The premise "plan in the latent effect space rather than in metadata space"
needs a condition it did not previously state: **the latent geometry must be
richer than the metadata parameterisation, and still locally mappable from it.**
On this benchmark those two requirements have a narrow overlap, and whether any
point inside it also keeps the effect labels learnable is the open question.

That sharpens rather than refutes the hypothesis, and it predicts where LDVA
should pay off: settings where the metadata parameterisation is genuinely poor —
high-dimensional, redundant, or not directly controllable, as with teleoperated
real-robot data — rather than a clean 3-parameter simulator, where acting
directly in metadata space is both easier and more specific (gap 5/6 seeds
passing against 0/6 for the learned encoding).
