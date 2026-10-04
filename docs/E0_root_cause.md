# Criterion 6: root cause, fixes, and what remains

Third and deepest pass over the preflight gate. [`E0_diagnosis.md`](E0_diagnosis.md)
found criterion 5's failure to be a measurement artefact;
[`E0_criteria_resolution.md`](E0_criteria_resolution.md) fixed that and
established a fit-versus-controllability trade-off. This pass found that the
trade-off was mostly an artefact too, located the actual cause, fixed it, and
narrowed what is left to a single step of the pipeline.

Raw output: [`results/E0_rootcause/`](results/E0_rootcause/).

**Read this first if you are developing assumptions.** The useful content is as
much the list of rejected explanations as the surviving one — six candidate
causes were measured and ruled out, and three of my own conclusions were
retracted.

## 1. Criterion 6 was not measuring control

The criterion was `mean cos(desired, realized) > 0.3`. A raw cosine is not
interpretable on its own: between two vectors in `d` dimensions the chance
level is about `sqrt(2/(pi*d))`, so a collapsed latent space inflates it for
free. Measured on 8 seeds, the encoder used a participation ratio of **1.0–2.2
out of 32** latent dimensions.

The consequence, measured rather than argued: a realized cosine of **+0.910**
— which reads as near-perfect control — sat only **1.3 standard deviations**
above requesting a *different* direction. Across 16 cells spanning two fit
levels and 8 seeds, **no cell reached 2 sd**.

So criterion 6 overstated control, and its 0.3 threshold sat *below* chance
when the space was collapsed: a degenerate representation passed while a
richer one failed.

### The null matters, and my first one was wrong

I first scored each direction against random unit vectors drawn isotropically
in the full 32-dimensional space. That inflated the z-scores threefold
(+2.64/+3.96 instead of +1.13/+1.35), because both the realized displacement
*and* every candidate direction live inside the same collapsed subspace while
an isotropic draw almost never does — it measured the collapse, not the
control.

The right null is a **permutation over the other candidate directions**: did
collection move the latents along what was asked for, rather than along
something else that could have been asked for? That controls for
dimensionality and for the structure of the direction set at once.

`ldva.analysis.direction_validation.direction_specificity` implements it, with
a unit test on the case that matters:

| synthetic case | raw cosine | specificity z |
|---|---|---|
| collapsed space, all directions near-parallel | **0.988** | 0.50 |
| rich space, orthogonal directions executed correctly | 0.970 | **11.57** |

The old rule passes both. Criterion 6 now requires `z >= 2`.

## 2. Six candidate causes, measured and ruled out

| candidate | measurement | verdict |
|---|---|---|
| effect labels too poor | effect-profile matrix has **38–64** effective dimensions, top singular value only 5–13% of variance | **rejected** — supervision is rich |
| metadata mapper / execution | realized cosine in raw metadata space is **1.000** | **rejected** — execution is exact |
| `L_metric` weight too low | swept 0 / 0.1 / 1 / 5: eff_dim 1.67 / 1.98 / 1.87 / 1.98, rho(weight, eff_dim) = **+0.24** | **rejected** |
| latent capacity | `latent_dim` 8 vs 32 gives eff_dim 1.80 vs 1.98 | **rejected** — collapse is absolute, not proportional |
| the encoder alone | the world's **ground-truth** latents also failed (z = 1.70–1.89) | **rejected as sole cause** |
| overshoot (realized/planned step) | rho(C6, overshoot) = **−0.23**, flat across step lengths | **rejected** (third measurement to do so) |

The decisive observation: in raw metadata space the realized cosine was
**1.000** and the specificity z-score was still only **1.79–1.91**. When
execution is perfect and the measure still fails, the limit cannot be in
execution — it is that the **candidate directions are not distinguishable from
each other**.

## 3. Root cause: D_0's geometry was rank-deficient

`default_initial_regions()` put D_0 in two tight modes in one corner. Coverage
has to be incomplete — the evaluation distribution is uniform over the whole
box, so acquisition needs somewhere to expand — but two tight modes also put
the metadata on a manifold of **1.16 effective dimensions out of 3**.

Local PCA inside a rank-deficient support yields near-duplicate candidate
directions (measured: ~12–15 candidates spanning ~2.5 effective dimensions,
mean pairwise |cos| ≈ 0.5). The permutation null is then full of near-copies of
the direction being tested, which caps specificity mechanically.

### Fix 1: D_0 is incomplete in *extent* but full *rank*

The two properties pull against each other, and I got it wrong once in between:

| version | box volume covered | off-centre | eff_dim (of 3) |
|---|---|---|---|
| original (2 modes, one corner) | — | — | **1.17** ✗ rank-deficient |
| my over-correction | **0.43** | 0.10 ✗ nothing to expand into | 2.23 |
| **final** | **0.12–0.15** | 0.23–0.28 | **2.21–2.33** |

Modes now sit inside a corner sub-box (45% of each axis) but are offset along
*different* axes with anisotropic widths. `tests/test_env_adapters.py` asserts
both properties, so a future change cannot quietly trade one for the other.

### Fix 2: `DirectionConfig.max_pairwise_cosine = 0.8`

Rejects a candidate pointing the same way as one already accepted, across
clusters as well as within one. The comparison is **signed**: `+v` and `−v`
have |cos| = 1 but are opposite requests, and an absolute-value test deleted
half of every candidate set — the test suite caught that immediately.

## 4. Result after the fixes

Six seeds, three representations, each with the whole pipeline
(clustering → directions → metadata mapping → realized movement):

| representation | eff_dim | direction-set dim | mean pairwise \|cos\| | **specificity z** | realized cos | **passes** |
|---|---|---|---|---|---|---|
| raw metadata | 2.27 | 2.51 | 0.447 | **+2.22** | 1.000 | **5 / 6** |
| world's true latents | 1.38 | 2.13 | 0.529 | +1.87 | 0.941 | 1 / 6 |
| learned encoding | 1.92 | 2.41 | 0.425 | +1.05 | 0.471 | 0 / 6 |

Metadata space went from **0/6 passing (z 1.79–1.91) to 5/6 (z +2.22)**. The
ceiling was real and the D_0 fix lifted it.

**What remains is a representation gap**, and it is now cleanly localised:
specificity falls from +2.22 in metadata space to +1.05 after encoding, and the
realized cosine from 1.000 to 0.471. The direction machinery and the execution
are sound; the loss happens in the encoder.

## 5. The finding that matters most for the hypothesis

The world's **ground-truth** latents also largely fail (z = +1.87, 1/6), and
they carry **fewer** effective dimensions than the metadata they were generated
from (1.38 vs 2.27).

That is a property of the benchmark's generative map, not of any model: the
synthetic world compresses 3 metadata dimensions into ~1.4 effective latent
dimensions. On this world, acting in latent space is therefore *intrinsically*
less controllable than acting in metadata space — which runs against LDVA's
premise that the latent effect geometry is the right space to plan in.

Three readings are open, and they are not equivalent:

1. **The synthetic world is too compressive to test the premise.** Its
   metadata→latent map should be made closer to full rank, or the benchmark
   replaced by DMC/MetaWorld where effect geometry is richer. This is cheap to
   check: raise `SyntheticConfig.latent_dim` / `map_scale` and re-measure the
   true-latent ceiling.
2. **The premise needs qualifying.** Latent-space planning may only pay off
   where the metadata parameterisation is poor (high-dimensional, redundant, or
   not directly controllable) — which is the real-robot case, not this one.
   That would be a scope claim to state explicitly rather than a bug.
3. **The encoder is losing recoverable structure.** The gap between the true
   latents (+1.87) and the learned ones (+1.05) is real and separate from the
   benchmark ceiling, so some of it is recoverable regardless of 1 and 2.

## 6. Retractions from this pass

- **"The fit-versus-controllability trade-off is a real constraint"** —
  overstated. The raw cosine fell resolvably in 4/8 seeds but the above-chance
  z-score in only **1/8**; mean Δz = −0.24 ± 0.20, not resolvable. Most of the
  trade-off was the dimensionality artefact.
- **"It is latent drift, remedied by constraining drift or selecting
  checkpoints on C6"** — the premise was the trade-off, so the remedy was aimed
  at the wrong target.
- **"`L_metric` weighting is the cause"** — a 1-seed, 1-replicate smoke test
  showed eff_dim 1.01→1.82 and specificity −0.07→+1.11 when the weight went
  0→1. With 4 seeds × 2 replicates the effect vanished (rho = +0.24). The
  replicated design caught this before it was reported as a result.
- **"The benchmark lacks effect dimensionality"** — rejected by measurement:
  the effect labels carry 38–64 effective dimensions.

## 7. What is still open

- **The encoder gap**: specificity +1.05 learned vs +2.22 in metadata space.
  Nothing tried so far (metric weight, latent dimension) moves it.
- **Criterion 6 still fails on the learned representation, 0/6 seeds.** It
  remains the blocker for E1.
- **Criterion 5** passes 3/5 seeds with the corrected measurement and is
  positive on all 5 (see `E0_criteria_resolution.md`). Not re-measured after
  the D_0 change, so those numbers predate it.
- **E1 (DMC closed loop) still unrun** — `experiments/run_e1_dmc.sh`, ~4.5h.
  Its premise is in better standing than before, but criterion 6 is what makes
  a latent direction executable, and it does not pass yet.
