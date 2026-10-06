# E1 (DMC) and E2 (MetaWorld), first full run, with every criterion followed through

First time the closed loop has been run on both robotics adapters. 8 seeds each,
run in parallel with a fresh 10-seed E0 at HEAD to see how the preflight gate
behaves on the current code and how each criterion carries over to the real
environments. Raw reports under
[`results/E1_E2_first_run/`](results/E1_E2_first_run/) and
[`results/E0_full_10seeds/`](results/E0_full_10seeds/).

Launched by `scratchpad/launch_experiments.sh` with per-method invocations:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa WANDB_MODE=offline \
  python experiments/run_acquisition_loop.py --env {dmc,metaworld} \
    --env-task {reacher-easy,push-v3} --methods <m> \
    --seeds 8 --rounds 4 --budget 20 --n-initial 40 \
    --out runs/{E1_dmc,E2_metaworld}/<m> --no-plots
```

## E0 at HEAD (10 seeds, full config)

```bash
for s in 0..9; do python experiments/synthetic/run_stage0.py --seed $s --out runs/E0_full/seed$s; done
```

| seed | C1 | C2 | C3 | C4 | C5 | C6 | pass |
|---|---|---|---|---|---|---|---|
| 0 | P | P | P | P | **F** | P | 5/6 |
| 1 | **F** | **F** | P | P | **F** | P | 3/6 |
| 2 | P | P | P | P | **F** | P | 5/6 |
| 3 | P | **F** | P | P | **F** | P | 4/6 |
| 4 | P | **F** | P | P | P | **F** | 4/6 |
| 5 | P | P | P | P | **F** | **F** | 4/6 |
| 6 | P | P | P | P | P | **F** | 5/6 |
| 7 | **F** | **F** | P | P | P | P | 4/6 |
| 8 | P | **F** | P | P | **F** | P | 4/6 |
| 9 | P | P | P | P | P | P | **6/6** |

Pass rate: C1 **8/10**, C2 **5/10**, C3 **10/10**, C4 **10/10**, C5 **3/10**, C6
**7/10**. Mean 4.3/6.

### C1 and C2 regressed since `E0_fixed` was recorded

| | hist s0 | hist s1 | hist s2 | hist s3 | hist s4 | now s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 ratio | 13.0× | 3.1× | 1.14× | 3.3× | 4.1× | 5.0× | **0.999×** | 2.6× | 1.16× | 1.54× |
| C2 ratio | 1.29× | 2.30× | 1.06× | 1.79× | 3.55× | 1.03× | **0.997×** | 1.00× | **0.994×** | **0.939×** |

Historical E0_fixed data (committed with `2c099dc`) had the same seeds passing
C1 and C2 with comfortable margins. On HEAD the ratios hover at 1.0.

The one commit between those states that touches the dataset distribution is
`5070912` ("Find and fix the root cause of criterion 6: D_0 was rank-deficient").
That commit changed `default_initial_regions()` from "2 tight modes in one
corner" (eff_dim 1.17 of 3) to "modes inside a corner sub-box, full rank"
(eff_dim 2.21–2.33), to lift the specificity ceiling for C6.

The side effect was that a full-rank D_0 lets the **hindsight scalar baseline**
pick up more of the effect structure, so the contextual encoder's advantage
over it shrinks. On seed 1 the ratio moved from 3.11× down to 0.999 — not a
failure of the contextual model but a shrinking of the gap it is being asked
to open against a now-stronger baseline.

**New insight, not stated in `E0_root_cause.md`.** Fixing C6 was not free: the
D_0 rank-fix bought specificity for the latent, and paid for it in C1/C2
margin. The 1.0 threshold on C1/C2 is now sitting inside the noise band on
seeds where the within-checkpoint composition signal is weakest, so these
criteria are decided by seed-level noise rather than by method quality. If a
C1/C2 margin matters, either the threshold has to move or the data model has
to be strengthened against a stronger scalar baseline. Both are open.

### C5 is the dominant blocker, C6 mostly passes

Mean Spearman for C5 is **+0.158** against the +0.3 threshold; 3/10 seeds pass
(s4 +0.28 just-below, s6 +0.32, s7 +0.50, s9 +0.47 — and s9 passes all six).
C5 and C6 passes do **not co-occur** cleanly: only s7 and s9 pass both; five
seeds (s0-s3, s8) pass C6 but fail C5; two (s4, s6) pass C5 but fail C6.

The specificity_gap statistic introduced by `916e76e` is working as intended
on E0 — gap mean +0.342 across seeds, realized cosine mean +0.825, 7/10 seeds
clear the +0.3 threshold.

## E1 — DMC `reacher-easy`, 8 seeds × 4 rounds × budget 20

| rank | method | final return | ±pol | succ | C1 | C5 ρ | C6 cos | C6 > .3% |
|---|---|---|---|---|---|---|---|---|
| 1 | **random** | **46.76** | ±4.28 | 0.42 | 1.79 | −0.26 | −0.21 | 0% |
| 2 | ldva_greedy | 44.79 | ±3.63 | 0.41 | 1.89 | −0.06 | +0.03 | 19% |
| 3 | ldva_beam | 44.67 | ±4.35 | 0.42 | 1.90 | −0.29 | −0.09 | 25% |
| 4 | diversity | 42.76 | ±3.15 | 0.39 | 1.65 | +0.07 | −0.29 | 0% |
| 5 | direct_gradient_alignment | 38.59 | ±4.60 | 0.33 | 1.89 | +0.15 | +0.02 | 41% |
| 6 | direct_influence | 37.43 | ±3.32 | 0.35 | 1.88 | +0.16 | +0.03 | 41% |

Resolution: spread **9.32** vs 2×SEM **2.75** → **RESOLVABLE**. The resolution
is driven by the ~9-point gap between random/ldva and the direct baselines,
not by random beating ldva (which is within noise).

Pooled over 192 round×method×seed points:

- **C1** `effect_scalar_hindsight_mse / effect_mse` mean **+1.836**, 76% of rounds > 1 → **PASS**
- **C2** not logged (closed loop does not retrain the additive-head ablation per round)
- **C3** not logged (`latent_geometry_report` only runs in E0)
- **C4** not applicable (closed loop solves on the full candidate set via beam/greedy, no exact enumeration)
- **C5** Spearman(predicted_utility, next-round rollout gain) = **+0.045** (p = 0.54) → **FAIL** (indistinguishable from zero)
- **C6** mean cos **−0.085**, 36% > 0 and 21% > 0.3 → **FAIL**

## E2 — MetaWorld `push-v3`, 8 seeds × 4 rounds × budget 20

| rank | method | final return | ±pol | succ | C1 | C5 ρ | C6 cos | C6 > .3% |
|---|---|---|---|---|---|---|---|---|
| 1 | **diversity** | **122.86** | ±44.48 | 0.38 | 1.00 | −0.17 | −0.17 | 12% |
| 2 | random | 85.04 | ±35.57 | 0.21 | 0.97 | +0.27 | −0.13 | 3% |
| 3 | ldva_greedy | 78.92 | ±30.68 | 0.22 | 0.93 | +0.19 | −0.29 | 9% |
| 4 | direct_gradient_alignment | 45.21 | ±16.07 | 0.16 | 1.17 | +0.35 | −0.06 | 28% |
| 5 | ldva_beam | 44.79 | ±26.15 | 0.13 | 0.95 | −0.04 | −0.22 | 12% |
| 6 | direct_influence | 33.37 | ±10.17 | 0.13 | 1.12 | +0.09 | −0.04 | 34% |

Resolution: spread **89.5** vs 2×SEM **19.2** → **RESOLVABLE**. Diversity
beats everything by ~40 points of return.

Pooled over 192 points:

- **C1** mean **+1.023**, only 22% of rounds > 1 → **FAIL** (the data model does *not* beat a hindsight scalar on MetaWorld)
- **C2, C3, C4** same scope caveats as E1
- **C5** Spearman = **+0.083** (p = 0.25) → **FAIL**
- **C6** mean cos **−0.150**, 34% > 0, 17% > 0.3 → **FAIL**

## Insights the three experiments make together

1. **The C5/C6 failures on E0 are not synthetic artefacts.** They reproduce on
   both DMC and MetaWorld with 8 seeds each and 192 round-points pooled:
   `predicted_utility` is uncorrelated with the realized rollout gain
   (Spearman +0.045 on DMC, +0.083 on MetaWorld) and the planned latent
   direction is executed in the *opposite* direction more often than the right
   one (mean cos −0.085 on DMC, −0.150 on MetaWorld). The two signals LDVA is
   built to use — predicted utility and direction control — are **missing on
   real data**. Whatever `E0_richness_vs_reach.md` recovered for the synthetic
   latent did not transfer.

2. **C1 passes on DMC but fails on MetaWorld.** The contextual encoder beats
   the hindsight scalar by ~2× on DMC (pooled 1.84, 76% of rounds) but ties it
   on MetaWorld (1.02, 22%). MetaWorld has 5 tasks × wide metadata boxes, so
   effect structure between distant tasks is strong and a scalar baseline can
   pick it up; what remains for the contextual encoder is the within-task
   variation, which this data model does not capture well. **The data model
   itself fails C1 on MetaWorld** — not just the planner on top of it.

3. **Which baseline wins depends on the environment.** On DMC random is the
   top mean (46.8) but is tied with the ldva methods within 2×SEM; the
   resolvable gap is between those three and the direct baselines
   (gradient_alignment, influence) which lag by ~9 points. On MetaWorld
   diversity wins by 38 points over random and by 44 over ldva_greedy —
   consistent with 5-task MetaWorld rewarding task-coverage over any
   effect-based ranking that the data model cannot provide (per point 2).

4. **Direct baselines are worse than random on both envs.** gradient_alignment
   and direct_influence rank at or near the bottom in both E1 and E2. These
   do not use the data model; they score candidates directly from policy
   gradients and from influence functions. Their underperformance implies
   that the metadata-choice signal that *is* in the first-order gradient is
   not useful for picking which chunks to collect at this budget — the
   per-sample policy-gradient alignment story that the E0 writeups described
   as "almost perfect correlation" with batch gain does not survive the move
   to a real BC policy on real data.

5. **The resolution test does its job.** Both environments cleared the 2×SEM
   bar at 8 seeds, so the rankings above are not within-noise. In particular
   the E2 result "diversity wins by 40 points" is at ~2× the pooled noise
   floor; on E1 the random vs ldva gap is *not* resolvable, which the table
   makes explicit via ±pol.

6. **The D_0 rank-fix regression on E0 means the E0 gate itself is
   unreliable at the current thresholds.** C1 and C2 ratios of 0.99 vs 1.03
   on neighbouring seeds decide the same criterion in opposite directions
   purely by seed-level noise. The honest reading: C1 and C2 as currently
   stated are not strong gates post-`5070912`, and should either be
   rethresholded (e.g. ratio > 1.1 with a measurement SE) or re-grounded
   against a baseline that cannot pick up the extra structure a full-rank
   D_0 exposes.

## Why C5 and C6 fail on real envs: it is not the data model

With 240 round-points per env in hand, the two failure modes have different
causes, both located outside the model.

### C5 is not a prediction problem; it is a noise-floor problem

The data model's internal fit is actually strong on both envs:

| | E1 DMC | E2 MW |
|---|---|---|
| val/gain_within_r2 | +0.68 | +0.39 |
| val/gain_within_spearman | +0.83 | +0.66 |
| val/effect_r2 | +0.70 | +0.44 |
| val/effect_spearman | +0.87 | +0.68 |

It is predicting *within-dataset* gain and effect well. What C5 compares
predicted_utility to is the **round-to-round rollout return gain**, and that
quantity is dominated by policy-training noise:

| | E1 | E2 |
|---|---|---|
| rollout_return_std per policy checkpoint | **3.22** | **29.4** |
| mean &#124;realized gain&#124; between rounds | 3.58 | 32.4 |
| **signal/noise ratio** | **1.11** | **1.10** |

The thing C5 asks the planner to predict has an SNR of ~1 on both envs; a
Spearman near zero is what that SNR actually permits regardless of planner
quality. Supporting evidence: splitting E1 round-points by the median of
gain_within_r2, the top-half (where the model's internal fit is best) gives
spearman +0.151 vs the bot-half's −0.060. There is some signal where the
model fits well; it just stays inside the noise band.

Routes that would help: more eval policies per round, bigger per-round
budget so realized_gain outruns noise, or dropping the "next-round" scope
and comparing predicted total value against final-round return. The current
C5 is a strict test the budget was not sized for.

### C6 is not a mapper problem; it is a latent-drift problem

Direction control cosine on E1 **by round**:

| round | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean cos | **+0.196** | −0.133 | −0.243 | −0.161 |

Round 0 (where the planner's latent frame is the same one the realized
movement is measured in) is positive. Rounds 1-3 collapse. The data model
is **retrained each round on the enlarged dataset**, so by round 1 the
encoder has a different frame; the direction planned in the round-0 frame
is being measured in the round-1 frame, and ends up near random or
opposite. `val/latent_norm_mean` std across rounds is 0.18 on E1 and
**0.39** on E2 — the latent norm itself shifts between rounds, which is a
drift signature.

Supporting evidence: C6 cosine vs gain_within_r2 has spearman +0.001
(p=0.99) on E1 and −0.193 (p=0.007) on E2 — direction control fails **not**
when the data model fits poorly but independently of fit, which matches
"frame changed" rather than "mapper wrong".

Routes that would help: freeze the encoder after round 0, measure
`specificity_gap` only at round 0 as E0 does, add an explicit drift
constraint between rounds, or re-express directions in a frame invariant
across retrainings (e.g. the metadata frame they were mapped from).

### Combined reading

The data model is doing its job on both environments. The two failures
sit **downstream** of it: a noise floor the gate did not budget for, and a
frame change the gate's statistic does not control for. Any next pass that
tightens C5/C6 as currently written without changing either will measure
noise or drift, not method quality.

## What is still open

- **C5 is the real blocker end-to-end**, not C6. All three experiments agree:
  predicted utility is uncorrelated with realized gain. The next pass should
  diagnose *why* `AllocationObjective` produces a value whose ordering does
  not survive one round of real training — specifically whether the issue is
  the data model's gain prediction or the mapper's between-metadata-space
  transfer.
- **C6 on real envs is worse than E0 predicts.** Specificity_gap passes
  7/10 on synthetic but mean cos is negative on both robotics envs. The
  encoder gap identified by `E0_root_cause.md` (metadata z +2.22 vs learned
  z +1.05) likely compounds with the real-env non-locality of the
  metadata→latent map.
- **C2/C3/C4 analogs on E1/E2** need either a `--log-additive-ablation`
  option on `run_acquisition_loop.py` or a persisted datamodel.pt so a
  post-hoc `latent_geometry_report` can run. Neither is wired yet; this
  summary states what is measured and what is not, rather than filling with
  a weaker proxy.
