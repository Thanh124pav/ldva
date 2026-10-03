# LDVA — Latent Data Valuation for Budgeted Robot Acquisition

> A robot sample does not have a fixed scalar value. Its training effect is
> contextual: it depends on the current policy and on the other samples used
> with it. LDVA learns a latent representation of sample-level optimization
> effect, learns a set-level utility model on top of those latents, and uses the
> resulting geometry to decide how to expand data support under a fixed
> acquisition budget.

`SETUP.md` and `PLAN.md` are the specification; this code implements them.
Section references in docstrings point back to those two files.

## Status

Implemented, and verified end to end on the Stage 0 synthetic benchmark:

| SETUP.md §29 coding order | state |
|---|---|
| 1. synthetic acquisition test | done — `experiments/synthetic/run_stage0.py` |
| 2. context supervision format | done — `ldva/data/context_dataset.py` |
| 3. gradient-alignment effect estimator | done — plus influence, TRAK, one-step, leave-one-out |
| 4. sample encoder | done — MLP / GRU / transformer backbones |
| 5. DeepSets context encoder | done — with exact O(N) leave-one-out |
| 6. contextual effect readout | done — plus the scalar-score ablation |
| 7. batch utility predictor | done — DeepSets, additive, pairwise |
| 8. latent diagnostics | done — `ldva/analysis/latent_geometry.py` |
| 9. KMeans clustering | done — plus GMM and HDBSCAN |
| 10. PCA direction generation | done — with all four filters of §16 |
| 11. exact allocation solver | done |
| 12. beam-search solver | done — plus greedy and a width sweep |
| 13. PushT integration | **not done** — config, metadata spec and adapter stub |
| — DMC (PLAN.md §15) | done — `ldva/envs/dmc`, real MuJoCo, scripted experts |
| 14. metadata mapper | done — local Jacobian + constrained inversion |
| 15. closed-loop acquisition | done on synthetic — `experiments/run_acquisition_loop.py` |
| 16. MetaWorld scale-up | adapter **done** (`ldva/envs/metaworld`); the 5-task sweep has not been run |

Also implemented: the cost-aware budget of §31 (`sum_a c_a n_a <= C`) across all
three solvers, the nine-method baseline suite of §20, and ten of the eleven
ablations of PLAN.md §18 (the eleventh, exact-vs-beam, is reported inside the
solver ablation). Stages 4–6 of the benchmark ladder (static datasets, paid
data, real arm) are not started.

Everything above `ldva/envs/base.py` is environment-agnostic, so moving up the
ladder means implementing one `EnvAdapter` — `metadata_spec`, `collect`,
`evaluation_set` — and nothing else. Three adapters are real (synthetic, DMC,
MetaWorld); the PushT and ManiSkill stubs raise `NotImplementedError` listing
exactly what to implement, so the integration point is unambiguous rather than
silently wrong.

Nothing had to be installed for any of this: the shared `deeplearning` env
already had `metaworld` 3.1.1, `dm_control` 1.0.27 and `mujoco` 3.3.0, and
there were no version conflicts.

## Environments

| adapter | metadata (what acquisition controls) | expert | speed |
|---|---|---|---|
| `synthetic` | object x/y, difficulty | analytic | instant |
| `dmc` (`reacher-easy`) | target **radius, angle** | 2-link analytic IK + joint PD (186/200) | ~0.1s / 6 episodes |
| `dmc` (`point_mass-easy`) | start x/y (goal fixed) | PD to origin (~94/200) | ~0.1s / 6 episodes |
| `metaworld` (5 tasks) | task id, object xyz, goal xyz | 5 `Sawyer*V3Policy` scripts (100% success) | ~0.06s / episode |

Run the full pipeline on any of them:

```bash
python experiments/run_pipeline.py --env dmc --task reacher-easy
python experiments/run_pipeline.py --env metaworld
python experiments/run_pipeline.py --env synthetic --quick
```

Two MetaWorld deviations from SETUP.md §6, both forced by the installed
package and both recorded in the adapter docstring: metaworld 3.1.1 ships only
`*-v3` environments (all five named tasks exist), and the controllable metadata
width differs per task (6 values for reach/push/pick-place, 3 for
drawer-open/button-press), so the spec is the union of the per-task reset boxes
and `collect` records what the simulator actually realized.

## Environment

Uses the shared conda env `deeplearning` (python 3.12, torch 2.11+cu128, numpy,
scikit-learn, scipy, matplotlib, wandb). No new environment is needed:

```bash
~/miniconda3/envs/deeplearning/bin/python -m pytest tests/ -q
```

Per SETUP.md §3, simulator dependencies stay in their own envs — PushT /
MetaWorld / ManiSkill should be installed into separate conda envs
(`robo-mujoco`, `robo-maniskill`), never alongside the LDVA core.

## Run it

```bash
# Stage 0 gate: the six success criteria of SETUP.md 30 (~15 min)
~/miniconda3/envs/deeplearning/bin/python experiments/synthetic/run_stage0.py

# fast smoke version (~3 min)
~/miniconda3/envs/deeplearning/bin/python experiments/synthetic/run_stage0.py --quick

# closed-loop acquisition, LDVA against 6 baselines at equal budget
~/miniconda3/envs/deeplearning/bin/python experiments/run_acquisition_loop.py --quick

# ablations of PLAN.md 18
~/miniconda3/envs/deeplearning/bin/python experiments/synthetic/run_ablations.py --quick
```

`run_stage0.py` prints a pass/fail table and writes `stage0_report.json` plus
figures. It is a gate, not a demo: SETUP.md §4 says do not move to robotics
until it passes.

`--quick` is a smoke test for the plumbing, not a measurement. It uses ~⅓ the
data and ⅓ the epochs, and its effect sizes are correspondingly weaker — the
set-vs-additive ratio comes out between 0.95 and 1.07 on `--quick` against 2.49
on the full config, so **criterion 2 typically fails on `--quick`** for lack of
data rather than for any structural reason. Quote numbers from a full run, with
`--seeds` ≥ 3.

## Stage 0 result

The gate passes 6/6 on the default config (seed 0, single seed):

| SETUP.md §30 criterion | measured | rule |
|---|---|---|
| 1. contextual beats a scalar per sample | 1.92× lower MSE (1.42× vs the scalar-readout ablation) | ratio > 1 |
| 2. set utility beats additive utility | 2.49× lower within-checkpoint gain MSE (R² 0.65 vs 0.12) | ratio > 1 |
| 3. latent neighbours share effect profiles | 0.59 neighbour/random effect distance; 73% of probes | < 1 |
| 4. beam ≈ exact on small problems | 0.0000 relative gap (sub-problem A=4, B=8) | ≤ 2% |
| 5. predicted high-value batches realize gain | Spearman +0.51 over 14 compositions | > 0.3 |
| 6. metadata interventions move latents | mean cosine +0.41, 64% of directions correct | > 0.3 |

Two things worth reading together with that table:

- On the **full** 14-direction candidate set beam search beats greedy by 1.89%
  (1.3538 vs 1.3287) at 375 evaluations instead of enumerating. On the small
  sub-problem greedy ties exact, which is why criterion 4 validates beam against
  exact and the greedy comparison is reported separately — complementarity needs
  enough directions before it shows up.
- Criterion 5 passes on ranking (+0.51) but `picked_the_best` is false: the
  top-predicted composition was not the top-realized one, with 3.3% normalized
  regret. Ranking is what the planner needs, and that is the claim PLAN.md §17.2
  makes, but the distinction should not be blurred.

## Layout

```
ldva/
  data/         Sample / SampleStore, ContextRecord, metadata spec, effect profiles
  policy/       BC policy, checkpoint store, training that *emits* checkpoints
  supervision/  EffectEstimator interface + 6 targets, context-record generator
  models/       sample encoder, context encoder, effect readout, batch utility
  training/     losses, metrics, trainer
  acquisition/  clustering, directions, latent sampler, objective, 3 solvers,
                metadata mapper, 9 baselines
  analysis/     latent geometry, acquisition calibration, plots
  envs/synthetic/   Stage 0 world and its ground-truth oracle
configs/        env / policy / datamodel / acquisition
experiments/    stage 0, ablations, closed-loop acquisition
tests/          106 tests, CPU-only, ~16s
```

One deviation from the structure in SETUP.md §28: the code lives under an
importable `ldva/` package rather than at the top level, so `data/` cannot
collide with runtime data directories. `configs/`, `experiments/` and `tests/`
stay where §28 puts them.

## How the pieces fit

```
metadata m ──f──> true latent u ──> chunk x
                                     │
                          E_phi(x, theta) ──> effect latent z
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
  ContextEncoder(Z_B)        EffectReadout(z_i, h_B\i)   BatchUtility(Z_B, D, theta)
        │                            │                            │
        └──> cluster ──> local PCA ──> outward directions ──> allocation planner
                                             │                     │
                                   metadata mapper (J_k)      V_hat(n)
                                             │
                                   collect ──> D_{t+1}
```

The hinge is that `BatchUtility` is defined over **latents**, not raw data, so
`model.utility_from_latents(Z)` can score acquisitions that do not exist yet.

## Findings from Stage 0

These came out of building it and are worth keeping in mind; each is recorded in
the relevant docstring.

- **Context-free proxies cannot stand in for contextual labels.** Gradient
  alignment tracks the leave-one-out *batch gain* almost perfectly
  (Spearman ≈ 1.0 for `lr ≤ 0.2`), but correlates only ρ ≈ 0.22 with
  leave-one-out *per-sample marginal* effects — it does not know what else is in
  the batch. Leave-one-out is the right primary target; the proxies are for
  scaling up and for ablations.
- **A sample's solo effect is not its marginal effect** (r ≈ 0.45). That gap is
  the contextual signal LDVA is built on.
- **98% of raw batch-gain variance is between-checkpoint.** Evaluating batch
  utility over all contexts therefore measures checkpoint identification, not
  composition. Set-level claims must be made on the within-checkpoint residual
  (`gain_within_*`).
- **The additive-utility baseline must be fitted out of sample.** A per-sample
  scalar fitted on the evaluation split is ~1.8× optimistic; an additive fit over
  batch compositions has one free parameter per sample and reports *perfect*
  additivity whenever contexts ≤ samples. Both are cross-validated now.
- **Latent-space outwardness does not imply metadata actionability.** Encoded
  latents carry per-chunk sampling noise (73% of latent variance for an
  untrained encoder), so leading PCA directions are partly noise directions that
  no metadata change can produce. SETUP.md §16's fourth filter is what fixes
  this: filtering on the *constrained* achievable cosine raised realized
  direction control from −0.18 to +0.37.
- **Step size trades off against directional fidelity.** Realized direction
  cosine is 0.33 at `delta_scale=0.4` but 0.18 at 1.5, because the local linear
  metadata map degrades over a long step.
- **The label update size controls whether batch interaction exists at all.**
  The additive part of a batch gain is `O(lr)` and the interaction part
  `O(lr²)`, so a small update produces nearly additive gains and Stage 0 cannot
  test the set-level claim. Measured on this world: at `lr=0.1` only 7% of gain
  variance is composition-dependent and an additive fit over sample identity
  reaches R² +0.45, while at `lr=0.3, n_steps=4` composition explains 84% and
  the additive fit fails (R² −0.36). Stage 0 therefore labels at `lr=0.3`. The
  price is that the cheap first-order proxies stop tracking leave-one-out there,
  and that effect prediction gets harder (Spearman 0.59 vs 0.92) — the
  calibration report makes both visible.
- **Measuring direction control needs variance reduction.** A single chunk's
  latent noise (0.50) exceeds a planned displacement (0.22), so an unpaired
  12-vs-12 comparison is pure noise. Common random numbers make 12 paired
  samples as accurate as 256 unpaired ones.
- **The declared metadata box must *be* the environment's feasible set.** The
  DMC reacher arm reaches an annulus, not a box, so a Cartesian `(target_x,
  target_y)` spec forced the adapter to clip the radius — a non-local,
  angle-dependent kink that no local linear Jacobian can represent. It looked
  like a model failure: plans whose latent displacement stayed small tracked
  the intended direction well (cosine +0.83, +0.79, +0.86) while plans that hit
  the clip were thrown elsewhere (|displacement| ≈ 2.0, cosine −0.98 to −0.36),
  dragging the mean to +0.01. Reparameterizing the target in polar coordinates
  makes the box exactly feasible, and requests then come back bit-identical.
- **Real observations need normalizing; the synthetic world hid this.** DMC
  `reacher` states have per-dimension standard deviations spanning 0.10 to 3.66
  and means up to 1.8. An MLP[128,128] on them diverged to NaN under the SGD
  settings that work on the synthetic world, and the checkpoint-spreading init
  perturbation made it worse by adding an absolute-size kick to a layer whose
  init std is ~0.09. The policy now standardizes observations in buffers (so
  the statistics never enter `flat_params()` and the effect estimators are
  unaffected) and perturbs each tensor relative to its own scale.

## Gates

The code refuses to paper over the failure modes in PLAN.md §19. Each has a
diagnostic that returns a verdict, not just a number:

| failure mode | diagnostic |
|---|---|
| F1 no stable effect geometry | `neighbor_effect_consistency`, `latent_vs_effect_distance_correlation` |
| F2 batch utility nearly additive | `additivity_report` (cross-validated) |
| F3 labels are estimator noise | `label_noise_report`, cheap-vs-expensive calibration |
| F4 directions not metadata-actionable | `filter_actionable_directions` (reports `F4_warning`) |
| F5 unreliable out-of-support predictions | dual trust region + MC standard errors |

`latent_geometry_report(...)["gate_passed"]` is the Phase 3 go/no-go. An
undetermined diagnostic (NaN) counts as *not* passed, never as a pass.

## Reproducibility

`SeedBundle` keeps the five seed roles of SETUP.md §24 independent — `env`,
`policy`, `context`, `latent`, `acquisition` — so changing the search seed does
not perturb data generation. Use ≥3 seeds in development and ≥5 for final
numbers (`--seeds`).

The closed-loop script will not let a noisy ranking be read as a result. Each
round's performance is averaged over independently initialized policies, and the
run reports whether the spread between methods exceeds twice the standard error
of a method's mean. At `--quick --seeds 1` it does not (spread 0.014 against a
policy-noise floor of 0.10), and the script says `NOT RESOLVABLE` instead of
printing a winner.
