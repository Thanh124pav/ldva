# Criteria 5 and 6 resolved

Follow-up to [`E0_diagnosis.md`](E0_diagnosis.md), doing the two things it left
open: fix criterion 5's measurement inside the gate and re-run it, and test the
fit-versus-controllability tension causally.

Raw output: [`results/E0_fixed/`](results/E0_fixed/) (5 seeds),
[`results/E0_controllability/`](results/E0_controllability/) (15 seeds).

## (a) Criterion 5: measurement fixed, now positive on every seed

```bash
for s in 0 1 2 3 4; do python experiments/synthetic/run_stage0.py --seed $s; done
```

| seed | criterion 5 | criterion 6 | passed |
|---|---|---|---|
| 0 | +0.177 | +0.314 | 5/6 |
| 1 | +0.317 | +0.459 | **6/6** |
| 2 | +0.450 | +0.012 | 5/6 |
| 3 | +0.499 | +0.419 | **6/6** |
| 4 | +0.299 | +0.199 | 4/6 |
| **mean** | **+0.348** | +0.280 | |

Criterion 5 is **positive on 5 of 5 seeds** (+0.18…+0.50, mean +0.348, passing
the 0.3 threshold on 3) where the original measurement gave +0.034, −0.076,
−0.569 and passed none.

The change: the candidate set is de-duplicated and no longer consists only of
what the planners picked. It is now planner picks + the one-hot allocations
(whole budget on one direction) + mixed random ones, declared in advance.

**Criterion 6 is now the binding constraint**, failing on seeds 2 and 4.

### What the candidate set must and must not be

Which allocation type carries the realized signal is **seed-dependent**, and
getting this wrong cost two retractions:

| seed | planner only | planner+one-hot | all three | one-hot only | mixed only |
|---|---|---|---|---|---|
| 0 | +0.267 | **+0.522** | +0.177 | +0.405 | +0.244 |
| 1 | +0.091 | +0.271 | **+0.317** | +0.143 | **+0.402** |

On seed 0 the mixed allocations dilute badly — their realized gains span only
0.098, inside the noise, against 0.778 for the planner picks — and dropping
them lifts Spearman from +0.177 to +0.522. On seed 1 the *same* mixed
allocations are the most informative group of all (+0.402) and are what makes
that seed pass.

So the set cannot be chosen per seed by whichever scores highest: that is
selecting on the outcome, and it makes the gate measure the measurer rather
than the model. The rule is fixed in advance and includes all three types;
`calibration.by_source` reports the breakdown as a diagnostic only.

## (b) The fit-versus-controllability trade-off is real

> **Largely retracted by [`E0_root_cause.md`](E0_root_cause.md).** Scored
> against the right null - the other candidate directions rather than the raw
> cosine - the raw cosine fell resolvably in 4/8 seeds but the above-chance
> z-score in only 1/8 (mean Δz = −0.24 ± 0.20, not resolvable). Most of what
> is below was the latent space growing from ~1.1 to ~2.1 effective
> dimensions, which lowers a cosine for free. The measurements stand; the
> causal reading does not.

```bash
python experiments/synthetic/diagnose_controllability.py \
    --seeds 0..15 --epochs 5,60 --smooth-weights 0.1 --replicates 2
```

`E0_diagnosis.md` reported rho(C6, effect_spearman) = −0.833 across 8 seeds, but
that was observational: `effect_spearman` is a property of a seed's trained
model, so anything else varying by seed was a candidate confound. This varies
the fit *within* a fixed seed — same world, same dataset, same supervision
records, same clustering and direction seeds — and gives each seed one signed
contrast dC6 = C6(60 epochs) − C6(5 epochs) with a standard error from
replicates.

```
7 trade-off   |   1 positive   |   7 unresolved        (15 seeds)
raw sign: 12 negative / 3 positive
mean dC6 = -0.237 +/- 0.061   ->  |t| = 3.9
```

The trade-off is real, and this is stronger evidence than the original
correlation because it is within-seed and therefore unconfounded. But it is
**not universal** — seed 1 moved the other way (dC6 = +0.207).

### Conditions that separate the groups

| | trade-off (7) | positive (1) | unresolved (7) |
|---|---|---|---|
| **fit gain** (how much fit improved) | **+0.296** | **+0.153** | +0.204 |
| **C6 before training** | **+0.714** | **+0.426** | +0.495 |
| C6 after training | +0.308 | +0.633 | +0.364 |
| fit after training | +0.735 | +0.582 | +0.562 |
| gain within-ckpt R² | +0.630 | +0.025 | +0.410 |
| overshoot | 1.94 | 1.65 | 2.04 |

Two conditions separate them:

1. **A large fit gain predicts the trade-off.** Trade-off seeds improved fit by
   +0.296, the positive seed by +0.153. At the extremes: seed 11 (+0.508) →
   dC6 −0.519; seed 0 (+0.409) → −0.465; while seeds 6 and 9 (+0.065, +0.075)
   barely moved (−0.202, −0.017).
2. **High starting controllability predicts a large loss.** Trade-off seeds
   started at C6 +0.714 against +0.426 for the positive one; seed 5 fell from
   **+0.910** to +0.353.

### The mechanism is not what the name suggests

> Also superseded: the "latent drift" reading below was built on the trade-off,
> and the root cause turned out to be D_0's rank-deficient geometry. See
> `E0_root_cause.md` section 3.

C6 *after* training is similar across all three groups (+0.308 / +0.633 /
+0.364). What differs is where each seed **started**. So training does not
"destroy" controllability in proportion to fit quality — it pulls
controllability toward ~0.3–0.4 regardless of its starting value. Seeds that
happened to start high lose a lot; the one that started low gained.

That reading matters for what to do about it. This is not a trade-off that has
to be accepted: it is a symptom of the latent representation drifting freely
between training runs. The remedies it points to are constraining that drift,
or selecting checkpoints on C6 as well as on effect fit — not training less,
which would cost criterion 5.

**Overshoot does not separate the groups** (1.94 / 1.65 / 2.04), the third
measurement to reject that hypothesis.

## Retractions

Recorded because several intermediate conclusions in this investigation were
wrong, and the write-ups should not read as if they were not:

- **"Averaging realized gain over draws fixes criterion 5"** — no. Over 24
  cells averaging moved Spearman by −0.005.
- **"Overshoot explains criterion 6"** — no. rho = −0.23 over 24 cells, and the
  group means above are flat.
- **"One-hot allocations are the right candidate set"** — no, seed-dependent;
  see the table in (a).
- **"P0.4 caused the criterion-5 regression"** — inconclusive; the two A/B
  seeds disagreed in direction.

## Two bugs found along the way

- **Model initialization was not seeded.** `DataModelTrainer` calls `set_seed`
  in its own `__init__`, which is *after* `LDVADataModel(...)` has drawn its
  weights, so the initialization depended on whatever the global torch RNG had
  reached. Identical configurations gave C6 +0.471 and +0.622 on one seed and
  +0.530 and +0.060 on another — a noise floor as large as the effects under
  study. This affected **every ablation** built on `run_ablations.Fixture`, not
  just this experiment. Fixed by seeding before construction.
- With that fixed, training became fully deterministic, so replicates came out
  byte-identical (std = 0.000). Replicates now vary the training seed through
  `Fixture.train(seed_offset=...)`, which is the only place that re-seeds.

## What is still open

- Criterion 6 fails on 2 of 5 seeds and is the blocker for E1.
- The drift remedy is untested: constrain latent drift between rounds, or select
  checkpoints on C6 as well as effect fit.
- E1 (DMC closed loop) still unrun. `experiments/run_e1_dmc.sh`, ~4.5h.
