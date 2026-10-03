# Why E0 criteria 5 and 6 failed — 24-cell diagnosis

E0 failed criterion 5 on all three seeds (+0.034, −0.076, −0.569) and criterion
6 on seed 2 (+0.012). This is the follow-up that locates both causes.

```bash
python experiments/synthetic/diagnose_criteria.py \
    --seeds 0,1,2,3,4,5,6,7 --delta-scales 0.1,0.2,0.4 \
    --repeats 6 --n-random 14 --wandb --experiment E0diag
python experiments/synthetic/analyse_diagnosis.py \
    --report runs/E0_diag/diagnose_criteria.json
```

8 seeds × 3 step lengths = 24 cells. Raw output in
[`results/E0_diagnosis/`](results/E0_diagnosis/).

## Criterion 5: the measurement, not the method

### The decisive comparison

Same seed, same step length, only the measurement changed:

| seed | C5 at `delta=0.40` (corrected) | C5 at `delta=0.40` (E0) |
|---|---|---|
| 0 | +0.206 | +0.034 |
| 1 | −0.024 | −0.076 |
| **2** | **+0.388** | **−0.569** |

Seed 2 — the seed that produced E0's worst number — flips from **−0.569 to
+0.388**, above the 0.3 threshold, at the *same* `delta_scale`. So the negative
correlation was not a reversed method and not an effect of step length.

Across all 24 cells: mean **+0.309**, range [−0.226, +0.874], **11 positive /
12 null / 1 negative**. With 8 seeds the correlation is essentially never
negative, so criterion 5 is neither meaningless nor inverted.

### What was wrong with the original measurement

`predicted` is a Monte-Carlo **expectation** over the latents a direction could
yield. `realized` was a **single draw**: `plan_direction` re-samples anchors on
every call, so an identical allocation vector produces different metadata, and
the 4-step update at `lr=0.3` can diverge (E0 seed 1 recorded a realized gain
of **−2953** against a normal range of ±1.5).

Worse, the 15 "compositions" were the allocations 6 solvers and 9 baselines
*chose*, so they were near-optimal by construction and often identical. On E0
seed 2, nine of fifteen shared the allocation `000000800000` — one predicted
value (+2.8381) against realized gains of −26.6, +2.6, −6.7, −3.2, −17.8, −1.1,
+2.2, −3.0, +1.9. That within-allocation spread was **15× the entire spread of
predicted values**.

### Which part of the fix mattered

Not averaging. Over 24 cells, averaging realized gain across draws moved
Spearman by **−0.005** — exactly nothing, helping in 12 of 24 cells:

| | effect of averaging |
|---|---|
| mean over 24 cells | −0.005 |
| cells where it helped | 12 / 24 |

What mattered was **de-duplicating allocations** and **widening the candidate
set** with random allocations, which breaks the range restriction the
solver-chosen set imposes. An earlier draft of this document attributed the fix
to averaging; the sweep refutes that.

### Conditions for a positive, null, or negative result

| | positive (11) | null (12) | negative (1) |
|---|---|---|---|
| noise/signal | 0.74× | 0.83× | **9.21×** |
| gain within-checkpoint R² | +0.726 | +0.259 | **−0.229** |
| effect Spearman (model fit) | +0.750 | +0.553 | **+0.085** |
| predicted spread | 1.505 | 0.449 | **0.092** |

The single negative cell is **seed 7 at `delta=0.40`**, where the data model
essentially failed to learn (effect Spearman 0.085, gain R² *negative*) *and*
measurement noise was 9.2×. A negative correlation therefore marks a failed fit
plus extreme noise, not a method that works backwards.

On 8 independent seeds, criterion 5 tracks how well the set-utility head fits:

```
C5 vs gain_within_r2       rho = +0.714   p = 0.047  *
C5 vs effect_spearman      rho = +0.643   p = 0.086   (n.s.)
```

### Step length acts through noise, not directly

`delta_scale` barely correlates with criterion 5 (rho −0.19), but it drives the
noise sharply within a seed:

| seed | noise/signal at 0.10 | at 0.20 | at 0.40 | growth |
|---|---|---|---|---|
| 2 | 1.08× | 1.72× | 4.12× | 3.8× |
| 6 | 0.22× | 0.35× | **5.63×** | **25.7×** |
| 7 | 0.65× | 0.97× | **9.21×** | **14.1×** |

Longer step → more out-of-distribution data → divergent updates → noise
explodes → criterion 5 degrades. That is the mechanism by which E0's
`delta_scale=0.4` made the statistic unreadable.

## Criterion 6: the overshoot hypothesis is refuted

The three E0 seeds suggested overshoot (realized ÷ planned latent displacement)
explained the collapse: 1.24× / 2.40× / 3.47× against cosines +0.314 / +0.459 /
+0.012. **24 cells do not support it:**

```
rho(C6, overshoot) = -0.23
overshoot per step length:  2.26x (0.10)   2.27x (0.20)   2.14x (0.40)
C6        per step length: +0.395         +0.384         +0.351
```

Overshoot is flat across step lengths and so is criterion 6. A three-point
correlation was reading a coincidence.

## Criterion 6: a fit-versus-controllability tension instead

On 8 independent seeds — `effect_spearman` is a property of the trained model
and is constant across the three step lengths within a seed, so pooling 24
cells would be pseudo-replication and inflates this to −0.814:

```
C6 vs effect_spearman      rho = -0.833   p = 0.010  *
C6 vs gain_within_r2       rho = -0.690   p = 0.058   (n.s.)
```

**The better the effect model fits, the worse the metadata direction control.**
The two extremes:

| seed | effect Spearman | C6 |
|---|---|---|
| 7 | **+0.085** (model failed) | **+0.494** |
| 4 | **+0.789** | **−0.075** (fails at all three step lengths) |

Per seed:

| seed | effect Spearman | gain R² | C5 mean | C6 mean |
|---|---|---|---|---|
| 0 | +0.690 | +0.162 | +0.228 | +0.492 |
| 1 | +0.514 | +0.409 | +0.035 | +0.502 |
| 2 | +0.805 | +0.887 | +0.442 | +0.222 |
| 3 | +0.679 | +0.304 | +0.408 | **+0.796** |
| 4 | +0.789 | +0.882 | **+0.796** | −0.075 |
| 5 | +0.701 | +0.692 | +0.608 | +0.356 |
| 6 | +0.725 | +0.517 | −0.021 | +0.222 |
| 7 | +0.085 | −0.229 | −0.023 | +0.494 |

A plausible mechanism: a sharper effect representation makes the latent space
*less linear* in the metadata, so the local linear Jacobian the mapper relies
on stops being valid. This also resolves the paradox in E0, where seed 2 had
the **best** held-out Jacobian R² (0.429) and the **worst** control (+0.012) —
a good local fit says nothing about validity over an executed step.

### How far to trust this

n = 8, and only one of the four correlations clears p < 0.05 once
pseudo-replication is removed. It is a strong lead, not a conclusion. It is
also not an absolute trade-off: seeds 3 and 5 do well on both criteria
(C5 +0.41 / +0.61, C6 +0.80 / +0.36). Criteria 5 and 6 are not even coupled
across seeds (rho = −0.41, p = 0.32), and seed 4 scores C5 +0.80 with C6
−0.075, so they are measuring genuinely different things.

## Separately: the checkpoint-ID A/B is inconclusive

The hypothesis was that E0's regression came from P0.4 removing the
checkpoint-ID embedding, since an older README recorded criterion 5 at +0.51.
Running the old behaviour (`--ckpt-id-ablation --val-split-by context`):

| seed | new (no ckpt-ID, held-out checkpoints) | old (ckpt-ID + context split) |
|---|---|---|
| 0 | +0.034 | **−0.414** |
| 2 | **−0.569** | −0.259 |

The two seeds disagree in direction, and the old behaviour additionally fails
criterion 6 on seed 2 (−0.098, 3/6 overall). Given that noise/signal reaches
9×, one run per seed cannot separate these variants. The old +0.51 is **not**
attributable to the checkpoint-ID flags.

## What follows

1. `run_stage0.py`'s criterion 5 should adopt the corrected measurement —
   de-duplicated allocations and a candidate set wider than what the solvers
   picked — so the gate measures what it intends to. Until then its criterion 5
   number should not be quoted.
2. The fit-versus-controllability tension needs a targeted test. If it holds it
   is a design constraint on the whole method, not a tuning detail.
3. E1 remains unrun. Its premise is now in better standing (criterion 5 is
   positive on 23 of 24 cells once measured correctly), but criterion 6 is
   still unresolved and is what makes a latent direction executable.
