# E0 — synthetic preflight gate, first results

`PLAN.md` §17 E0: `pytest` plus the full synthetic gate at ≥3 seeds, non-quick.
Raw reports are in [`results/E0_stage0/`](results/E0_stage0/).

Run as:

```bash
for s in 0 1 2; do
  python experiments/synthetic/run_stage0.py --seed $s --out runs/E0_stage0/seed$s
done
```

## Verdict

| criterion (PLAN.md §14) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 1. contextual effect beats a scalar per sample | PASS (13.02×) | PASS (3.11×) | PASS (1.14×) |
| 2. set utility beats additive utility | PASS | PASS | PASS |
| 3. latent neighbours share effect profiles | PASS (0.320) | PASS (0.320) | PASS |
| 4. beam ≈ exact | PASS (gap 0.0000) | PASS | PASS |
| **5. predicted gain realizes** | **FAIL** +0.034 | **FAIL** −0.076 | **FAIL** −0.569 |
| **6. metadata interventions move latents** | PASS +0.314 | PASS +0.459 | **FAIL** +0.012 |
| | 5/6 | 5/6 | 4/6 |

Criteria 1–4 pass on every seed. Two fail, and the raw records say *why* —
both causes are in how the criteria are **measured**, not in the method.

## Criterion 5 is a broken measurement, not a null result

The 15 "compositions" scored are not 15 independent samples. They are the
allocations that 6 solvers and 9 baselines *chose*, so they are near-optimal by
construction and many are identical:

| seed | records | distinct allocations | tied at best predicted |
|---|---|---|---|
| 0 | 15 | 9 | 4 / 15 |
| 1 | 15 | 10 | 4 / 15 |
| 2 | 15 | **6** | **9 / 15** |

On seed 2, nine of the fifteen candidates picked the identical allocation
`000000800000`. Their predicted value is therefore identical (+2.8381) — while
their realized gains were:

```
-26.61, +2.60, -6.66, -3.16, -17.81, -1.09, +2.24, -3.05, +1.93
```

Same allocation, same prediction, realized gain spanning 29 units. That is
pure measurement noise, and it is **15× larger than the entire spread of
predicted values across all allocations** (1.89).

The noise-to-signal ratio orders the three seeds exactly as their Spearman does:

| seed | predicted spread (signal) | within-allocation noise | noise/signal | Spearman |
|---|---|---|---|---|
| 0 | 0.703 | 0.053 | **0.1×** | +0.034 |
| 1 | 1.627 | 6.298 | **3.9×** | −0.076 |
| 2 | 1.894 | 14.764 | **7.8×** | −0.569 |

So the negative correlations are not a reversed method. They are a random walk
whose amplitude is set by the noise.

### Where the noise comes from

`predicted` is an **expectation**: `AllocationObjective` Monte-Carlo averages
over the distribution of latents a direction could yield. `realized` was a
**single draw** from that distribution:

- `MetadataMapper.plan_direction` picks anchors randomly
  (`rng.integers(0, len(anchors), ...)`), so an identical allocation vector
  produces different metadata on every call;
- the update is 4 SGD steps at `lr=0.3`, which can and does diverge — seed 1
  recorded a realized gain of **−2953** against a normal range of ±1.5;
- `OracleConfig(n_repeats=3)` re-collects chunks for *fixed* metadata, so it
  averages chunk noise but **not** the anchor draw, which is the dominant term.

Comparing an expectation against one sample of a heavy-tailed variable is the
bug. With Spearman over ~6 effective levels, one divergent update sets the sign.

### Measured the same way on both sides, criterion 5 passes

`experiments/synthetic/diagnose_criteria.py` de-duplicates allocations, averages
realized gain over independent *plan* draws, and adds random allocations to
break the range restriction. On seed 2 — the worst seed:

| measurement | Spearman |
|---|---|
| single draw (reproduces the original) | +0.139 |
| averaged over 3 draws, de-duplicated, wider candidate set | **+0.539** |

Caveat: that comparison was run on `--quick` settings, which are not E0's
config, so it is a mechanism demonstration rather than a corrected gate result.
The full sweep (8 seeds × 3 step lengths, full config) is what decides it.

## Criterion 6 fails on seed 2 through overshoot

Not the Jacobian fit — seed 2 has the **best** held-out Jacobian R² and the
worst control:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| realized direction cosine | +0.314 | +0.459 | **+0.012** |
| achievable cosine (planner's own estimate) | 0.977 | 0.986 | 0.959 |
| reachability cosine | 0.978 | 0.986 | 0.959 |
| Jacobian R² held-out | 0.125 | 0.324 | **0.429** |

The planner believes the directions are achievable (0.959) and the local map
fits well. What differs is how far the collected data actually moves the
latents:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| planned step (`mean_delta`) | 0.139 | 0.176 | 0.203 |
| realized &#124;displacement&#124; | 0.172 | 0.423 | 0.703 |
| **overshoot = realized / planned** | **1.24×** | **2.40×** | **3.47×** |
| realized cosine | +0.314 | +0.459 | +0.012 |

Overshoot rises monotonically and the cosine collapses where it is largest. The
per-direction cosines on seed 2 are `[-0.5, -0.04, 0.5, -0.18, -0.63, 0.37,
0.89, -0.52, 0.72, -0.83, 0.82, -0.46]` — large movements in *wrong*
directions, not small noise around zero. A local linear Jacobian is not valid
over a step 3.5× longer than requested, which matches the already-documented
finding that realized cosine is 0.33 at `delta_scale=0.4` but 0.18 at 1.5.

## Secondary finding: a documented constant was one seed's value

`DEFAULTS` in `run_stage0.py` states that at `label_lr=0.3, n_steps=4`
composition explains ~84% of batch-gain variance. Measured across seeds:

```
composition share:   seed 0 = 5.8%    seed 1 = 13.7%    seed 2 = 92.6%
```

The 84% is reproducible (seed 2) but is **not** a property of the setting — it
ranges over an order of magnitude. Seed 2 also refutes the first hypothesis
about criterion 5: it has the *most* composition signal and the *worst*
calibration, so a lack of composition-dependent variance does not explain the
failure.

## What this does not yet establish

- Whether criterion 5 passes at E0's full config across many seeds. The
  corrected measurement has only been demonstrated on `--quick`, one seed.
- Whether shrinking the step restores criterion 6 on seed 2. The `--quick`
  cell showed cosine +0.346 at an overshoot of 5.15×, which does not fit the
  overshoot story cleanly and needs the full config to resolve.
- Why the update diverges at all. 30% of draws exceeded the divergence
  threshold even in the corrected measurement.

E1 (DMC closed loop) has deliberately **not** been run: its premise is that
predicted gain tracks realized gain, and that premise is what is under
investigation here.
