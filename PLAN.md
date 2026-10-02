# PLAN.md — Scoring for Future Robot Data Acquisition

## 1. Goal

Build an end-to-end prototype for **future robot data acquisition guided by a learned data model**.

The core hypothesis is:

> A robot sample should not have a fixed scalar value. Its training effect is contextual: it depends on the current policy and the other samples used with it. We therefore learn a latent representation of sample-level optimization effect, learn a batch/set-level utility model on top of those latents, and use the resulting geometry to decide how future data support should be expanded under a fixed acquisition budget.

The MVP should answer three questions:

1. Can a learned latent vector represent the **optimization effect** of a robot sample across multiple batch/policy contexts?
2. Can a set-level utility model predict the training gain of a hypothetical future acquisition batch?
3. Can we use local latent domains and directional expansion to allocate acquisition budget better than simple baselines?

The first implementation should run entirely in simulation.

---

# 2. Problem Formulation

At acquisition round `t`, we have:

- current policy: `theta_t`
- existing dataset: `D_t`
- acquisition budget: `B`
- each sample:
  - robot data `x_i`
  - acquisition metadata `m_i`
  - optionally trajectory / chunk context
- a fixed downstream utilization rule `pi_util`

Examples of `x_i`:

- transition `(s_t, a_t, s_{t+1})`
- trajectory chunk
- full demonstration trajectory

For the first version, prefer **trajectory chunks** over isolated transitions because they retain local temporal context.

We learn:

```text
E_phi(x_i, theta_t) -> z_i
```

where `z_i` is an **effect latent**.

A separate set utility model predicts:

```text
F_psi({z_i}, D_t, theta_t) -> predicted training gain
```

The latent should satisfy local geometry constraints:

```text
similar optimization behavior -> nearby latent vectors
nearby latent vectors -> similar contextual training effects
small policy change -> small contextual score change
small batch-context change -> small contextual score change
```

The acquisition problem is not:

```text
select one sample x with the highest scalar score
```

Instead, it is:

```text
select a composition of future data whose joint predicted marginal gain is maximal
```

---

# 3. Definitions

## 3.1 Contextual sample effect

For sample `x_i`, batch/context `B`, and policy `theta`:

```text
s_i(B, theta)
```

is the contribution of `x_i` under that context.

A preferred expensive target is a leave-one-out intervention:

```text
Delta_i(B, theta)
=
U(Update(theta, B))
-
U(Update(theta, B \ {x_i}))
```

where `U` is a downstream evaluation utility.

Because exact leave-one-out retraining is expensive, support multiple supervision targets:

1. exact / short-horizon leave-one-out gain
2. gradient alignment proxy
3. influence-function proxy
4. change in validation / evaluation loss after a local update

The implementation must keep the target interface modular.

---

## 3.2 Multi-context supervision

A sample must appear under multiple contexts:

```text
(x_i, B_1, theta_1)
(x_i, B_2, theta_1)
(x_i, B_3, theta_2)
...
```

This is mandatory.

The model must not learn a single historical scalar attached to `x_i`.

Instead, it must learn a representation that supports different contextual readouts.

---

## 3.3 Local effect domain

After learning `z_i`, cluster the latent space.

A local effect domain `C_k` is a cluster of samples whose latent vectors are nearby.

The intended semantic is:

```text
within a local domain, nearby samples have similar interaction profiles
with downstream batch/policy contexts
```

For the MVP, use Euclidean geometry in latent space only if metric regularization successfully enforces that geometry.

---

# 4. Main Architecture

Implement the following modules separately.

## 4.1 Sample Encoder

```text
E_phi(x, theta_context) -> z
```

Possible inputs:

- trajectory chunk representation
- current policy features
- optional task/env descriptors

MVP options:

- MLP for state-based robotics environments
- Transformer/GRU for trajectory chunks
- policy context can initially be represented by:
  - training step
  - checkpoint embedding
  - compressed policy statistics
  - or omitted in the very first smoke test

Do not begin with image observations.

Start with low-dimensional state-based environments.

---

## 4.2 Context / Set Encoder

Given a batch:

```text
Z_B = {z_1, ..., z_n}
```

encode it using a permutation-invariant model:

MVP:

```text
DeepSets
```

Alternative later:

```text
Set Transformer
```

The model should support:

```text
ContextEncoder(Z_B) -> h_B
```

---

## 4.3 Sample Effect Readout

Predict the contextual contribution of a sample:

```text
R_phi(z_i, h_{B\i}, theta_context) -> s_hat_i
```

Supervise with multi-context influence/effect labels.

This module is mainly used to shape the latent geometry.

---

## 4.4 Batch Utility Model

Predict utility of the full batch:

```text
F_psi(Z_B, D_context, theta_context) -> V_hat(B)
```

This must capture:

- redundancy
- complementarity
- saturation
- batch composition effects

Do not assume additive sample utility.

This is the model later used by the acquisition planner.

---

# 5. Training Objectives

Use a weighted combination of objectives.

## 5.1 Contextual effect prediction

```text
L_effect =
MSE(
    R_phi(z_i, h_{B\i}, theta),
    target_effect_i(B, theta)
)
```

---

## 5.2 Batch utility prediction

```text
L_batch =
MSE(
    F_psi(Z_B, theta),
    target_batch_gain(B, theta)
)
```

Possible batch gain target:

```text
U(Update(theta, B)) - U(theta)
```

measured after one or a small number of policy update steps.

---

## 5.3 Metric / local geometry regularization

Encourage latent distance to reflect similarity of optimization behavior.

For two samples `i, j`:

```text
d_effect(i,j)
=
average_c |target_effect_i(c) - target_effect_j(c)|
```

over shared or matched contexts.

Train latent distances to correlate with effect distances:

```text
L_metric =
| ||z_i-z_j||_2 - normalize(d_effect(i,j)) |
```

Alternative:

- contrastive loss
- triplet loss
- neighborhood consistency loss

Start with contrastive/triplet if direct regression is unstable.

---

## 5.4 Local smoothness

For nearby samples / nearby policies:

```text
small input/context perturbation
=> small score perturbation
```

Possible regularizer:

```text
L_smooth =
|s_hat_i(B, theta) - s_hat_i(B', theta')|
```

for deliberately sampled nearby `(B', theta')`.

Do not over-weight this term.

---

## 5.5 Optional acquisition-metadata regularization

Each sample has metadata:

```text
m_i
```

Examples:

- task ID
- object pose
- initial state
- goal pose
- source/operator
- difficulty
- scene configuration

Do **not** force metadata geometry and effect geometry to be globally identical.

Instead, optionally enforce local predictability:

```text
Delta m -> predictable Delta z
```

Fit a local directional model:

```text
G_omega(z, m, Delta m) -> Delta z_hat
```

Loss:

```text
L_meta =
|| (z_j - z_i) - G_omega(z_i, m_i, m_j-m_i) ||^2
```

for locally matched sample pairs.

Keep this optional in the first MVP.

---

## 5.6 Total loss

Start with:

```text
L =
L_effect
+ lambda_batch * L_batch
+ lambda_metric * L_metric
+ lambda_smooth * L_smooth
```

Later:

```text
+ lambda_meta * L_meta
```

---

# 6. Generating Supervision Data

This is critical.

For each policy checkpoint:

```text
theta_1, theta_2, ..., theta_T
```

sample many training batches.

For each batch:

1. compute baseline policy update
2. compute batch-level gain
3. generate sample-level effect targets
4. store:
   - sample IDs
   - batch composition
   - policy checkpoint
   - metadata
   - target effects
   - batch gain

Dataset format:

```text
ContextRecord:
    policy_id
    batch_sample_ids
    batch_gain
    per_sample_effects
```

Each sample should appear in many different contexts.

Recommended minimum:

```text
>= 20 contexts/sample
```

for initial experiments, if computationally feasible.

---

# 7. Latent Clustering

After encoder pretraining:

```text
z_i = E_phi(x_i, theta_ref)
```

Cluster latent vectors.

MVP:

```text
KMeans
```

Also test:

```text
GMM
HDBSCAN
```

Do not assume one clustering method is part of the contribution.

Store per cluster:

```text
ClusterState:
    cluster_id
    member_ids
    centroid
    covariance
    PCA basis
    metadata distribution
    boundary points
```

---

# 8. Candidate Direction Generation

For each cluster `C_k`:

## 8.1 Compute local PCA

```text
Sigma_k = Cov(z_i in C_k)
```

Eigen decomposition:

```text
lambda_1 >= lambda_2 >= ...
v_1, v_2, ...
```

Choose smallest intrinsic rank `r_k` such that:

```text
sum_{j<=r_k} lambda_j / sum_j lambda_j >= rho
```

Default:

```text
rho = 0.90
```

Cap:

```text
r_k <= r_max
```

Default:

```text
r_max = 5
```

Generate candidate directions:

```text
+v_1, -v_1, ..., +v_r, -v_r
```

Maximum directions per domain:

```text
2 * r_k
```

---

## 8.2 Keep only outward directions

A direction is useful only if it expands the current support.

For each direction `v`:

1. choose one or more boundary anchors `z_b`
2. propose:
   ```text
   z' = z_b + delta * v
   ```
3. keep the direction if:
   - distance to cluster centroid increases
   - local density decreases
   - but the point is not too far from observed support

Use a trust-region constraint:

```text
dist(z', support) <= epsilon_expand
```

This prevents meaningless long-range extrapolation.

---

# 9. Hypothetical Future Latent Generation

For each candidate acquisition direction:

```text
a = (cluster k, direction v)
```

Generate hypothetical future samples in latent space:

```text
z_new =
z_boundary
+ delta * v
+ epsilon
```

where:

```text
epsilon ~ local residual distribution
```

MVP:

```text
epsilon ~ N(0, sigma^2 * Sigma_local)
```

This approximates uncertainty in future collected samples.

Do not represent one direction with a single deterministic latent point.

---

# 10. Future Batch Utility

Let there be `A` candidate acquisition directions.

An allocation is:

```text
n = (n_1, ..., n_A)
```

with:

```text
sum_a n_a = B
```

For allocation `n`:

1. sample `n_a` hypothetical latents from each direction
2. combine into a hypothetical future acquisition batch
3. feed the complete batch into the batch utility model
4. repeat using Monte Carlo

Estimate:

```text
V_hat(n | D, theta)
=
mean_m F_psi(Z_future^(m), D, theta)
```

This is the predicted utility of the **whole acquisition composition**.

It must not be reduced to independent per-direction scores.

---

# 11. Solving for Q*

We want:

```text
n* =
argmax_n V_hat(n | D, theta)

subject to:
sum_a n_a <= B
```

## 11.1 Exact enumeration

Use when both `A` and `B` are small.

Number of allocations:

```text
C(B + A - 1, A - 1)
```

This serves as an oracle for small experiments.

Implement this first.

---

## 11.2 Beam search

Use for larger problems.

Pseudo-code:

```python
beam = {zero_allocation}

for step in range(B):
    candidates = []

    for n in beam:
        for a in candidate_directions:
            n_new = n.copy()
            n_new[a] += 1

            score = monte_carlo_predict_utility(n_new)

            candidates.append((n_new, score))

    beam = top_H_unique(candidates)

return best(beam)
```

Hyperparameter:

```text
beam width H
```

Start with:

```text
H = 10 or 20
```

---

## 11.3 Greedy baseline

Implement:

```text
choose direction with maximum one-step marginal gain
```

at every step.

This is a baseline, not the main planner.

It will expose whether complementarity matters.

---

# 12. Mapping Latent Directions to Acquisition Metadata

This is required for actual acquisition.

For each local domain, learn a local forward map:

```text
f_meta:
m -> z
```

or directional map:

```text
Delta m -> Delta z
```

Given a target latent direction `v`, solve:

```text
Delta m*
=
argmin_Delta_m
|| G_k Delta_m - alpha v ||^2
```

subject to metadata feasibility constraints.

MVP metadata should be simulator-controlled variables.

Good initial examples:

- object initial position
- target position
- initial robot configuration
- task difficulty
- object category if discrete
- perturbation magnitude

Do not begin with free-form real-world metadata.

---

# 13. Acquisition Loop

Each acquisition round:

```text
1. Train/update policy theta_t on D_t
2. Generate multi-context supervision
3. Update data model
4. Encode D_t into latent space
5. Cluster latent space
6. Generate outward candidate directions
7. Predict batch utility for candidate allocations
8. Solve for Q*
9. Map selected directions to metadata perturbations
10. Collect new robot data
11. D_{t+1} = D_t union D_new
12. Repeat
```

---

# 14. Important Assumptions

State these explicitly in code/docs.

## A1. Local effect smoothness

Nearby effect latents have similar contextual optimization behavior.

## A2. Local acquisition predictability

Small controllable metadata changes produce locally predictable latent changes.

This does not require global invertibility.

## A3. Short-horizon stationarity

Within one acquisition round, the current data model remains predictive enough for the next acquisition decision.

## A4. Fixed utilization rule

The first version assumes a fixed downstream training/utilization rule.

Example:

```text
uniform minibatch sampling from D_t union D_new
```

The acquisition planner optimizes under this fixed rule.

Do not jointly optimize utilization yet.

---

# 15. MVP Experimental Setting

Start small.

Recommended environments:

1. ManiSkill state-based tasks
2. MetaWorld state-based tasks
3. DMC / MuJoCo for debugging

Prefer a setting where acquisition metadata is controllable.

Example task family:

```text
PushCube / PickCube
```

Metadata:

```text
object initial x/y
goal x/y
robot initial configuration
task variation
```

Policy:

```text
PPO or SAC
```

For the first implementation, PPO is preferable if an existing codebase is already available.

---

# 16. Baselines

Implement simple baselines first.

## Acquisition baselines

1. Uniform random acquisition
2. Equal budget per metadata-defined region
3. Novelty / diversity acquisition
4. Uncertainty acquisition
5. Gradient-norm acquisition
6. Gradient-alignment-based acquisition
7. Greedy predicted utility
8. Proposed beam-search set utility acquisition

Later add robotics-specific data curation/acquisition baselines where compatible.

---

# 17. Evaluation

Evaluate three levels separately.

## 17.1 Representation quality

Measure:

- contextual effect prediction error
- batch utility prediction error
- nearest-neighbor consistency
- latent distance vs effect-distance correlation
- stability across nearby policy checkpoints
- stability across slightly modified batch contexts

---

## 17.2 Acquisition prediction quality

For predicted candidate allocation `Q`:

- predicted gain
- realized gain after actual collection/training
- rank correlation over candidate acquisition compositions

Important:

The core test is not whether individual sample scores are correct.

The core test is:

```text
Can the model rank future acquisition batches correctly?
```

---

## 17.3 Downstream acquisition performance

Under the same acquisition budget:

```text
final policy return / success rate
```

versus:

```text
number of newly acquired trajectories
collection cost
```

Plot:

```text
policy performance vs acquisition budget
```

---

# 18. Critical Ablations

Must include:

1. scalar sample score vs latent effect representation
2. single-context vs multi-context supervision
3. additive utility vs set-level utility
4. no metric loss vs metric loss
5. no clustering vs clustered local domains
6. random directions vs PCA/local directions
7. local-only exploitation vs outward expansion
8. greedy vs beam search
9. exact enumeration vs beam search on small problems
10. no metadata-direction model vs metadata-aware acquisition
11. latent dimension sweep

---

# 19. Failure Modes to Detect Early

Stop or redesign if any of the following occurs.

## F1. No stable effect geometry

If nearby effect latents do not have similar contextual behavior, clustering-based acquisition is not justified.

## F2. Batch utility is nearly additive

If:

```text
F({z_i}) ~= sum f(z_i)
```

then the batch-interaction motivation is weak.

Measure this explicitly.

## F3. Multi-context labels are too noisy

If leave-one-out/influence labels vary mostly due to estimator noise, latent learning will collapse.

## F4. Latent directions are not metadata-actionable

If no local metadata perturbation can reliably move latent support, directional acquisition cannot be executed.

## F5. Out-of-support utility predictions are unreliable

Use trust regions and uncertainty estimates.

Do not extrapolate arbitrarily far.

---

# 20. Implementation Order

## Phase 0 — Synthetic sanity test

Before robotics:

Create synthetic samples with known:

- latent effect factors
- redundancy
- complementarity
- metadata-to-latent map

Verify the full pipeline can recover useful acquisition directions.

Deliverable:

```text
tests/test_synthetic_acquisition.py
```

---

## Phase 1 — Supervision generator

Implement:

```text
generate_context_records.py
```

Outputs:

```text
policy checkpoints
batch composition
per-sample effect target
batch gain
metadata
```

---

## Phase 2 — Data model

Implement:

```text
models/sample_encoder.py
models/set_encoder.py
models/effect_readout.py
models/batch_utility.py
```

Train on stored context records.

---

## Phase 3 — Latent diagnostics

Implement:

```text
analysis/latent_geometry.py
```

Produce:

- PCA plots
- nearest-neighbor effect consistency
- local smoothness metrics
- cluster statistics

Do not proceed to acquisition unless the latent geometry is meaningful.

---

## Phase 4 — Clustering and directions

Implement:

```text
acquisition/clustering.py
acquisition/directions.py
```

Support:

```text
KMeans
local PCA
boundary detection
outward filtering
trust region
```

---

## Phase 5 — Hypothetical acquisition simulator

Implement:

```text
acquisition/latent_sampler.py
```

Input:

```text
cluster
direction
budget
```

Output:

```text
hypothetical future latent batch
```

---

## Phase 6 — Allocation planner

Implement:

```text
acquisition/exact_search.py
acquisition/greedy.py
acquisition/beam_search.py
```

All planners call the same:

```text
predict_allocation_utility(allocation)
```

---

## Phase 7 — Metadata controller

Implement:

```text
acquisition/metadata_mapper.py
```

First version:

- local linear regression
- local Jacobian estimate
- constrained least squares inversion

---

## Phase 8 — Closed-loop acquisition experiment

Implement:

```text
experiments/run_acquisition_loop.py
```

Run:

```text
train policy
-> fit/update datamodel
-> cluster
-> generate directions
-> plan acquisition
-> collect
-> retrain
-> evaluate
```

---

# 21. Suggested Repository Structure

```text
project/
|
|-- configs/
|   |-- env/
|   |-- datamodel/
|   |-- acquisition/
|
|-- data/
|   |-- context_dataset.py
|   |-- metadata.py
|
|-- models/
|   |-- sample_encoder.py
|   |-- set_encoder.py
|   |-- effect_readout.py
|   |-- batch_utility.py
|
|-- supervision/
|   |-- influence.py
|   |-- gradient_alignment.py
|   |-- leave_one_out.py
|   |-- generate_context_records.py
|
|-- acquisition/
|   |-- clustering.py
|   |-- directions.py
|   |-- latent_sampler.py
|   |-- metadata_mapper.py
|   |-- exact_search.py
|   |-- greedy.py
|   |-- beam_search.py
|
|-- analysis/
|   |-- latent_geometry.py
|   |-- acquisition_calibration.py
|   |-- plots.py
|
|-- experiments/
|   |-- train_policy.py
|   |-- train_datamodel.py
|   |-- run_acquisition_loop.py
|
|-- tests/
|   |-- test_synthetic_acquisition.py
|   |-- test_set_utility.py
|   |-- test_direction_generation.py
|
|-- PLAN.md
```

---

# 22. First Coding Milestone

Do **not** begin with the entire closed loop.

The first milestone is:

```text
Given:
- a fixed set of samples
- multiple policy checkpoints
- multiple random batches

Train:
- E_phi
- contextual effect readout
- batch utility model

Verify:
1. same sample under different contexts gets correctly predicted effects
2. nearby latent samples have similar effect profiles
3. batch utility prediction beats an additive scalar-score baseline
```

If this milestone fails, do not proceed to clustering/acquisition.

---

# 23. Second Coding Milestone

On a fixed trained latent space:

```text
1. cluster samples
2. generate PCA directions
3. synthesize hypothetical future latent batches
4. compare exact search / greedy / beam search
5. test whether the planner recovers known high-value compositions
```

Use a synthetic oracle before real robot acquisition.

---

# 24. Third Coding Milestone

Connect acquisition directions to simulator metadata.

Verify:

```text
desired latent direction
-> predicted metadata perturbation
-> actual collected sample
-> realized latent movement
```

Metric:

```text
cosine_similarity(
    desired_latent_direction,
    realized_latent_delta
)
```

Only after this is stable should the full closed-loop acquisition experiment be run.

---

# 25. Main Research Claim to Preserve During Implementation

Do not let implementation drift into ordinary data curation.

The intended claim is:

> We learn a policy-conditioned latent model of how robot data interacts during optimization, use its local geometry to represent actionable directions for expanding the data distribution, and optimize the composition of future acquisitions based on predicted joint training utility.

The three key distinctions are:

```text
1. future acquisition, not only selection of already collected data
2. contextual/set-level utility, not fixed scalar sample value
3. directional expansion of effect domains, not only resampling known regions
```

---

# 26. Questions to Revisit After Initial Results

Do not solve these before the MVP unless required by experiments.

1. Is PCA the right way to define expansion directions?
2. Should domains be hard clusters or soft/local neighborhoods?
3. Should policy context be explicitly embedded?
4. Can acquisition and utilization be jointly optimized?
5. Is a separate control latent needed in addition to effect latent?
6. Does the learned geometry exhibit a stable empirical phenomenon worth formalizing?
7. Can latent direction generation be learned rather than PCA-based?
8. Can the approach scale from state-based robots to vision/VLA data?
