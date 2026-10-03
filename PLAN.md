# PLAN.md — LDVA: Latent Data Valuation for Budgeted Robot Acquisition

This file is the **single source of truth** for the project. `SETUP.md` is retired.

LDVA studies **prospective robot-data acquisition**, not only curation of data that has already been collected.

The central hypothesis is:

> A robot sample does not have a fixed scalar value. Its effect on learning is contextual: it depends on the current policy and on the other samples used with it. LDVA therefore learns an effect-aware latent representation, predicts the joint utility of future data compositions, and uses the learned geometry to decide where the robot-data distribution should expand under a limited acquisition budget.

---

# 1. Research Scope

## 1.1 Problem

At acquisition round `t`, LDVA has:

- current policy `theta_t`
- owned dataset `D_t`
- fixed downstream utilization rule `pi_util`
- controllable acquisition metadata `m`
- acquisition budget `B` or monetary budget `C`

Each observed sample/chunk is:

```text
x_i, m_i
```

LDVA learns a policy-conditioned effect latent:

```text
E_phi(x_i, theta_t) -> z_i
```

and a set-level utility model:

```text
F_psi({z_i}, D_t, theta_t) -> predicted training gain
```

The acquisition problem is **not**:

```text
argmax_x scalar_score(x)
```

It is:

```text
choose a composition of future, controllable data whose joint downstream gain is maximal
```

under a count or monetary budget.

---

## 1.2 Three research questions

### RQ1 — Effect representation

Can multi-context optimization effects be compressed into a locally meaningful latent geometry?

Desired properties:

```text
similar optimization behavior -> nearby z
nearby z -> similar contextual effects
small batch-context changes -> small effect changes
small policy changes -> small effect changes
```

### RQ2 — Prospective set valuation

Can this geometry predict the joint utility of **data that has not yet been collected**?

The model must capture:

- redundancy
- complementarity
- saturation
- interaction with the current dataset
- interaction with the current policy

### RQ3 — Actionable acquisition

Can promising directions in effect space be translated into controllable robot-data collection conditions and improve downstream robot performance under a fixed budget?

---

# 2. What LDVA Is Not

LDVA is not intended to be only:

```text
sample scoring
existing-data filtering
existing-dataset retrieval
domain reweighting
variance allocation
latent visualization
```

Those can appear as baselines or intermediate components.

The full claim requires:

```text
contextual effect representation
+ set-level prospective utility
+ future controllable acquisition
+ budgeted robot-data collection
```

---

# 3. Core Formulation

## 3.1 Sample unit

Default sample unit:

```text
trajectory chunk
```

rather than an isolated transition.

Reason:

- preserves local temporal context
- creates enough samples for multi-context supervision
- is easier to associate with acquisition metadata than arbitrary minibatch fragments

Each sample record should contain:

```text
sample_id
trajectory_id
start_t / end_t
observation chunk
action chunk
reward / success
policy/source checkpoint
acquisition metadata
round_id
cost
```

---

## 3.2 Contextual sample effect

For sample `x_i`, batch `B`, and policy `theta`:

```text
s_i(B, theta)
```

is the contribution of `x_i` inside that optimization context.

Preferred expensive target:

```text
Delta_i(B, theta)
=
U_local(Update(theta, B))
-
U_local(Update(theta, B \ {x_i}))
```

Possible supervision estimators:

1. exact / short-horizon leave-one-out
2. one-step update effect
3. gradient alignment
4. influence-function proxy
5. TRAK-like proxy

Cheap proxies are scaling tools, not automatically ground truth.

---

## 3.3 Multi-context supervision

The same sample must appear under multiple contexts:

```text
(x_i, B_1, theta_1)
(x_i, B_2, theta_1)
(x_i, B_3, theta_2)
...
```

This is mandatory.

A sample must **not** be assigned one historical scalar label.

Recommended development target:

```text
>= 20 contexts/sample
```

with multiple policy checkpoints and multiple independent restarts.

---

## 3.4 Two notions of utility

This distinction is critical.

### Dense supervision utility

During representation learning, a cheap local utility may be used to generate many labels, e.g.:

```text
U_local(theta) = - BC validation loss
```

This is acceptable for learning optimization-effect structure.

### Paper-level downstream utility

The acquisition claim must be evaluated with **actual environment performance** after training/retraining:

```text
U_robot(theta)
=
rollout success rate / return on a fixed evaluation distribution
```

Therefore the paper-level realized acquisition gain is:

```text
Gain(Q)
=
U_robot(Train(D_t union Collect(Q)))
-
U_robot(Train(D_t))
```

A thresholded BC action-MSE proxy must **not** be called robot success.

---

# 4. Model

## 4.1 Sample encoder

```text
E_phi(x, theta_context) -> z
```

MVP:

```text
MLP / GRU on state-action trajectory chunks
```

Later:

```text
Transformer
visual encoder
JEPA-style predictive objectives
```

Do not begin with image reconstruction or a VAE unless experiments show it is necessary.

---

## 4.2 Policy context

The main experiment must use a policy representation that can generalize to **unseen future checkpoints**.

Preferred main representation:

```text
continuous policy features
```

Examples:

```text
training progress
train loss
held-out loss
gradient norm
compact learned policy statistics
```

### Important rule

Do **not** rely on checkpoint-ID embeddings in the main result.

A checkpoint-ID embedding can memorize observed checkpoints and does not define a representation for an unseen future policy.

Use checkpoint embeddings only as an ablation if desired.

Validation must include:

```text
hold out entire policy checkpoints
```

not only unseen batch compositions.

---

## 4.3 Context / set encoder

Given:

```text
Z_B = {z_1, ..., z_n}
```

use a permutation-invariant context encoder.

MVP:

```text
DeepSets
```

Later:

```text
Set Transformer
```

---

## 4.4 Contextual effect readout

```text
R_phi(z_i, h_{B\i}, theta_context) -> s_hat_i
```

This readout shapes the latent geometry and tests whether sample effects are truly contextual.

---

## 4.5 Batch utility model

```text
F_psi(Z_B, D_t, theta_context) -> V_hat(B)
```

The set model must be compared against an additive model.

If the set model does not outperform a properly cross-validated additive model, the batch-interaction motivation is weak and must be re-examined.

---

# 5. Training Objectives

Use:

```text
L
=
L_effect
+ lambda_batch L_batch
+ lambda_metric L_metric
+ lambda_smooth L_smooth
+ optional lambda_meta L_meta
```

## 5.1 Effect loss

```text
L_effect
=
MSE(s_hat_i(B, theta), target_effect_i(B, theta))
```

## 5.2 Batch utility loss

```text
L_batch
=
MSE(V_hat(B), target_batch_gain(B, theta))
```

## 5.3 Metric loss

Euclidean distance in latent space is allowed only if training explicitly gives it effect semantics.

For matched sample pairs:

```text
effect_distance(i,j)
=
average contextual difference across shared/matched contexts
```

Encourage:

```text
||z_i - z_j||_2
```

to track effect-profile difference.

Possible implementations:

- contrastive
- triplet
- neighborhood consistency
- direct metric regression

## 5.4 Smoothness

For nearby batch/policy contexts:

```text
small context perturbation -> small predicted-effect perturbation
```

Use this as a regularizer, not as the primary objective.

## 5.5 Metadata-direction regularization

Do not force metadata geometry and effect geometry to be globally identical.

Require only local predictability:

```text
Delta m -> predictable Delta z
```

Possible model:

```text
G_omega(z, m, Delta m) -> Delta z_hat
```

---

# 6. Local Effect Domains

After training at a reference policy `theta_ref`:

```text
z_i = E_phi(x_i, theta_ref)
```

cluster the effect latent space.

MVP:

```text
KMeans
```

Ablations:

```text
GMM
HDBSCAN
soft/local neighborhoods
```

A local effect domain is intended to contain samples with similar **interaction profiles**, not merely visually similar samples.

Store:

```text
cluster_id
members
centroid
covariance
PCA basis
boundary samples
metadata statistics
```

---

# 7. Directional Expansion

## 7.1 Candidate directions

For cluster `C_k`, compute local PCA:

```text
Sigma_k = Cov(z_i in C_k)
```

Choose the smallest rank `r_k` satisfying:

```text
sum_{j<=r_k} lambda_j / sum_j lambda_j >= rho
```

Default:

```text
rho = 0.90
r_k <= 5
```

Initial candidates:

```text
+v_1, -v_1, ..., +v_r, -v_r
```

## 7.2 Outwardness

For boundary anchor `z_b`:

```text
z' = z_b + delta v
```

Keep only candidates that:

- move outward relative to local support
- remain inside a latent trust region
- retain reasonable local-density support
- are actionable through acquisition metadata

Latent-space outwardness alone is insufficient.

## 7.3 Actionability

A direction is actionable only if feasible metadata perturbations can approximately realize it.

Locally fit:

```text
Delta z ~= J_k Delta m
```

Then solve:

```text
Delta m*
=
argmin ||J_k Delta m - alpha v||^2
```

subject to the environment's true feasible metadata set.

The declared metadata parameterization must match the environment's feasible geometry; avoid boxes that secretly require clipping onto curved/non-box feasible sets.

---

# 8. Prospective Future-Batch Utility

For candidate acquisition directions `a_1, ..., a_A`, allocation is:

```text
n = (n_1, ..., n_A)
```

with count budget:

```text
sum_a n_a <= B
```

or monetary budget:

```text
sum_a c_a n_a <= C
```

For each direction, generate hypothetical future latents around boundary anchors:

```text
z_new
=
z_boundary + delta v + epsilon
```

where `epsilon` follows a local residual distribution.

Estimate joint utility by Monte Carlo:

```text
V_hat(n | D_t, theta_t)
=
mean_m F_psi(Z_future^(m), D_t, theta_t)
```

Do not collapse the composition into independent scalar direction scores.

---

# 9. Solving for Q*

Optimization problem:

```text
n*
=
argmax_n V_hat(n | D_t, theta_t)
```

subject to budget constraints.

Implement and compare:

## Exact search

Use on small candidate spaces as the learned-model oracle.

## Greedy

Repeatedly add one unit to the direction with highest predicted marginal gain.

This is an LDVA ablation / simple planner.

## Beam search

Primary practical planner.

Beam search is useful when complementarity makes greedy suboptimal.

Compare beam to exact search on small problems and report relative gap and evaluation count.

---

# 10. Acquisition Loop

The real simulator loop must use `EnvAdapter`, not a synthetic-world special case.

Each round:

```text
1. train/retrain policy on D_t
2. evaluate rollout performance on a fixed held-out evaluation distribution
3. generate multi-context supervision from policy checkpoints
4. train/update LDVA data model
5. encode D_t
6. cluster local effect domains
7. generate actionable outward directions
8. predict utilities of candidate future compositions
9. solve Q*
10. map chosen directions to metadata requests
11. collect new data through EnvAdapter.collect
12. D_{t+1} = D_t union D_new
13. repeat
```

The evaluation distribution must be defined **before acquisition** and never changed to favor the collected data.

---

# 11. Environment Interface

Every simulator / real-data source should expose:

```text
metadata_spec
collect(metadata)
evaluation_conditions / evaluation_set
acquisition_cost(metadata)
evaluate_policy(policy, fixed_eval_conditions)
```

The last function is required for paper-level robotics evaluation.

`evaluation_set(obs, act)` may remain useful for dense BC supervision, but it is not a substitute for rollout evaluation.

---

# 12. Baselines

There are two distinct groups. Do not mix them in the paper.

## 12.1 LDVA internal ablations

These may reuse the LDVA representation/model because they isolate one design choice:

```text
LDVA scalar readout
LDVA additive utility
LDVA no-context
LDVA single-context supervision
LDVA no metric loss
LDVA no clustering
LDVA random directions
LDVA local-only / no outward expansion
LDVA greedy vs beam
LDVA checkpoint-ID context vs continuous policy context
```

The existing `*_style` implementations that score hypothetical directions through the LDVA latent model belong here.

Do **not** present them as reproductions of published methods.

## 12.2 Independent external baselines

These must compute their own scores/objectives from their own method assumptions rather than borrowing LDVA predictions.

Minimum local-stage baselines:

```text
Random / Uniform acquisition
Diversity / Core-set acquisition
Direct Gradient Alignment
Direct Influence
```

Paper-level baseline targets:

```text
Influence Functions
TracIn
DemInf
CUPID
Re-Mix
DataMIL
QoQ
```

Optional / scale-dependent:

```text
Data Shapley on small problems
ATHENA for large VLA-scale experiments
```

For published methods whose original problem is post-hoc curation rather than future acquisition, clearly define the prospective adaptation and document the deviation.

---

# 13. Benchmark and Policy Roadmap

## Local development

Primary choices:

```text
DMC / MuJoCo state-based for cheap debugging
MetaWorld state-based for the first meaningful manipulation result
```

Policy:

```text
MLP Behavior Cloning
```

Embodiment:

```text
DMC task-specific embodiment
MetaWorld Sawyer
```

Do not block local work on ManiSkill; Vulkan/driver dependence is unnecessary for the MVP.

## Server experiments

Scale to:

```text
MetaWorld multi-task
robosuite Panda
optional LIBERO
optional DMC/MuJoCo generality suite
```

Policy architectures:

```text
MLP BC for controlled analysis
Diffusion Policy as main stronger policy
ACT as architecture-generalization / teleop-friendly policy
```

## Real robot

Preferred if accessible:

```text
Franka Panda / FR3
```

Low-cost fallback:

```text
SO-101
```

Use the same or compatible embodiment/sensor/action interface as the acquired teleoperation data whenever possible.

---

# 14. Project Stages

The project uses four research stages.

## Stage 0 — Literature review, overlap, and research-gap validation

Goal:

```text
ensure the novelty is the intersection of prospective acquisition,
contextual optimization effect, set utility, and robot controllability
```

Nearest neighbors to track closely:

```text
Datamodels
AirRep
Inter-Sample Influence Graphs
DataMIL
CUPID
DemInf
Re-Mix
QoQ
ATHENA
cost-aware multi-source data acquisition
```

Questions to keep updated:

- Has someone already learned an attribution/effect representation?
- Has someone already modeled sample interactions?
- Has someone already optimized future acquisition rather than existing-data selection?
- Has someone translated latent/effect directions into controllable robot collection conditions?

### Exit criterion

Maintain a defensible gap statement:

> Existing methods can value, filter, retrieve, or reweight collected data, and some model training-data effects or interactions. LDVA targets the missing prospective loop: learn an optimization-effect geometry, predict the joint utility of controllable future robot data, and use that geometry to expand the data distribution under an acquisition budget.

---

## Stage 1 — Local machine MVP

Goal:

```text
prove the mechanism on cheap real simulators
```

### Stage 1A — Synthetic preflight

The existing synthetic benchmark is a **preflight gate**, not the research Stage 0.

It should validate:

- contextual effect prediction
- set utility > additive utility where interactions exist
- effect-neighbor consistency
- beam ≈ exact on small problems
- predicted acquisition ranking correlates with realized local gain
- metadata interventions move latents in intended directions

### Stage 1B — DMC smoke experiment

Purpose:

```text
verify the full closed-loop plumbing through a real simulator
```

Use 3 development seeds.

Required change before running:

```text
run_acquisition_loop.py must operate through EnvAdapter
```

Measure actual rollout return, not only BC validation loss.

### Stage 1C — MetaWorld first meaningful experiment

Start with:

```text
push-v3
```

then add:

```text
reach-v3
pick-place-v3
drawer-open-v3
button-press-v3
```

First meaningful plot:

```text
MetaWorld rollout success rate vs acquired episodes
```

Minimum comparison:

```text
Random
Diversity
Direct Gradient Alignment
Direct Influence
LDVA Greedy
LDVA Beam
```

### Stage 1 exit criteria

Proceed to server only if:

1. set utility predicts unseen batch compositions better than additive utility
2. acquisition ranking predicts realized gain
3. LDVA improves rollout success/return under equal acquisition budget on at least one real simulator task
4. the result is stable over >= 3 seeds
5. metadata direction control is measurably better than chance

---

## Stage 2 — Server / paper-level simulation

Goal:

```text
scale, establish robustness, and compare against strong independent baselines
```

Tasks:

- MetaWorld multi-task sweep
- robosuite Panda benchmark
- stronger policies: Diffusion Policy and/or ACT
- >= 5 final seeds
- true baseline reproductions/adaptations
- latent dimension sweep
- context supervision ablations
- set-vs-additive ablation
- clustering/direction ablations
- greedy/beam/exact comparison
- count-budget and heterogeneous-cost experiments
- held-out checkpoint generalization
- predicted-vs-realized acquisition calibration

Optional:

- LIBERO
- DMC/MuJoCo generality suite
- static robot datasets for representation/valuation external validation

### Stage 2 exit criterion

LDVA must outperform strong baselines across multiple tasks/environments under equal acquisition cost while maintaining calibrated prospective predictions.

---

## Stage 3 — Paid teleop data + real robot

Goal:

```text
demonstrate real acquisition efficiency, not only simulation sample efficiency
```

Preferred data:

```text
paid/custom teleoperation demonstrations
same or compatible embodiment as evaluation robot
clear acquisition metadata
known monetary cost
```

Acquisition variables can include:

```text
task
object configuration
initial state
goal
scene
clutter
difficulty
operator/source
camera configuration
```

Budget:

```text
sum_a c_a n_a <= C
```

Train imitation policy:

```text
ACT and/or Diffusion Policy
```

Evaluate on a fixed real-robot test distribution declared before acquisition.

Primary figure:

```text
real-robot success rate vs monetary acquisition cost
```

Secondary figures:

```text
success vs acquired trajectories
cost to reach target success
predicted vs realized gain
budget allocation across conditions
```

---

# 15. Immediate Code Audit Fixes — Must Do Before Stage 1 Experiments

These are the current highest-priority tasks.

## P0.1 — Add real rollout evaluation

Current BC evaluation is useful for supervision but is not robot success.

Add an environment-level API:

```python
evaluate_policy(policy, eval_conditions)
```

For DMC report environment return.

For MetaWorld report actual rollout success and return.

Keep fixed evaluation conditions across all methods and rounds.

---

## P0.2 — Generalize closed-loop acquisition to EnvAdapter

Current synthetic closed-loop logic must be refactored so the same loop runs on:

```text
synthetic
DMC
MetaWorld
future robosuite / real robot adapters
```

No synthetic-only data-generation call may remain in the core acquisition loop.

---

## P0.3 — Separate ablations from true baselines

Existing `CUPID-style`, `DataMIL-style`, `Re-Mix-style`, gradient-style routines that obtain scores from the LDVA latent model are **LDVA ablations**.

Implement independent local baselines first:

```text
Direct Gradient Alignment
Direct Influence
```

Then add paper-level external methods.

---

## P0.4 — Fix policy-context indexing and remove checkpoint-ID dependence

Ensure the reference checkpoint's continuous features and checkpoint index cannot silently refer to different checkpoints.

For the main model:

```text
disable checkpoint-ID embedding
```

unless explicitly running the checkpoint-ID ablation.

Evaluate on held-out checkpoints.

---

## P1 — Experiment plumbing

Before large sweeps:

- make experiment scripts load YAML configs rather than duplicate hard-coded defaults
- cache collected datasets
- cache context records and expensive leave-one-out labels
- store git SHA with every run
- use structured JSON/CSV/W&B logging
- add multi-seed launch and aggregation
- pin simulator environments separately from the LDVA core environment
- add CI/test command if practical

---

# 16. Current Repository Status

Already implemented:

```text
sample/context schemas
multi-context supervision
gradient / influence / leave-one-out estimators
MLP/GRU/Transformer sample encoders
DeepSets context encoder
contextual effect readout
set/additive/pairwise utility models
latent diagnostics
KMeans/GMM/HDBSCAN
PCA direction generation
metadata mapper
exact / greedy / beam allocation
count and monetary budgets
synthetic preflight
DMC adapter
MetaWorld adapter
unit tests
```

Not yet paper-ready:

```text
real rollout utility in the acquisition objective/evaluation
EnvAdapter-based closed-loop simulation
independent published baselines
held-out-policy generalization without checkpoint-ID memorization
server-scale experiments
paid data
real robot
```

PushT and ManiSkill are optional at this point; they are not blockers for Stage 1.

---

# 17. First Experiment Sequence

Run in this order after the P0 fixes.

## E0 — Unit + synthetic preflight

```text
pytest
full synthetic gate
>= 3 seeds for measurement
```

Do not quote quick-mode results.

## E1 — DMC Reacher closed-loop

Methods:

```text
Random
Diversity
Direct GradAlign
Direct Influence
LDVA Greedy
LDVA Beam
```

Measure:

```text
rollout return vs acquired episodes
predicted vs realized gain
metadata-direction control
```

## E2 — MetaWorld push-v3

Same methods.

Measure:

```text
actual success rate
return
performance vs acquisition budget
```

If E2 has no stable signal, stop and diagnose before scaling.

## E3 — MetaWorld 5-task sweep

Only after E2 succeeds.

---

# 18. Main Evaluation Metrics

## Representation

```text
contextual effect MSE / rank correlation
latent-distance vs effect-distance correlation
nearest-neighbor effect consistency
held-out-checkpoint performance
```

## Set utility

```text
batch-gain MSE / Spearman
within-checkpoint residual prediction
set vs additive prediction gap
```

## Prospective acquisition

```text
predicted vs realized acquisition gain
candidate-composition rank correlation
normalized regret of chosen composition
```

## Robotics outcome

```text
rollout success rate
rollout return
performance vs acquired episodes
performance vs monetary acquisition cost
```

## Metadata control

```text
desired-vs-realized latent direction cosine
latent displacement error
actionable-direction survival rate
```

---

# 19. Critical Failure Modes

## F1 — No stable effect geometry

If nearby latents do not share contextual effect behavior, clustering-based acquisition is not justified.

## F2 — Batch utility is effectively additive

If a cross-validated additive model matches the set model, the interaction story is weak.

## F3 — Effect labels are estimator noise

Calibrate cheap proxies against expensive leave-one-out labels on a subset.

## F4 — Latent directions are not actionable

If feasible metadata changes cannot reproduce useful latent directions, directional acquisition is not executable.

## F5 — Out-of-support predictions are unreliable

Use local trust regions and Monte Carlo uncertainty; do not extrapolate arbitrarily far.

## F6 — BC utility improves but robot performance does not

This is now a first-class failure mode.

If acquired data reduces imitation loss but fails to improve rollout success/return, the method has not yet demonstrated robot-acquisition value.

## F7 — Policy context does not generalize

If performance collapses on held-out checkpoints, the policy-conditioned representation is memorizing training checkpoints rather than modeling policy state.

---

# 20. Main Ablations

Required:

1. scalar sample score vs contextual latent effect
2. single-context vs multi-context supervision
3. additive vs set-level utility
4. no metric regularization
5. no policy context
6. checkpoint-ID vs continuous policy context
7. no clustering / local neighborhood alternative
8. random directions vs PCA directions
9. no actionability filter
10. local-only exploitation vs outward expansion
11. greedy vs beam vs exact
12. latent dimension sweep
13. BC-loss prediction vs downstream rollout correlation

---

# 21. Reproducibility

Development:

```text
>= 3 seeds
```

Final simulation:

```text
>= 5 seeds
```

Record separately:

```text
environment seed
policy seed
context-generation seed
latent-model seed
acquisition-search seed
```

Cache expensive labels and acquisition datasets.

Every reported run must store:

```text
config
git commit SHA
seed bundle
policy checkpoint(s)
data-model checkpoint
acquisition plan
realized collected metadata
metrics
```

---

# 22. Scope Control

Do not prematurely expand into:

```text
generic world models
joint acquisition-utilization optimization
full active RL
VLA foundation-model pretraining
cross-embodiment transfer
arbitrary real-world scene generation
```

The first paper should remain centered on:

```text
learning optimization-effect geometry
predicting joint future-data utility
mapping promising directions to controllable acquisition conditions
allocating a limited robot-data budget
```

If these four claims are strong, more complex policies and embodiments are scaling experiments, not changes to the core method.
