# SETUP.md — LDVA Development Setup

## 1. Project

**LDVA — Latent Data Valuation for Budgeted Robot Acquisition**

Repository name:

```text
ldva
```

Goal:

```text
Learn an effect-aware latent representation of robot data,
predict set-level training utility,
and use that model to guide future budgeted data acquisition.
```

---

## 2. Recommended Base Code

Use **CUPID** as the primary implementation skeleton because it already contains:

```text
policy training
-> rollout generation
-> influence estimation
-> trajectory scoring
-> retraining / evaluation
```

LDVA should replace scalar valuation with:

```text
sample
-> effect latent
-> contextual/set-level utility
-> latent clustering
-> directional acquisition
-> budget allocation
```

Keep these as reference/baseline repositories:

```text
CUPID
DataMIL
DemInf
Re-Mix
```

Do not merge all projects into one environment.

---

## 3. Environment Strategy

Use separate environments:

```text
.venv-ldva
.venv-cupid
.venv-datamil
```

Main LDVA dependencies should stay minimal.

Recommended progression:

```text
Synthetic
-> PushT low-dimensional
-> MetaWorld
-> ManiSkill
-> optional large offline datasets
```

---

## 4. Stage 0 — Synthetic Sanity Test

Before robotics, create a synthetic benchmark with known:

```text
latent effect factors
batch redundancy
batch complementarity
metadata -> latent mapping
optimal acquisition allocation
```

Example:

```text
metadata m in R^2
true latent z = f(m)
utility U(B) =
    additive gain
    - redundancy
    + pairwise complementarity
```

Verify:

```text
latent learning
set-utility prediction
clustering
direction generation
exact/beam allocation
metadata -> latent control
```

Do not move to robotics if this fails.

---

## 5. Stage 1 — PushT Low-Dimensional

Use PushT first for debugging.

Why:

```text
small state space
fast iteration
clear controllable metadata
easy visualization
compatible with CUPID-style pipelines
```

Suggested metadata:

```text
block x/y
block orientation
agent initial x/y
goal configuration
difficulty / perturbation magnitude
```

Default sample unit:

```text
trajectory chunk
```

Test chunk lengths:

```text
8
16
32
```

---

## 6. Stage 2 — MetaWorld

Use MetaWorld as the first serious benchmark.

Suggested initial tasks:

```text
reach-v2
push-v2
pick-place-v2
drawer-open-v2
button-press-v2
```

Suggested metadata:

```text
task ID
object initial position
goal position
robot initial state
object orientation
task variation
difficulty proxy
policy/source checkpoint
```

Recommended initial scale:

```text
5 tasks
3-5 seeds
multiple acquisition rounds
```

Do not start with MT50.

---

## 7. Stage 3 — ManiSkill

Use ManiSkill after the method is stable.

Suggested tasks:

```text
PushCube
PickCube
StackCube
PegInsertionSide
PickSingleYCB
```

Advantages:

```text
controllable simulator states
more realistic manipulation
scalable data generation
good fit for metadata-conditioned acquisition
```

Use state observations first.

Move to RGB only after the acquisition pipeline works.

---

## 8. Static Offline Datasets

Static datasets are useful for:

```text
representation pretraining
valuation sanity checks
offline baselines
external validation
```

They are not sufficient for the main future-acquisition experiment.

### RoboMimic

Use as a small-scale validation dataset only.

Pros:

```text
clean manipulation tasks
multiple quality regimes
human demonstrations
easy policy-learning baselines
```

Limitations:

```text
limited scale
restricted acquisition intervention
cannot actively expand support
```

Do not use RoboMimic as the main LDVA benchmark.

### Larger optional datasets

Later:

```text
BridgeData V2
Open X-Embodiment
DROID
LIBERO datasets
```

Use them for scaling/transfer, not as the first closed-loop benchmark.

---

## 9. Policy Learning

Start simple.

Recommended order:

```text
MLP BC
-> PPO / SAC
-> diffusion policy if needed
```

Suggested default:

```text
MetaWorld: PPO or SAC
PushT: low-dimensional diffusion policy or MLP
ManiSkill: PPO or SAC
```

Avoid vision/VLA policies early because effect-label generation is already expensive.

---

## 10. Sample Definition

Candidate units:

```text
transition
trajectory chunk
full trajectory
```

Default:

```text
trajectory chunk
```

Each sample should store:

```text
sample_id
trajectory_id
start_t
end_t
observation chunk
action chunk
reward/success
policy checkpoint ID
task ID
acquisition metadata
```

---

## 11. Multi-Context Supervision

Each sample must appear in many optimization contexts.

Store:

```text
ContextRecord:
    policy_checkpoint_id
    batch_sample_ids
    per_sample_effect_targets
    batch_gain_target
    utilization_rule_id
```

Recommended:

```text
20-100 contexts/sample
```

Monitor:

```text
contexts/sample histogram
batch co-occurrence
policy-checkpoint diversity
```

Avoid repeated near-identical batch compositions.

---

## 12. Effect Labels

Keep the interface modular.

```python
class EffectEstimator:
    def sample_effect(...):
        ...

    def batch_gain(...):
        ...
```

Cheap targets:

```text
gradient alignment
gradient norm
validation-gradient alignment
```

Medium:

```text
first-order influence
TRAK-like influence
```

Expensive/oracle:

```text
leave-one-out local update
short-horizon retraining difference
```

Use expensive targets on a subset for calibration.

---

## 13. Core Models

Implement:

```text
SampleEncoder
ContextEncoder
EffectReadout
BatchUtilityModel
```

MVP:

```text
SampleEncoder: MLP / GRU
ContextEncoder: DeepSets
EffectReadout: MLP
BatchUtilityModel: DeepSets + MLP
```

Later:

```text
Set Transformer
trajectory Transformer
JEPA-style objectives
```

Do not start with VAE reconstruction unless experiments justify it.

---

## 14. Latent Geometry

Sweep latent dimension:

```text
8
16
32
64
128
```

Track:

```text
effect prediction
batch utility prediction
nearest-neighbor effect consistency
cluster stability
metadata-direction predictability
```

Do not assume larger latent is automatically better.

---

## 15. Clustering

Start with:

```text
KMeans
```

Then test:

```text
GMM
HDBSCAN
```

Each cluster should expose:

```text
member IDs
centroid
covariance
local PCA basis
boundary samples
metadata statistics
```

---

## 16. Direction Generation

For each cluster:

```text
compute covariance
compute PCA
retain directions explaining rho = 0.90 variance
cap intrinsic dimensions at r_max = 5
consider +/- directions
```

Filter by:

```text
outward movement
support-density decrease
trust-region distance
metadata actionability
```

Do not allow uncontrolled long-range extrapolation.

---

## 17. Metadata-to-Latent Direction Model

Store controllable simulator metadata explicitly.

Fit local:

```text
Delta metadata -> Delta latent
```

MVP:

```text
local linear regression
```

Approximation:

```text
Delta z ~= J_k Delta m
```

For desired latent direction `v` solve:

```text
Delta m*
=
argmin ||J_k Delta m - alpha v||^2
```

subject to simulator constraints.

Evaluate:

```text
cosine similarity(
    desired latent direction,
    realized latent movement
)
```

---

## 18. Acquisition Utility Prediction

Let candidate directions be:

```text
a_1, ..., a_A
```

An allocation is:

```text
n = (n_1, ..., n_A)
sum n_a = B
```

Generate future hypothetical latents:

```text
z_new =
boundary anchor
+ delta * direction
+ local noise
```

For each allocation:

```text
sample multiple hypothetical batches
run BatchUtilityModel
average predictions
```

Estimate:

```text
V_hat(n)
```

This must score the **whole acquisition composition**, not independent directions.

---

## 19. Acquisition Solvers

Implement three.

### Exact Search

Use for small `A` and `B`.

Purpose:

```text
oracle optimum under the learned utility model
```

### Greedy

At each step:

```text
allocate one unit to the direction
with largest one-step marginal gain
```

Use as a baseline.

### Beam Search

Primary practical planner.

Initial beam widths:

```text
5
10
20
50
```

Compare beam search against exact search on small problems.

---

## 20. Baselines

Minimum acquisition baseline suite:

```text
Random / Uniform
Equal allocation
Diversity / Core-set
Uncertainty
Gradient norm
Gradient alignment
CUPID-style influence
DemInf-style scoring
DataMIL-style utility prediction
Re-Mix-style domain mixture
LDVA Greedy
LDVA Beam
```

External baselines do not all need to share one environment.

If exact reproduction is impossible, reimplement the algorithmic principle and document deviations.

---

## 21. Evaluation

Separate three questions.

### A. Valuation quality

```text
Can LDVA predict contextual sample effects?
```

### B. Set utility quality

```text
Can LDVA rank candidate future batches?
```

### C. Acquisition quality

```text
Under the same acquisition budget,
does LDVA produce a better final policy?
```

Do not collapse these into one metric.

---

## 22. Metrics

Representation:

```text
sample-effect MSE
sample-effect rank correlation
batch-utility MSE
batch-utility rank correlation
neighbor consistency
cluster stability
```

Acquisition:

```text
predicted vs realized gain
candidate-batch rank correlation
policy success vs acquisition budget
return vs acquisition budget
collection cost to target performance
```

Metadata control:

```text
desired-vs-realized direction cosine
latent displacement error
metadata intervention success rate
```

---

## 23. Logging

Use Weights & Biases.

Log:

```text
policy metrics
effect-label statistics
batch-gain targets
latent norms
PCA spectrum
cluster sizes
direction counts
allocation predictions
realized acquisition gain
beam candidates
metadata-to-latent alignment
```

Save:

```text
policy checkpoints
data-model checkpoints
latent embeddings
cluster assignments
context datasets
acquisition plans
```

---

## 24. Seeds / Reproducibility

Development:

```text
>= 3 seeds
```

Final:

```text
>= 5 seeds
```

Record independently:

```text
environment seed
policy seed
context-generation seed
latent-model seed
acquisition-search seed
```

---

## 25. Compute Strategy

Parallelize:

```text
rollout collection
context construction
effect-label generation
policy seeds
```

Cache all expensive effect labels.

Do not recompute leave-one-out/influence targets unless necessary.

---

## 26. First Experiment

Environment:

```text
PushT low-dimensional
```

Suggested data:

```text
5k-20k trajectory chunks
multiple policy checkpoints
20+ contexts/sample
```

Models:

```text
MLP/GRU encoder
DeepSets context encoder
DeepSets utility model
latent dim = 32
```

Validate:

```text
contextual effect prediction
batch utility prediction
nearest-neighbor consistency
exact allocation on small candidate sets
```

Do not run full closed-loop acquisition until these work.

---

## 27. First Serious Experiment

Environment:

```text
MetaWorld
```

Tasks:

```text
5 manipulation tasks
```

Loop:

```text
initial collection
-> train policy
-> generate multi-context supervision
-> train LDVA
-> cluster latent space
-> generate local directions
-> predict acquisition compositions
-> solve budget allocation
-> map directions to metadata
-> collect new data
-> retrain
-> evaluate
```

Compare:

```text
Random
Diversity
Uncertainty
Gradient Alignment
CUPID-style
Re-Mix-style
LDVA Greedy
LDVA Beam
```

---

## 28. Suggested Repository Structure

```text
ldva/
|
|-- README.md
|-- PLAN.md
|-- SETUP.md
|
|-- configs/
|   |-- env/
|   |-- policy/
|   |-- datamodel/
|   |-- acquisition/
|
|-- envs/
|   |-- pusht/
|   |-- metaworld/
|   |-- maniskill/
|
|-- data/
|   |-- samples.py
|   |-- context_dataset.py
|   |-- metadata.py
|
|-- policy/
|   |-- train.py
|   |-- evaluate.py
|   |-- checkpoints.py
|
|-- supervision/
|   |-- base.py
|   |-- gradient_alignment.py
|   |-- influence.py
|   |-- leave_one_out.py
|
|-- models/
|   |-- sample_encoder.py
|   |-- context_encoder.py
|   |-- effect_readout.py
|   |-- batch_utility.py
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
|   |-- plotting.py
|
|-- experiments/
|   |-- synthetic/
|   |-- pusht/
|   |-- metaworld/
|   |-- maniskill/
|
|-- tests/
```

---

## 29. Immediate Coding Order

Implement in this order:

```text
1. synthetic acquisition test
2. context supervision format
3. gradient-alignment effect estimator
4. sample encoder
5. DeepSets context encoder
6. contextual effect readout
7. batch utility predictor
8. latent diagnostics
9. KMeans clustering
10. PCA direction generation
11. exact allocation solver
12. beam-search solver
13. PushT integration
14. metadata mapper
15. closed-loop acquisition
16. MetaWorld scale-up
```

Do not begin with:

```text
vision
VLA
Open X
large-scale diffusion-policy training
joint acquisition-utilization optimization
```

---

## 30. Success Criteria

Proceed only if:

```text
1. contextual effect prediction beats scalar/constant baselines
2. non-additive set utility beats additive utility prediction
3. latent neighbors have similar effect profiles
4. beam search approximately matches exact search on small problems
5. predicted high-value acquisition batches realize higher gain
6. metadata interventions move samples approximately along intended latent directions
```

---


## 31. Commercial Robot Data Track

Use at least one paid/commercial acquisition setting for the final paper if access is available.

Purpose:

```text
demonstrate that LDVA improves real acquisition efficiency,
not only simulation sample efficiency
```

Preferred sources:

```text
Inverted Lambda
Telemanual
ipto.ai
custom teleoperation / robot-data vendors
```

The exact provider is not part of the method.

The requirement is that the provider exposes sufficiently clear acquisition conditions and costs.

Examples of acquisition conditions:

```text
task
object configuration
initial state
scene
difficulty
operator/source
robot embodiment
camera setup
trajectory type
```

Each acquisition option should have a measurable cost:

```text
c_a = monetary cost per trajectory / episode / collection request
```

The budget constraint becomes:

```text
sum_a c_a * n_a <= C
```

instead of only:

```text
sum_a n_a <= B
```

The planner should maximize predicted downstream gain under the monetary budget:

```text
Q* =
argmax_Q V_hat(Q | D, theta)

subject to:
Cost(Q) <= C
```

Report both:

```text
performance vs number of acquired trajectories
performance vs monetary acquisition cost
```

---

## 32. Commercial Data Training Setup

Commercial robot data should preferably be:

```text
teleoperated demonstrations
same embodiment as evaluation robot
same/similar observation interface
same action representation
compatible task family
```

Preferred format:

```text
LeRobot-compatible data
or
a format that can be converted losslessly to the LDVA dataset schema
```

Avoid using a vendor dataset collected on an unrelated embodiment unless cross-embodiment transfer is itself part of the experiment.

That would introduce an additional research problem and weaken the acquisition story.

### Recommended first policy

Use imitation learning.

Preferred order:

```text
ACT
-> Diffusion Policy
-> larger policy / VLA only if necessary
```

ACT is the preferred first real-robot baseline because it is relatively simple, data-efficient, and compatible with teleoperation demonstrations.

The real-data loop should be:

```text
commercial / teleop data D_0
-> train policy pi_0
-> train/update LDVA
-> plan next acquisition Q*
-> purchase / collect D_new
-> D_1 = D_0 union D_new
-> retrain policy pi_1
-> evaluate on fixed real-robot test conditions
```

---

## 33. Real Robot Arm Evaluation

Yes: use a real robot arm for the final commercial-data validation.

Preferred setup:

```text
same robot embodiment for:
    data collection
    policy training interface
    final evaluation
```

Possible examples:

```text
SO-100 / SO-101
Franka
UR5 / UR5e
other arm supported by the available teleoperation pipeline
```

Choose the robot that matches the acquired data.

Do not pick the robot first and then force incompatible commercial data into the experiment.

### Fixed evaluation protocol

Define the evaluation distribution before acquisition.

Example controlled factors:

```text
object initial position
goal position
object orientation
arm initial state
scene clutter
lighting/background
task difficulty
```

Use a fixed held-out grid or sampled distribution.

Do not change the test distribution after observing which data LDVA acquires.

Primary metric:

```text
success rate
```

Secondary metrics:

```text
return / task reward
completion time
failure type
robustness across evaluation conditions
```

Recommended number of real-robot rollouts:

```text
>= 50 per major condition group when practical
```

For expensive experiments, use a smaller pilot first and pre-register the final rollout budget.

---

## 34. Real Acquisition Baselines

For the paid-data experiment, compare at equal monetary cost.

Minimum:

```text
Random acquisition
Equal allocation across acquisition conditions
Diversity / coverage
Uncertainty-based acquisition
Influence / gradient-based selection or allocation
LDVA Greedy
LDVA Beam
```

If applicable:

```text
CUPID-style policy-aware influence
Re-Mix-style source/domain allocation
DataMIL-style predicted utility
```

The comparison must use the same:

```text
initial dataset
policy architecture
training budget
evaluation distribution
monetary acquisition budget
```

---

## 35. Commercial-Data Impact Figures

The main figure should be:

```text
real-robot success rate vs monetary acquisition cost
```

Recommended additional plots:

```text
success rate vs number of newly acquired trajectories
cost required to reach a target success rate
predicted gain vs realized real-robot gain
budget allocation across acquisition conditions
```

Example paper-level claim format:

```text
LDVA reaches a fixed success target with less acquisition cost
```

or:

```text
under the same monetary budget, LDVA yields higher real-robot success
```

Do not report monetary savings unless the provider prices and costs are documented.

---

## 36. Commercial-Data Metadata Requirements

For every paid trajectory / episode, store:

```text
provider/source
robot embodiment
task
scene
initial condition
goal condition
operator/source if available
camera configuration
collection timestamp or batch
acquisition price / cost
quality-control flags
policy or teleoperator source
```

The metadata is required because LDVA must eventually map:

```text
desired latent expansion
-> actionable acquisition condition
```

The commercial-data experiment is not useful if the acquired dataset arrives as an opaque unstructured corpus with no controllable acquisition dimensions.

---

## 37. Updated Benchmark Ladder

Use the following progression:

```text
Stage 0:
Synthetic
    -> verify optimization-effect latent and allocation logic

Stage 1:
PushT low-dimensional
    -> debug multi-context supervision and set utility

Stage 2:
MetaWorld
    -> first serious controllable acquisition benchmark

Stage 3:
ManiSkill
    -> larger and more realistic simulation validation

Stage 4:
Static robot datasets
    -> representation / valuation external validation
    -> RoboMimic, BridgeData V2, DROID, Open X, LIBERO

Stage 5:
Paid/commercial robot data
    -> cost-aware acquisition

Stage 6:
Real robot arm
    -> final downstream policy evaluation
```

The strongest complete paper should ideally contain:

```text
large controlled simulation
+
real paid-data acquisition
+
real-robot evaluation
```

---

## 38. Updated First-Paper Scope

The first paper should focus on:

```text
effect-aware latent valuation
set-level utility prediction
directional future acquisition
budgeted allocation
cost-aware acquisition
real-robot validation when feasible
```

Do not expand prematurely into:

```text
generic world modeling
full active RL
joint curriculum learning
robot foundation-model pretraining
arbitrary cross-embodiment transfer
```

Core story:

```text
LDVA learns how robot data affects optimization,
uses the learned geometry to identify promising future acquisition directions,
allocates a limited collection budget to the best joint composition,
and validates acquisition efficiency under real collection cost.
```


The first paper should focus on:

```text
effect-aware latent valuation
set-level utility prediction
directional future acquisition
budgeted allocation
```

Do not expand prematurely into:

```text
generic world modeling
full active RL
joint curriculum learning
robot foundation-model pretraining
arbitrary real-world acquisition planning
```

Core story:

```text
LDVA learns how robot data affects optimization,
uses the learned geometry to identify promising future acquisition directions,
and allocates a limited collection budget to the best joint composition.
```
