# MetaWorld diagnosis — 2026-11-04

Read-only diagnosis of completed runs. No training/config/data changed; DataMIL continues separately. CPU-only probes loaded trusted local checkpoints with CUDA tensors mapped to CPU.

## Confirmed observations

- All 60 MetaWorld jobs completed successfully; all 10 seeds per method/regime are available.
- Final AMF matched task means: cup_left .246, cup_right .202, faucet_open .730, faucet_close .808.
- Final AMF mismatch task means: cup_left .344, cup_right .254, faucet_open .714, faucet_close .688.
- Mismatch pretraining (across 10 seeds) cup_left .648 and cup_right .584, versus .344/.254 after AMF fine-tuning. This directly demonstrates forgetting of pretraining cup tasks, despite adaptive prior.
- Loaded actual mismatch-AMF checkpoints for seeds 0, 4, 9: epoch219, MLP two hidden layers of256 with LayerNorm, batch256, pretrain200, paper adaptive mixture enabled, exactly60000 fine-tuning steps. Thus these inspected runs are not silently using the old architecture/budget.

## Data quality defect in released collection script

`external/amf/scripts/collect_metaworld.py:122` accepts `rewards.sum() >= 0`. Rewards are nonnegative binary success signals, so failed trajectories are accepted as well. Script is unchanged from release; this is not a newly introduced patch. It does not prove which trajectories the paper authors actually used.

Current HDF5 files each contain40 demonstrations of60 steps. Successful trajectories (any reward>0):

|Task|Successes|Failures|
|---|---:|---:|
|cup_left|38|2|
|cup_right|28|12|
|faucet_open|25|15|
|faucet_close|29|11|

Total40/160 trajectories never succeeded within collection horizon. All checked observations/actions finite. Dataset loader reconstructs terminal flags itself, so collector's all-one `dones` field is NOT evidence of broken trajectory splitting in the actual training loader.

## Environment sanity check (CPU)

Executed released scripted expert without collection action noise on20 fixed seeds10000..10019 per task. No dataset writes.

|Task|Success by step60|Success by step100|
|---|---:|---:|
|cup_left|1.00|1.00|
|cup_right|1.00|1.00|
|faucet_open|.75|1.00|
|faucet_close|.80|1.00|

This argues against a completely broken task/reward/environment; it does not establish exact physics equivalence with the paper environment. Current collector adds Gaussian action noise std.15, unlike this clean expert sanity check.

## Frozen-checkpoint ablation (CPU)

Checkpoint: mismatch AMF seed0, epoch219. Identical10 seeds10000..10009 per task,100 steps. No optimization or model edits.

|Task|Pretraining prior|Final raw policy|Final EMA policy|Adaptive mixture|
|---|---:|---:|---:|---:|
|cup_left|.90|.20|.20|.20|
|cup_right|.40|.30|.00|.30|

Stored alpha values: cup_left .57846, cup_right .55543, faucet_open .98484, faucet_close .99507. Adaptive blending fails to preserve cup performance in this checkpoint. This is an observed algorithmic failure, not yet proof of an erroneous alpha-gradient implementation. BC proxy loss can prefer a policy whose closed-loop return is worse.

## Acquisition observations

- Actual acquired demo means (excluding one initial warm-start per task), mismatch AMF: [5.5,5.2,4.6,4.7] for [cup_left,cup_right,faucet_open,faucet_close]. More acquisition is allocated to pretraining-covered cups (10.7) than to initially unseen faucets (9.3). This differs from the intended paper behavior; it does not alone prove a coding bug.
- On seed0 final checkpoint, selector CPU probe found24 separate trajectories, embeddings6x1024 per trajectory and finite, nonzero importance weights. No collapse to a single trajectory, all-zero weights or NaNs was observed. One sampled selector call chose cup_left.
- Importance-weight task means in that probe: [.95709,.96250,.93371,.97342]. Means alone do not describe trajectory-level discrimination.

## Conclusions and remaining checks

Confirmed problems: unfiltered failed collection demos; substantial cup-task forgetting; adaptive prior does not prevent it; acquisition allocation is not reproducing expected behavior. No single verified implementation bug yet explains the full14–26 percentage-point gap.

Priority next experiments, only after requesting implementation/rerun authority: paired-seed clean-successful-data comparison with identical paper budget; controlled adaptive-prior/EMA ablations; per-round acquisition and importance-weight validation against the paper equations. Keep original60-run outputs as archived evidence and do not overwrite them or change several factors simultaneously.

Separate Robomimic caveat: beta.001 matches the released default.yaml comment, but conflicts with paper Appendix M's .01 for non-Kitchen suites. Earlier description as an unexplained choice should be qualified: it was inherited from release; still not paper-matched.
