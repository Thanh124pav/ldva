# MetaWorld data fix and validation

Validation completed 2026-11-04 16:19 UTC. DataMIL kept running throughout validation.

## Full seed launched after user approval

At 2026-11-04 16:58 UTC, launched one fresh mismatch-AMF seed0 on CPU, two math threads, with successful-only data. Full budget:200 pretraining epochs;20 acquisition rounds x3000 gradient steps; batch256; MLP2x256 with LayerNorm; beta.01;50 evaluation episodes/task. No shortened rollout override. Detached screen `ldva-amf-clean-train-seed0-64012`, launcher `scratchpad/run_amf_clean_seed0.sh`. W&B: https://wandb.ai/hust_edu_vn/paper-reproduction/runs/x988w4ef . Epoch-level atomic checkpoints preserve optimizer and RNG for resume. Status: `logs/amf_metaworld_fix/train_seed0_status.json`; console: `logs/amf_metaworld_fix/train_seed0.log`; outputs: `external/amf/outputs/metaworld_success_v1/amf-metaworld-mismatch-posterior_grad-seed0-cpu`. Initial pretraining metrics and checkpoint verified; no OOM observed. DataMIL was active at trial index12 (thirteenth trial).

## Changes

- `external/amf/scripts/collect_metaworld.py`: reject trajectories without positive success reward and trajectories with nonfinite/misaligned arrays; correct terminal flags; close each MuJoCo environment; deterministic collection seed; explicit attempt budget; exclusive temporary HDF5 creation; separate output root; verified reuse; optional W&B logging.
- Collection retains released60-step horizon and Gaussian action noise std.15. New data root: `external/amf/data/metaworld_success_v1`; original data unchanged. Failed attempts rejected, not used for training. This is a successful-data diagnostic variant, not proof of author-identical paper data.
- `external/amf/dataset.py`: opt-in `require_successful_demos=True` rejects failed/nonfinite rewards at load time. Default behavior unchanged for old experiments and Robomimic.
- No changes to adaptive-prior/acquisition equations: available tests do not establish a formula bug. Do not claim that the full performance discrepancy has been fixed.

## Collection results

|Task|Successful saved|Attempts|
|---|---:|---:|
|cup_left|40|41|
|cup_right|40|59|
|faucet_open|40|61|
|faucet_close|40|53|

Collection seed20261104; task/attempt seeds recorded in each demo. All160 saved demos passed finite-array, shape, success and terminal-flag checks. New RNG/seeds mean data differs beyond simply filtering the existing160 trajectories; a later improved score alone would not isolate filtering as the only causal factor.

## Tests

- `scratchpad/test_amf_metaworld_fixes.py`: 5/5 CPU tests pass: failed/nonfinite/misaligned-demo rejection; real dataset-loader rejection; adaptive-mixture endpoints/interpolation; real Trainer alpha update vs autograd for paper LR1; actual acquisition selection and task permutation on an orthogonal toy problem.
- `scratchpad/check_amf_clean_data.py`: fresh CPU smoke v2 passed train → acquisition → evaluation → checkpoint → W&B. One pretraining epoch, one acquisition round, two fine-tuning steps,2 evaluations/task with16-step rollout limit. NOT a paper experiment; zero smoke success is not a final performance result.
- Guarded data loader enabled in v2 smoke. Runtime10.25seconds excluding imports. Original four datasets' SHA256 hashes unchanged before/after smoke (hashes in JSON evidence).
- `git diff --check` and Python compilation passed for modified/new modules.

## Evidence

- Regression log: `logs/amf_metaworld_fix/regression_tests.log`.
- Structured validation: `logs/amf_metaworld_fix/cpu_smoke_passed.json`.
- Collection manifest: `external/amf/data/metaworld_success_v1/collection_manifest.json`.
- CPU smoke log: `logs/amf_metaworld_fix/cpu_smoke_v2.log`.
- Collection W&B: https://wandb.ai/hust_edu_vn/paper-reproduction/runs/s1woqy5n
- Smoke W&B: https://wandb.ai/hust_edu_vn/paper-reproduction/runs/qgcdmww8

Next useful experiment: one fresh mismatch-AMF seed0 with original paper training budget, success-only data and separate output/W&B identity. Do not resume old checkpoints or relabel old60-run results as fixed. Measure RAM pressure before increasing concurrency; current allocation has64GiB RAM and DataMIL is active.
