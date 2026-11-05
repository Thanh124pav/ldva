# Overnight execution audit — 2026-11-03 UTC

UPDATE 2026-11-04: the trajectory/10-trial DataMIL policy jobs described below
were explicitly stopped by the user. Replacement paper H15/T30 settings and
resume commands are in `scratchpad/datamil_paper_h15_status.md`. AMF continues.

All work is on `acquisition`; pre-existing deletions of root README.md / PLAN.md were preserved.

## Gates that actually passed

- AMF MetaWorld: GPU weighted-mixture regression and differentiable holdout regression; real train → acquire demo → finetune → rollout → local checkpoint → W&B. Smoke run: `w0staqmm`.
- DataMIL evaluation: five real Book-Caddy rollouts completed, 3 successes; JSON + W&B artifacts, run `payx4wsy`. This is a smoke sample, NOT a numerical paper comparison.
- DataMIL fresh training: two actual gradient steps at batch 8, finite losses and parameter/full-state checkpoint at step 2. Run `datamil-fresh-training-smoke_20261103_184312`.
- DataMIL production-batch pilot: 20 actual gradient steps, batch 128, finite loss, parameter/full-state checkpoint at step 20. Median steady time 0.2186 s/step; peak allocator 27.37 GiB. Run `datamil-batch128-pilot_20261103_184811`.
- AMF Robomimic: batch-8 diffusion smoke train/acquisition/finetune/evaluation completed, checkpoint epoch 1. Run `g5opfr45`; deliberately shortened rollout/denoising, not comparable paper success.
- GP 100-candidate smoke completed, checkpoint and W&B artifacts, run `4x6pxj4b`.
- Shell syntax and Python compilation checked for new launchers and changed source.

## Faults diagnosed and fixed

1. Earlier GP run kept all results in memory until plotting; missing LaTeX package lost that run's numerical outputs. New runner fsyncs and atomically replaces metrics + NumPy RNG checkpoint after EVERY method/seed, before any plot. Fault-injection/resume was verified bit-identical. The failed first run is not counted as reproduced results.
2. DataMIL simulator + modern Transformers loaded an unrelated CUDA-13 Triton runtime, segfaulting in native module import. `USE_TORCH=0` and `USE_FLAX=1` retain Flax text encoding and avoid that backend. Both real rollout and fresh training now pass.
3. Official LIBERO initial-state files contain NumPy pickle objects; new Torch defaults to weights-only deserialization. Explicit trusted local benchmark loading restores original semantics.
4. TensorFlow preprocessing must remain CPU-only while JAX owns GPU inference/training. Thread limits bound TensorFlow/OpenBLAS usage.
5. AMF task embedding input was CPU while embeddings were CUDA: create IDs on configured device.
6. AMF acquisition accidentally received the adaptive mixture wrapper instead of a differentiable holdout model: preserve the holdout policy path.
7. Vector-env auto-reset cleared recorded wrapper rewards before evaluation retrieved them: retain maxima from returned step rewards. Renderer uses vector-env method dispatch.
8. Short-trajectory pruning could create zero-length batches / NaN task IDs: keep at least one transition.
9. A minibatch consisting entirely of holdout examples has no hold-in update: skip that empty update, rather than propagating NaN or invalid diffusion reshape.
10. Acquisition uses analytical last-layer features and requires no autograd graph: no-grad prevents retaining unnecessary diffusion graphs.
11. Pilot stopping-step override shared a FieldReference with learning-rate decay, invalidating warmup. Reproduction config resolves references, keeping the full 10k-step LR schedule independent of pilot duration, and retains typed CLI placeholders.
12. Robosuite 1.5 composite controllers need arm `input_type=absolute`, not the old root `control_delta` key.

## Scientific differences that must stay visible

- Released GP uses 20 candidates; appendix says 100. Both are running separately, with separate checkpoints and W&B configuration.
- Released AMF neural code differs from appendix in MLP size/layout, fixed gradient budget and adaptive-prior action combination. These are explicit paper-mode flags, not silently changed defaults.
- MetaWorld data came from the released noisy scripted demonstrator, whose collection script does not set a seed. Exact paper trajectories are unavailable.
- Downloadable Robomimic data uses simulator v1.5.1; original AMF pins v1.2. Even a successful full run is a simulator-version-qualified reproduction.
- DataMIL supplied ten datamodel trials; policy training uses their reaggregated top-10% selection (450/4500 trajectories). The full 30-trial selection computation has not been reproduced anew.
- Existing DataMIL checkpoints and NEW training runs are separate. Fresh evaluations carry distinct provenance/group. A model config reports window 2 inherited from pretrained, while its finetune config says window 1; the actual finetuning and released evaluation use window 1. No unsupported change to window 2 was made.
- Kitchen Sdoor is unavailable from the public Google Drive source. A four-task subset cannot be called the paper's five-task Kitchen experiment.
- DataMIL MetaWorld UTBox source times out; real-robot evaluation cannot be done without hardware.

## Current launchers and durable files

Existing run is already active; **do not execute another master launcher over it**.
`start_overnight_pipeline.sh` is a guarded clean-start recipe, not a command to duplicate the current processes.

- MetaWorld queue: `run_reproductions.sh`, 4 workers, 60 full runs.
- Supplied DataMIL checkpoints: `run_datamil_verified.sh`, 5×50 rollouts.
- New DataMIL training: `run_datamil_fresh.sh`, 20-step pilot then 5×10k steps, checkpoint every 1k steps and 50-rollout evaluation after each seed.
- Appendix GP: `run_gp_paper.sh`, CPU, 12 prior supports ×10 seeds ×2 methods; separate from released-notebook run.
- Robomimic: `run_robomimic_verified.sh`, production-batch/100-denoise pilot then one full matched-prior AMF seed, with VRAM gate.
- `write_reproduction_report.py --watch` refreshes Markdown every20 minutes without model calls.
- GPU/host resource samples every15 seconds in `logs/overnight/resources.jsonl` and W&B.
- Metrics and checkpoints are local first, W&B second. New failures stop the affected queue and are logged; the monitor itself does not rewrite arbitrary code.

Main report: `scratchpad/reproduction_overnight_report.md`.
Project: https://wandb.ai/hust_edu_vn/paper-reproduction
