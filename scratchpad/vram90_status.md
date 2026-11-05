# VRAM target adjustment — 2026-11-04

SUPERSEDED: user stopped the DataMIL trajectory/10-trial jobs and this controller.
These measurements are historical. Current paper H15/T30 pipeline is documented
in `scratchpad/datamil_paper_h15_status.md`; do not restart this old controller.

- Target: approximately 90% of the visible H100's VRAM, including normal JAX allocator/cache pools.
- Measured after launching both workers: **72446/81559 MiB = 88.83%**; GPU utilization 92–96% at these samples.
- Real work: DataMIL fresh seeds 1 and 2, each batch 128 and the same 10k-step schedule. No batch-size or learning-rate change was made to fill memory.
- Pool limits: seed 1 = 0.38; seed 2 = 0.40. Peak working allocator demand from the validated batch-128 pilot was ~27.4 GiB per training job.
- Pending training: seeds 3 and 4 reuse a freed worker slot. One evaluator serializes completed seed evaluations, preserving the 50-rollout protocol.
- Original sequential DataMIL launcher PID 3298365 is intentionally SIGSTOP-paused to prevent duplicate seed training. Its already-running seed-0 evaluator continues normally.
- The controller resumes that launcher only after seeds 1–4 have all trained and evaluated; the launcher then finds completed outputs and skips them.
- Other AMF/Robomimic/GP jobs are left running. Their memory use and phase changes can move the measured total above or below 90%; this is a target, not a guarantee of an exact constant percentage.

## Independent execution and logs

- Screen session: `ldva-vram90` (daemon PID 3306356, PPID 1).
- Controller source: `scratchpad/vram90_controller.py`.
- Live GPU/pool/job state: `logs/vram_target_90/status.json` (every 10 seconds).
- Training logs: `logs/vram_target_90/train_seed{1,2,3,4}.log`.
- Evaluation logs: `logs/vram_target_90/eval_seed{1,2,3,4}.log`.
- W&B config records pool fraction, preallocation, and VRAM target separately from scientific hyperparameters.
- Allocation remains Slurm job 63887, ending **2026-11-04 05:51:18 UTC**. This adjustment does not extend its duration.

VRAM occupancy includes reserved allocator memory and is not the same as live-tensor occupancy or GPU compute utilization.
