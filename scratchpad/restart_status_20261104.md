# Restart status — 2026-11-04

## Independent execution

- Screen daemon: `3293638.ldva-reproduce-20261104`, PID 3293638, **PPID 1**.
- Supervisor: PID 3293640; parent is the screen daemon, not Codex.
- Screen socket directory: `/mnt/data/thanhpv/ldva/logs/screen` (mode 0700).
- Daemon creation returned successfully; a later, separate terminal confirmed daemon and jobs still alive. This verifies survival of launcher-terminal exit; an actual Codex thread-unload test was not performed on the production jobs.
- Allocation: Slurm job **63887**, scheduled end **2026-11-04 05:51:18 UTC**. Detaching does not extend that allocation or survive its termination.

## Resumed work

- AMF MetaWorld: original four shards; skip four completed experiments, restore epoch/model/optimizer/RNG for incomplete experiments. Same 60-run queue, no change to paper-mode hyperparameters.
- DataMIL existing checkpoints: seed 0's completed 50-rollout result is reused. Partial seed 1 evaluation is restarted from the beginning of its 50-rollout cohort; partial numbers are not added to the rerun's denominator.
- DataMIL fresh training: added explicit full-state restoration. Seed 0 restores **step 4000**, including optimizer and JAX RNG, into the original run directory and W&B identity. The seeded TF input stream is skipped to the corresponding batch position; bitwise dataloader replay has NOT been verified. Steps 4001–4399 from the interrupted run were not checkpointed and must be recomputed.
- GP notebook and 100-candidate variant restore their saved NumPy RNG and completed-method checkpoint; completed methods are skipped. Online mirror resumes its previous W&B identity.
- Robomimic: skip completed production-batch pilot; restore full-run epoch 23 and continue at epoch 24 (200 pretraining epochs, then 20 acquisition rounds).

## Evidence and monitoring

- Supervisor heartbeat / child PID / exit status: `logs/restart_20261104/status.json`, every 15 seconds.
- Append-only supervisor logs: `logs/restart_20261104/*.log`.
- Per-job exit markers: `logs/restart_20261104/*.exit.json`.
- Algorithm checkpoints and metrics remain in their original directories.
- Numerical report: `scratchpad/reproduction_overnight_report.md`, refreshed every 20 minutes.
- W&B project: https://wandb.ai/hust_edu_vn/paper-reproduction
- Errors: `logs/reproduction_errors.txt`, plus the full console log of the affected algorithm.
- The supervisor records failed children; it does not silently rewrite code or restart an indefinitely failing algorithm.

## Inspect without starting duplicates

```bash
SCREENDIR=/mnt/data/thanhpv/ldva/logs/screen screen -ls
cat /mnt/data/thanhpv/ldva/logs/restart_20261104/status.json
```

Do not rerun the master launchers over the active detached jobs.
