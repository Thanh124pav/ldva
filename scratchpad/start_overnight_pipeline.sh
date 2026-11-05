#!/usr/bin/env bash
# Clean-start entrypoint. Do not run a second copy over the active overnight run.
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva
mkdir -p logs/overnight
exec 9>logs/overnight/pipeline.lock
flock -n 9 || { echo 'Pipeline launcher already active'; exit 1; }
PY=/mnt/data/thanhpv/miniconda3/envs/amf/bin/python
"$PY" -c 'import psutil; scripts={"reproduce_nn_batch.py","scripts/cotraining.py","datamil/eval_libero_parallel.py","reproduce_gp.py"}; jobs=[p.pid for p in psutil.process_iter(["cmdline"]) if scripts.intersection(p.info["cmdline"] or [])]; assert not jobs, f"Existing reproduction jobs {jobs}; attach to logs instead of duplicating them"'

# Review/smoke gates already have durable numerical output. Rerun small train
# smoke; the other launchers validate their own checkpoint/results gates.
bash scratchpad/run_datamil_training_smoke.sh
declare -a PIPELINE_PIDS=()
for SCRIPT in run_reproductions.sh run_datamil_paper_h15.sh run_gp_paper.sh run_robomimic_verified.sh; do
  bash "scratchpad/$SCRIPT" > "logs/overnight/${SCRIPT%.sh}_launcher.log" 2>&1 &
  PIPELINE_PIDS+=("$!")
done
"$PY" -u scratchpad/write_reproduction_report.py --watch > logs/overnight/report_watcher.log 2>&1 &
REPORTER_PID=$!
PIPELINE_EXIT=0
for PIPELINE_PID in "${PIPELINE_PIDS[@]}"; do
  if ! wait "$PIPELINE_PID"; then PIPELINE_EXIT=1; fi
done
"$PY" scratchpad/write_reproduction_report.py
kill "$REPORTER_PID" 2>/dev/null || true
exit "$PIPELINE_EXIT"
