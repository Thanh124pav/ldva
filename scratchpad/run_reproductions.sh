#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva
mkdir -p logs/overnight
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-amf
export WANDB_PROJECT=paper-reproduction
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LIBERO_CONFIG_PATH=/mnt/data/thanhpv/ldva/scratchpad/libero_config
AMF_PY=/mnt/data/thanhpv/miniconda3/envs/amf/bin/python
DATA_PY=/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python
AUTH=/mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py
AMF_WORKERS=${AMF_WORKERS:-4}
ERROR_LOG=/mnt/data/thanhpv/ldva/logs/reproduction_errors.txt
declare -a JOB_PIDS=()
trap 'printf "[%s] launcher failed line %s: %s\n" "$(date -Is)" "$LINENO" "$BASH_COMMAND" >> "$ERROR_LOG"' ERR

# Review and real smoke checks before launching the expensive queue.
(
  cd external/amf
  export PYTHONPATH=/mnt/data/thanhpv/ldva/external/amf:/mnt/data/thanhpv/ldva/external/metaworld-history
  "$AMF_PY" -u "$AUTH" /mnt/data/thanhpv/ldva/scratchpad/review_amf_smoke.py
) > logs/overnight/review_smoke.log 2>> "$ERROR_LOG"

# Four small MLP jobs overlap training and CPU rollout phases. The measured
# single-worker tensor peak is ~0.071 GiB; preserve GPU room for Octo inference.
for ((AMF_WORKER=0; AMF_WORKER<AMF_WORKERS; AMF_WORKER++)); do
  (
    cd external/amf
    export PYTHONPATH=/mnt/data/thanhpv/ldva/external/amf:/mnt/data/thanhpv/ldva/external/metaworld-history
    "$AMF_PY" -u "$AUTH" reproduce_nn_batch.py --worker "$AMF_WORKER" --workers "$AMF_WORKERS"
  ) > "logs/overnight/amf_worker${AMF_WORKER}.log" 2>> "$ERROR_LOG" &
  JOB_PIDS+=("$!")
done

# The DataMIL smoke gate must have succeeded, with durable numeric results.
if [[ -f logs/overnight/datamil_smoke_passed.json ]]; then
  (
    cd external/datamil
    export PYTHONPATH=/mnt/data/thanhpv/ldva/external/datamil:/mnt/data/thanhpv/ldva/external/LIBERO
    "$DATA_PY" -u "$AUTH" datamil/eval_libero_parallel.py \
      --task_name book-caddy \
      --ckpt /mnt/data/thanhpv/datamil_data/exp/policy_learning/book-caddy/octo_finetune \
      --out_dir /mnt/data/thanhpv/ldva/logs/datamil_book_caddy_eval \
      --num_envs 5 --total_evals 50 --seeds 0 1 2 3 4
  ) > logs/overnight/datamil_evaluation.log 2>> "$ERROR_LOG" &
  JOB_PIDS+=("$!")
fi

"$AMF_PY" -u "$AUTH" scratchpad/reproduction_resource_watcher.py \
  > logs/overnight/watcher.log 2>> "$ERROR_LOG" &
WATCHER_PID=$!
printf '%s\n' "${JOB_PIDS[@]}" > logs/overnight/job_pids.txt
RUN_EXIT=0
for JOB_PID in "${JOB_PIDS[@]}"; do
  if ! wait "$JOB_PID"; then RUN_EXIT=1; fi
done
kill "$WATCHER_PID" 2>/dev/null || true
printf '{"finished":true,"exit_code":%d}\n' "$RUN_EXIT" > logs/overnight/launcher_result.json
exit "$RUN_EXIT"
