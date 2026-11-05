#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false USE_TORCH=0 USE_FLAX=1
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 WANDB_PROJECT=paper-reproduction
export LIBERO_CONFIG_PATH=/mnt/data/thanhpv/ldva/scratchpad/libero_config
export MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-datamil
export PYTHONPATH=/mnt/data/thanhpv/ldva/external/datamil:/mnt/data/thanhpv/ldva/external/LIBERO
DATA_PY=/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python
AUTH=/mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py
ERROR_LOG=/mnt/data/thanhpv/ldva/logs/reproduction_errors.txt
trap 'printf "[%s] DataMIL failed line %s: %s\n" "$(date -Is)" "$LINENO" "$BASH_COMMAND" >> "$ERROR_LOG"' ERR
# An actual completed 5-rollout result is required, not just successful import.
"$DATA_PY" -c 'import json,pathlib; root=pathlib.Path("logs/datamil_gpu_smoke"); files=list(root.glob("seed0/*/progress.json")); assert len(files)==1; r=json.loads(files[0].read_text()); assert r["complete"] and r["completed_rollouts"]==5; pathlib.Path("logs/overnight/datamil_smoke_passed.json").write_text(json.dumps(r))'
cd external/datamil
"$DATA_PY" -u "$AUTH" datamil/eval_libero_parallel.py \
  --task_name book-caddy \
  --ckpt /mnt/data/thanhpv/datamil_data/exp/policy_learning/book-caddy/octo_finetune \
  --out_dir /mnt/data/thanhpv/ldva/logs/datamil_book_caddy_eval \
  --num_envs 5 --total_evals 50 --seeds 0 1 2 3 4 \
  > /mnt/data/thanhpv/ldva/logs/overnight/datamil_evaluation.log 2>&1
