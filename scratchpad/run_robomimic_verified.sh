#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva/external/amf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-amf
export PYTHONPATH=/mnt/data/thanhpv/ldva/external/amf
/mnt/data/thanhpv/miniconda3/envs/amf-convert39/bin/python -u \
  /mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py \
  /mnt/data/thanhpv/ldva/scratchpad/run_robomimic_protocol.py \
  > /mnt/data/thanhpv/ldva/logs/overnight/robomimic_launcher.log 2>&1
