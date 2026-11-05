#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /mnt/data/thanhpv/miniconda3/envs/amf/bin/python -u scratchpad/train_amf_clean_seed0.py
