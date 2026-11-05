#!/usr/bin/env bash
# Resume the horizon-15 / 30-trial / one-policy-seed pipeline. Refuses duplicates.
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva
mkdir -p logs/datamil_paper_h15
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export USE_TORCH=0 USE_FLAX=1
export SCREENDIR=/mnt/data/thanhpv/ldva/logs/screen
exec /mnt/data/thanhpv/miniconda3/envs/datamil/bin/python -u scratchpad/datamil_paper_h15_pipeline.py
