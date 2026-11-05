#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva/external/amf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-amf
GP_PY=/mnt/data/thanhpv/miniconda3/envs/acvcl/bin/python
AUTH=/mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py
trap 'printf "[%s] GP-paper failed line %s\n" "$(date -Is)" "$LINENO" >> /mnt/data/thanhpv/ldva/logs/reproduction_errors.txt' ERR
"$GP_PY" -u "$AUTH" reproduce_gp.py --no-plots --shooting-candidates 100 \
  --smoke-test --wandb-mode online --output outputs/gp_paper100_smoke \
  > /mnt/data/thanhpv/ldva/logs/overnight/gp_paper_smoke.log 2>&1
"$GP_PY" -u "$AUTH" reproduce_gp.py --no-plots --shooting-candidates 100 \
  --wandb-mode online --output outputs/gp_paper100 \
  > /mnt/data/thanhpv/ldva/logs/overnight/gp_paper100.log 2>&1
