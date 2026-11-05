#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "DataMIL H15 GPU smoke failed at line %s; see logs/datamil_paper_h15/gpu_smoke_console.log\n" "$LINENO" >> /mnt/data/thanhpv/ldva/logs/reproduction_errors.txt' ERR
cd /mnt/data/thanhpv/ldva
export DATA_DIR=/mnt/data/thanhpv/datamil_data
export DATAMIL_TARGET=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
export DATAMIL_PAPER_MDS=/mnt/data/thanhpv/ldva/logs/datamil_paper_h15/gpu_smoke/mds
export DATAMIL_PAPER_TF=/mnt/data/thanhpv/datamil_data/paper_h15_t30/tf
export USE_TORCH=0 USE_FLAX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.75
export DATAMIL_DM_PROGRESS=1
export PYTHONPATH=/mnt/data/thanhpv/ldva/external/datamil
export MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-datamil-paper
DATA_PY=/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python
SMOKE_OUT=/mnt/data/thanhpv/ldva/logs/datamil_paper_h15/gpu_smoke/datamodels
if [[ -f "$SMOKE_OUT/libero90_horizon/book-caddy-smoke/iter_0/complete.json" ]]; then
  echo 'Reuse successful actual GPU metagradient smoke'
  exit 0
fi
CUDA_VISIBLE_DEVICES='' "$DATA_PY" scratchpad/prepare_datamil_dm_smoke.py
cd external/datamil
"$DATA_PY" -u /mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py \
  datamil/metagradients/do_libero_data_selection_parallel.py \
  --config scripts/configs/paper_h15_dm_config.py \
  --config.pretrained_path=/home/thanhpv/.cache/huggingface/hub/models--rail-berkeley--octo-small/snapshots/03d88976c54a58e10480d2043a8c762b35bc2611 \
  --config.num_steps=4 --config.bob_steps=2 --config.job_id=0 \
  --config.folder_name=book-caddy-smoke --config.save_dir="$SMOKE_OUT" \
  --config.wandb.group=datamil-paper-h15-smoke --debug=false
