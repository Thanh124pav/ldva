#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/thanhpv/ldva/external/datamil
export DATA_DIR=/mnt/data/thanhpv/datamil_data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export USE_TORCH=0 USE_FLAX=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/mnt/data/thanhpv/ldva/external/datamil
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-datamil
TASK_NAME=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
INDICES=/mnt/data/thanhpv/datamil_data/exp/datamodels/libero90/book-caddy/selected_indices_topk0.1.npy
/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python -u \
  /mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py scripts/cotraining.py \
  --config "scripts/configs/reproduction_cotrain_config.py:${TASK_NAME},libero90,${INDICES}" \
  --config.seed=0 --config.batch_size=8 --config.num_steps=2 \
  --config.shuffle_buffer_size=32 --config.log_interval=1 --config.save_interval=2 \
  --config.optimizer.learning_rate.warmup_steps=0 \
  --config.pretrained_path=/home/thanhpv/.cache/huggingface/hub/models--rail-berkeley--octo-small/snapshots/03d88976c54a58e10480d2043a8c762b35bc2611 \
  --config.save_dir=/mnt/data/thanhpv/ldva/logs/datamil_training_smoke \
  --name datamil-fresh-training-smoke --debug=false \
  > /mnt/data/thanhpv/ldva/logs/overnight/datamil_training_smoke.log 2>&1
