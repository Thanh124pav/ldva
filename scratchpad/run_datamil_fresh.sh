#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -f /mnt/data/thanhpv/ldva/logs/vram_target_90/stopped_by_user.json ]]; then
  echo 'Trajectory/10-trial DataMIL disabled by user. Use scratchpad/run_datamil_paper_h15.sh.'
  exit 0
fi
cd /mnt/data/thanhpv/ldva/external/datamil
export DATA_DIR=/mnt/data/thanhpv/datamil_data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export USE_TORCH=0 USE_FLAX=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/mnt/data/thanhpv/ldva/scratchpad/libero_config
export PYTHONPATH=/mnt/data/thanhpv/ldva/external/datamil:/mnt/data/thanhpv/ldva/external/LIBERO
export MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-datamil
DATA_PY=/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python
AUTH=/mnt/data/thanhpv/ldva/scratchpad/run_with_wandb.py
TASK_NAME=study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
INDICES=/mnt/data/thanhpv/datamil_data/exp/datamodels/libero90/book-caddy/selected_indices_topk0.1.npy
PRETRAINED=/home/thanhpv/.cache/huggingface/hub/models--rail-berkeley--octo-small/snapshots/03d88976c54a58e10480d2043a8c762b35bc2611
LOG_ROOT=/mnt/data/thanhpv/ldva/logs/overnight
OUTPUT_ROOT=/mnt/data/thanhpv/ldva/logs/datamil_fresh
trap 'printf "[%s] DataMIL fresh failed line %s: %s\n" "$(date -Is)" "$LINENO" "$BASH_COMMAND" >> /mnt/data/thanhpv/ldva/logs/reproduction_errors.txt' ERR

# Validate an actual successful train/checkpoint smoke before the larger batch.
"$DATA_PY" -c 'from pathlib import Path; import json; root=Path("/mnt/data/thanhpv/ldva/logs/datamil_training_smoke"); files=list(root.glob("**/training_metrics.jsonl")); assert any(len(p.read_text().splitlines())>=2 and (p.parent/"2").is_dir() for p in files)'

if ! "$DATA_PY" /mnt/data/thanhpv/ldva/scratchpad/profile_datamil_pilot.py >/dev/null 2>&1; then
"$DATA_PY" -u "$AUTH" scripts/cotraining.py \
  --config "scripts/configs/reproduction_cotrain_config.py:${TASK_NAME},libero90,${INDICES}" \
  --config.seed=0 --config.batch_size=128 --config.num_steps=20 \
  --config.log_interval=1 --config.save_interval=20 \
  --config.pretrained_path="$PRETRAINED" --config.save_dir="$OUTPUT_ROOT/pilot" \
  --name datamil-batch128-pilot --debug=false > "$LOG_ROOT/datamil_pilot.log" 2>&1
"$DATA_PY" /mnt/data/thanhpv/ldva/scratchpad/profile_datamil_pilot.py
fi

# Full, unchanged 10k-step schedule and batch size. Save state every 1000 steps.
for FRESH_SEED in 0 1 2 3 4; do
  TRAIN_STATUS=$("$DATA_PY" -c 'from pathlib import Path; import sys; root=Path(sys.argv[1])/"full/paper-reproduction/datamil-fresh-training"; paths=list(root.glob(f"seed{sys.argv[2]}_fresh_*")); assert len(paths)<=1, f"Ambiguous checkpoints: {paths}"; print("complete" if paths and (paths[0]/"10000").is_dir() else str(paths[0]) if paths else "new")' "$OUTPUT_ROOT" "$FRESH_SEED")
  if [[ "$TRAIN_STATUS" != complete ]]; then
  RESUME_ARGS=()
  if [[ "$TRAIN_STATUS" != new ]]; then RESUME_ARGS=(--resume_dir "$TRAIN_STATUS"); fi
  "$DATA_PY" -u "$AUTH" scripts/cotraining.py \
    --config "scripts/configs/reproduction_cotrain_config.py:${TASK_NAME},libero90,${INDICES}" \
    --config.seed="$FRESH_SEED" --config.batch_size=128 --config.num_steps=10000 \
    --config.log_interval=100 --config.save_interval=1000 \
    --config.pretrained_path="$PRETRAINED" --config.save_dir="$OUTPUT_ROOT/full" \
    --name "seed${FRESH_SEED}_fresh_libero90_book-caddy" --debug=false "${RESUME_ARGS[@]}" \
    >> "$LOG_ROOT/datamil_fresh_seed${FRESH_SEED}.log" 2>&1
  else
    echo "Reuse completed fresh seed $FRESH_SEED"
  fi
  "$DATA_PY" -u "$AUTH" datamil/eval_libero_parallel.py \
    --task_name book-caddy --ckpt "$OUTPUT_ROOT/full/paper-reproduction/datamil-fresh-training" \
    --out_dir "$OUTPUT_ROOT/evaluation" --num_envs 5 --total_evals 50 --seeds "$FRESH_SEED" \
    --checkpoint_provenance 'fresh training by reproduction pipeline; 10 supplied metagradient trials' \
    --run_group datamil-fresh-evaluation \
    > "$LOG_ROOT/datamil_fresh_eval_seed${FRESH_SEED}.log" 2>&1
done
