#!/usr/bin/env bash
# Second launch: more seeds for E0 (3..9), full --seeds 10 for E1/E2.
set -u

cd /home/thanhpv/ldva
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PY=/home/thanhpv/acvcl/.venv/bin/python

SEEDS=${SEEDS:-10}
ROUNDS=${ROUNDS:-4}
BUDGET=${BUDGET:-20}
N_INITIAL=${N_INITIAL:-40}
METHODS=(random diversity direct_gradient_alignment direct_influence ldva_greedy ldva_beam)

mkdir -p runs/E1_dmc runs/E2_metaworld logs

echo "=== Launching E1 (DMC reacher-easy), --seeds $SEEDS ==="
for m in "${METHODS[@]}"; do
  rm -rf "runs/E1_dmc/$m"
  nohup $PY -u experiments/run_acquisition_loop.py \
      --env dmc --env-task reacher-easy \
      --methods "$m" \
      --seeds "$SEEDS" --rounds "$ROUNDS" --budget "$BUDGET" \
      --n-initial "$N_INITIAL" \
      --out "runs/E1_dmc/$m" --no-plots \
      > "logs/E1_${m}.log" 2>&1 &
  echo "  E1 $m -> pid $!"
done

echo "=== Launching E2 (MetaWorld push-v3), --seeds $SEEDS ==="
for m in "${METHODS[@]}"; do
  rm -rf "runs/E2_metaworld/$m"
  nohup $PY -u experiments/run_acquisition_loop.py \
      --env metaworld --env-task push-v3 \
      --methods "$m" \
      --seeds "$SEEDS" --rounds "$ROUNDS" --budget "$BUDGET" \
      --n-initial "$N_INITIAL" \
      --out "runs/E2_metaworld/$m" --no-plots \
      > "logs/E2_${m}.log" 2>&1 &
  echo "  E2 $m -> pid $!"
done

echo "=== Launching E0 extra seeds 3..9 (seeds 0..2 already saved) ==="
for s in 3 4 5 6 7 8 9; do
  nohup $PY -u experiments/synthetic/run_stage0.py \
      --seed $s \
      --out "runs/E0_full/seed$s" \
      > "logs/E0_seed${s}.log" 2>&1 &
  echo "  E0 seed$s -> pid $!"
done

echo
echo "=== Launched. Total procs: $((6+6+7)) ==="
jobs -p > logs/pids.txt
wc -l logs/pids.txt
