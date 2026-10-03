#!/usr/bin/env bash
# E1 - DMC Reacher closed-loop acquisition (PLAN.md 17).
#
# The six methods PLAN.md 17 E1 names, at 3 seeds (PLAN.md 21: >= 3 for
# development), measuring rollout return and success against acquired episodes.
#
# Run detached, because the full grid takes hours:
#   nohup bash experiments/run_e1_dmc.sh > runs/E1_dmc.log 2>&1 &
#
# Sizing. `n_initial=40` episodes gives 120 chunks and `budget=20` per round
# adds ~60, so four rounds take the dataset from 120 to ~360 - a 3x change.
# The earlier default (60 initial, 12 per round) moved it only 180 -> 288, and
# a 60% change in dataset size is too small to separate acquisition methods
# against the policy-training noise floor.
set -u

cd "$(dirname "$0")/.."
PY="$HOME/miniconda3/envs/deeplearning/bin/python"
OUT="${OUT:-runs/E1_dmc}"
SEEDS="${SEEDS:-3}"
ROUNDS="${ROUNDS:-4}"
BUDGET="${BUDGET:-20}"
N_INITIAL="${N_INITIAL:-40}"
METHODS="${METHODS:-random,diversity,direct_gradient_alignment,direct_influence,ldva_greedy,ldva_beam}"
# wandb on by default (PLAN.md 15 P1). WANDB=0 turns it off; the run still
# writes the same JSON reports either way, so logging is never load-bearing.
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ldva}"
EXPERIMENT="${EXPERIMENT:-E1}"
WB_FLAGS=""
if [ "$WANDB" = "1" ]; then
  WB_FLAGS="--wandb --wandb-project $WANDB_PROJECT --experiment $EXPERIMENT"
fi

mkdir -p "$OUT"
echo "=== E1 DMC reacher-easy ==="
echo "methods : $METHODS"
echo "seeds   : $SEEDS   rounds: $ROUNDS   budget/round: $BUDGET   D_0: $N_INITIAL episodes"
echo "started : $(date -Is)"
echo "git     : $(git rev-parse --short HEAD 2>/dev/null)"
echo "wandb   : ${WANDB_PROJECT} (group ${EXPERIMENT}-dmc), enabled=${WANDB}"
echo

# One method per invocation, so a crash in one method cannot lose the others
# and partial results are on disk the whole way through. Each writes its own
# report; `aggregate_e1.py` merges them.
IFS=',' read -ra M <<< "$METHODS"
for m in "${M[@]}"; do
  echo "---- $m : $(date -Is) ----"
  "$PY" -u experiments/run_acquisition_loop.py \
      --env dmc --env-task reacher-easy \
      --methods "$m" \
      --seeds "$SEEDS" --rounds "$ROUNDS" --budget "$BUDGET" \
      --n-initial "$N_INITIAL" \
      --out "$OUT/$m" --no-plots $WB_FLAGS
  echo "---- $m done: $(date -Is) ----"
done

echo
echo "=== E1 finished $(date -Is) ==="
"$PY" -u experiments/aggregate_e1.py --root "$OUT"
