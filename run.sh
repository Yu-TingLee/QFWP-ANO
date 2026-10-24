#!/usr/bin/env bash
# Runs every experiment in the paper.
#   bash run.sh [PHASE ...]   (default: mtsf rl)
#
#   mtsf  Tables 1-2: QFWP-ANO-{R,O,RO} x k = 1..7 x D = 1, 3 x 4 ETT datasets x
#         5 horizons x 10 seeds, JOBS training runs at a time.
#   rl    Figure: {ANO-VQC, QFWP-ANO-R, -O, -RO} x {Acrobot, SimpleCrossingS9N1}
#         x input scaling off / on x 10 seeds, one run at a time (each uses
#         16 A3C workers).
#
# Environment overrides: PYTHON SEEDS JOBS THREADS_PER_JOB (mtsf), RL_SEEDS (rl)
set -e
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
RL_SEEDS=${RL_SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
JOBS=${JOBS:-5}
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
if [ -z "${THREADS_PER_JOB:-}" ]; then
  THREADS_PER_JOB=$(( NPROC / JOBS )); [ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1
fi

if [ "$#" -gt 0 ]; then PHASES="$*"; else PHASES=${PHASES:-"mtsf rl"}; fi

ETT="ETTh1 ETTh2 ETTm1 ETTm2"
ETT_PLS="1 5 16 32 48"
VARIANTS="QFWP-ANO-R QFWP-ANO-O QFWP-ANO-RO"
RL_ARMS="ANO-VQC QFWP-ANO-R QFWP-ANO-O QFWP-ANO-RO"
RL_ENVS="Acrobot-v1 MiniGrid-SimpleCrossingS9N1-v0"

running=0
throttle() {
  running=$((running + 1))
  if (( running >= JOBS )); then wait -n 2>/dev/null || true; running=$((running - 1)); fi
}

# launch MODEL DATASET [VAR=value ...]
launch() {
  local model="$1" data="$2"; shift 2
  env "$@" PYTHON="$PYTHON" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    bash "$BASE_DIR/scripts/train.sh" "$model" "$data" &
  throttle
}

phase_banner() { echo; echo "########## phase: $1 ##########"; }

# MTSF: 3 variants x k = 1..7 x D = 1, 3 x 4 datasets x 5 horizons.
phase_mtsf() {
  phase_banner "mtsf"
  local m k nl d pl seed
  for m in $VARIANTS; do
    for k in 1 2 3 4 5 6 7; do
      for nl in 1 3; do
        for d in $ETT; do
          for pl in $ETT_PLS; do
            for seed in $SEEDS; do
              launch "$m" "$d" "SEEDS=$seed" "PRED_LEN=$pl" "N_LOCAL=$k" "NUM_LAYERS=$nl"
            done
          done
        done
      done
    done
  done
  wait || true; running=0
}

# RL: 4 arms x 2 envs x input scaling off / on, sequentially.
phase_rl() {
  phase_banner "rl"
  local env arm ls status=0
  for env in $RL_ENVS; do
    for arm in $RL_ARMS; do
      for ls in 0 1; do
        SEEDS="$RL_SEEDS" LEARNABLE_SCALING="$ls" PYTHON="$PYTHON" \
          bash "$BASE_DIR/scripts/train_rl.sh" "$arm" "$env" || status=1
      done
    done
  done
  [ "$status" -eq 0 ] || echo "!! some RL runs failed; re-run to retry them (finished seeds are skipped)" >&2
}

echo "run.sh: JOBS=$JOBS THREADS_PER_JOB=$THREADS_PER_JOB (cores=$NPROC)  phases=[$PHASES]"
for phase in $PHASES; do
  case "$phase" in
    mtsf) phase_mtsf ;;
    rl)   phase_rl ;;
    *) echo "!! unknown phase: '$phase' (valid: mtsf rl)" >&2; exit 1 ;;
  esac
done

echo
echo "=== run.sh complete: phases [$PHASES] ==="
