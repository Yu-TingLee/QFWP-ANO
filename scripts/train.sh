#!/usr/bin/env bash
# Trains and tests one QFWP-ANO variant on one ETT dataset, once per seed.
#
#   scripts/train.sh MODEL DATASET      MODEL: QFWP-ANO-R | QFWP-ANO-O | QFWP-ANO-RO
#
# Environment overrides:
#   SEEDS ("1 2 3 4 5"), PRED_LEN (1), SEQ_LEN (16), N_LOCAL (1), NUM_LAYERS (3),
#   PYTHON (python), DRY_RUN (0)
#
# Results go to results/mtsf/<run>/<DATASET>_L<SEQ_LEN>_H<PRED_LEN>_s<seed>/, where <run>
# is MODEL-comb<N_LOCAL>local plus a suffix for every non-default setting.
set -e
model=${1:?usage: scripts/train.sh MODEL DATASET}
data=${2:?usage: scripts/train.sh MODEL DATASET}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=${SEEDS:-"1 2 3 4 5"}
PRED_LEN=${PRED_LEN:-1}
SEQ_LEN=${SEQ_LEN:-16}
N_LOCAL=${N_LOCAL:-1}
NUM_LAYERS=${NUM_LAYERS:-3}
PYTHON=${PYTHON:-python}

case "$data" in
  ETTh1|ETTh2) freq=h ;;
  ETTm1|ETTm2) freq=t ;;
  *) echo "train.sh: unknown dataset '$data'" >&2; exit 1 ;;
esac
case "$model" in
  QFWP-ANO-R|QFWP-ANO-O|QFWP-ANO-RO) ;;
  *) echo "train.sh: unknown model '$model'" >&2; exit 1 ;;
esac
enc_in=7

suffix="-comb${N_LOCAL}local"
[ "$NUM_LAYERS" != "3" ] && suffix="${suffix}-L${NUM_LAYERS}"
[ "$SEQ_LEN" != "16" ]   && suffix="${suffix}-lookback${SEQ_LEN}"

run="${model}${suffix}"
mkdir -p "logs/$run"
for seed in $SEEDS; do
  out_dir="results/mtsf/$run/${data}_L${SEQ_LEN}_H${PRED_LEN}_s${seed}"
  if [ -f "$out_dir/metrics.npz" ]; then echo "skip: $out_dir"; continue; fi

  cmd=("$PYTHON" -u run.py
    --random_seed "$seed"
    --model "$model"
    --save_suffix="$suffix"
    --data "$data"
    --root_path "$ROOT_DIR/dataset/ETT-small/"
    --data_path "${data}.csv"
    --features M
    --freq "$freq"
    --seq_len "$SEQ_LEN"
    --label_len 0
    --pred_len "$PRED_LEN"
    --enc_in "$enc_in"
    --num_qubits "$enc_in"
    --num_layers "$NUM_LAYERS"
    --n_local "$N_LOCAL"
    --d_hidden 64
    --d_dec 64
    --use_revin 1
    --QML_device default.qubit
    --batch_size 1024
    --num_workers 0
    --train_epochs 100
    --patience 10
    --learning_rate 0.001
    --lradj constant
    --checkpoints "./checkpoints/$run/")

  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%s\n' "${cmd[*]}"; continue; fi
  "${cmd[@]}" > "logs/$run/${model}_${data}_${SEQ_LEN}_${PRED_LEN}_s${seed}.log" 2>&1 \
    || echo "WARN $run $data H=$PRED_LEN seed=$seed exited non-zero"
  echo "done -> $out_dir"
done
