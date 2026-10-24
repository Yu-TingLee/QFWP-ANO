#!/usr/bin/env bash
# Trains one RL arm on one environment, once per seed, with A3C.
#
#   scripts/train_rl.sh ARM ENV_ID      ARM: ANO-VQC | QFWP-ANO-R | QFWP-ANO-O | QFWP-ANO-RO
#
# Environment overrides:
#   SEEDS ("0 1 2 3 4 5 6 7 8 9"), LEARNABLE_SCALING (0), WORKERS (16), PYTHON (python)
#
# Results go to results/rl/<ENV_ID>/<ARM>/d4[-inputscale]/seed<N>/. Each run already
# uses WORKERS processes, so runs are not parallelised here. A failed seed is
# reported and the next one still runs.
set -u
arm=${1:?usage: scripts/train_rl.sh ARM ENV_ID}
env_id=${2:?usage: scripts/train_rl.sh ARM ENV_ID}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
LEARNABLE_SCALING=${LEARNABLE_SCALING:-0}
WORKERS=${WORKERS:-16}
PYTHON=${PYTHON:-python}

tag="d4"
[ "$LEARNABLE_SCALING" = "1" ] && tag="${tag}-inputscale"

failed=0
for seed in $SEEDS; do
  out_dir="results/rl/$env_id/$arm/$tag/seed$seed"
  if [ -f "$out_dir/episode_returns.pkl" ]; then echo "skip: $out_dir"; continue; fi
  mkdir -p "$out_dir"
  echo "=== $env_id / $arm / $tag / seed $seed ==="
  "$PYTHON" -u -m rl.train --arm "$arm" --env_id "$env_id" --seed "$seed" \
    --workers "$WORKERS" --learnable_scaling "$LEARNABLE_SCALING" \
    2>&1 | tee "$out_dir/run.log"
  status=${PIPESTATUS[0]}   # the python run, not tee
  if [ "$status" -ne 0 ]; then
    echo "WARN $env_id $arm $tag seed=$seed exited $status" >&2
    failed=1
  fi
done
exit $failed
