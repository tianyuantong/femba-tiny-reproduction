#!/bin/bash
# FP16 and BF16 replays of the FP32 fine-tuned checkpoints, on every formal task. Run it in the STUDY directory.
#   usage: SEEDS="42 43 44" TASKS="tuab tuar" DTYPES="bfloat16 float16" ./run_half_precision.sh
set -u
PY=${PY:-/root/envs/femba/bin/python}
SRC=${SRC:-/root/work/femba-formal-source}
DATA_ROOT=${DATA_ROOT:-/root/work/femba-artifacts/subject-data-20261025}
TUSL_ROOT=${TUSL_ROOT:-/root/work/femba-artifacts/tusl-events-20261026-v2}
ART=${ART:-/root/work/femba-artifacts/formal-20261025}
OUT=${OUT:-$ART/half-precision}
SEEDS=${SEEDS:-42 43 44}
TASKS=${TASKS:-tuar tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2 tuab}
DTYPES=${DTYPES:-bfloat16 float16}
field() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
data_dir() {
  case "$1" in
    tuab) echo "$DATA_ROOT/tuab-ft" ;;
    tusl[0-9]-v2) local fold=${1#tusl}; echo "$TUSL_ROOT/fold${fold%-v2}" ;;
    *) echo "$DATA_ROOT/$1" ;;
  esac
}
failed=0
for dtype in $DTYPES; do
  for seed in $SEEDS; do
    for task in $TASKS; do
      run=$ART/seed$seed/fp32/finetune-$task
      echo "[$(date '+%F %T')] $dtype seed $seed $task"
      "$PY" half_precision.py --source "$SRC" --data "$(data_dir "$task")" --dtype "$dtype" \
        --checkpoint "$(field "$run/complete.json" best_checkpoint)" \
        --checkpoint-sha256 "$(field "$run/complete.json" best_checkpoint_sha256)" \
        --output "$OUT/$dtype/seed$seed/fp32/finetune-$task/eval" || failed=1
    done
  done
done
[ "$failed" = 0 ] && echo HALF_PRECISION_DONE || echo HALF_PRECISION_FAILED
