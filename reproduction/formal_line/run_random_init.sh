#!/bin/bash
# Random-initialization baseline at W8A8: group C fine-tuning with no pretrained parent on every
# formal task, then the formal evaluation, one stage at a time. Run it in the STUDY directory.
#   usage: SEEDS="42 43 44" WAIT_PID=<pid> ./run_random_init.sh
set -u
PY=${PY:-/root/envs/femba/bin/python}
SRC=${SRC:-/root/work/femba-formal-source}
DATA_ROOT=${DATA_ROOT:-/root/work/femba-artifacts/subject-data-20261025}
TUSL_ROOT=${TUSL_ROOT:-/root/work/femba-artifacts/tusl-events-20261026-v2}
ART=${ART:-/root/work/femba-artifacts/random-init-20261106}
TUSL_EPOCHS=${TUSL_EPOCHS:-20}
SEEDS=${SEEDS:-42 43 44}
TASKS=${TASKS:-tuar tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2 tuab}
field() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
data_dir() {
  case "$1" in
    tuab) echo "$DATA_ROOT/tuab-ft" ;;
    tusl[0-9]-v2) local fold=${1#tusl}; echo "$TUSL_ROOT/fold${fold%-v2}" ;;
    *) echo "$DATA_ROOT/$1" ;;
  esac
}
while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
mkdir -p "$ART/logs"
for seed in $SEEDS; do
  for task in $TASKS; do
    out=$ART/seed$seed/c_random_w8a8/finetune-$task
    log=$ART/logs/seed$seed-random-w8a8.log
    data=$(data_dir "$task")
    epochs=(); case "$task" in tusl*) epochs=(--max-epochs "$TUSL_EPOCHS") ;; esac
    echo "[$(date '+%F %T')] start $task seed$seed"
    if [ ! -f "$out/complete.json" ]; then
      rm -rf "$out"; mkdir -p "$(dirname "$out")"
      "$PY" train_stage.py --source "$SRC" --data-root "$data" --data-manifest "$data/data-manifest.json" --output "$out" \
        --group c --stage finetune --seed "$seed" --random-init "${epochs[@]}" \
        --weight-bit-width 8 --activation-bit-width 8 --weight-quant-scheme uniform >> "$log" 2>&1 \
        || { echo "[$(date '+%F %T')] RANDOM_INIT_FAILED $task seed$seed"; exit 1; }
    fi
    ckpt=$(field "$out/complete.json" best_checkpoint); sha=$(field "$out/complete.json" best_checkpoint_sha256)
    case "$task" in
      tusl*) "$PY" evaluate_events.py --source "$SRC" --data "$data" --checkpoint "$ckpt" --checkpoint-sha256 "$sha" \
               --output "$out/eval" --model-kind c >> "$log" 2>&1 ;;
      *) "$PY" evaluate_one.py --source "$SRC" --data "$data" --checkpoint "$ckpt" --checkpoint-sha256 "$sha" \
           --output "$out/eval" --model-kind c --seed "$seed" >> "$log" 2>&1 ;;
    esac || { echo "[$(date '+%F %T')] RANDOM_INIT_FAILED evaluate $task seed$seed"; exit 2; }
    echo "[$(date '+%F %T')] done $task seed$seed"
  done
done
echo "[$(date '+%F %T')] RANDOM_INIT_DONE"
