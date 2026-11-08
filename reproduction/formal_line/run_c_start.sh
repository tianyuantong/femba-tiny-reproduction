#!/bin/bash
# Group C on TUAB with its first activation scales at the 95th percentile; otherwise the
# formal fine-tuning and evaluation. RUNS lists activation width:seed pairs, one at a time.
#   usage: RUNS="4:42" WAIT_PID=501030 ./run_c_start.sh
set -u
PY=${PY:-/root/envs/femba/bin/python}
SRC=${SRC:-/root/work/femba-formal-source}
ART=${ART:-/root/work/femba-artifacts/activation-20261104}
DATA=${DATA:-/root/work/femba-artifacts/subject-data-20261025/tuab-ft}
PERCENTILE=${PERCENTILE:-95}
RUNS=${RUNS:?activation width:seed pairs}

field() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

while [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
for run in $RUNS; do
  IFS=: read -r abits seed <<< "$run"
  out=$ART/seed$seed/c_p${PERCENTILE}_w8a$abits/finetune-tuab
  log=$ART/logs/seed$seed-c-p$PERCENTILE-w8a$abits.log
  parent=$ART/seed$seed/fp32/pretrain/complete.json
  echo "[$(date '+%F %T')] start a$abits seed$seed"
  if [ ! -f "$out/complete.json" ]; then
    "$PY" train_stage.py --source "$SRC" --data-root "$DATA" --data-manifest "$DATA/data-manifest.json" --output "$out" \
      --group c --stage finetune --seed "$seed" --pretrained "$(field "$parent" best_checkpoint)" \
      --pretrained-sha256 "$(field "$parent" best_checkpoint_sha256)" --activation-percentile "$PERCENTILE" \
      --weight-bit-width 8 --activation-bit-width "$abits" --weight-quant-scheme uniform >> "$log" 2>&1 \
      || { echo "[$(date '+%F %T')] C_START_FAILED a$abits seed$seed"; exit 1; }
  fi
  "$PY" evaluate_one.py --source "$SRC" --data "$DATA" --checkpoint "$(field "$out/complete.json" best_checkpoint)" \
    --checkpoint-sha256 "$(field "$out/complete.json" best_checkpoint_sha256)" --output "$out/eval" \
    --model-kind c --seed "$seed" >> "$log" 2>&1 \
    || { echo "[$(date '+%F %T')] C_START_FAILED evaluate a$abits seed$seed"; exit 2; }
  echo "[$(date '+%F %T')] done a$abits seed$seed"
done
echo "[$(date '+%F %T')] C_START_DONE"
