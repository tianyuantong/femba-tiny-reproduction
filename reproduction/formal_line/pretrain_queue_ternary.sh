#!/bin/bash
# Ternary-weight (TWN, A8) QAP pretraining for seeds 42/43/44, after the main queue.
set -u
PY=${PY:-/root/envs/femba/bin/python}
STUDY=${STUDY:-/root/work/femba-formal-study}
SRC=${SRC:-/root/work/femba-formal-source}
DATA=${DATA:-/root/work/femba-artifacts/subject-data-20261025/tuab}
ART=${ART:-/root/work/femba-artifacts/formal-20261025}
WAIT_PID=${WAIT_PID:-}
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
for seed in 42 43 44; do
  out=$ART/seed$seed/qap_ternary_a8/pretrain
  if [ -f "$out/complete.json" ]; then echo "[$(date '+%F %T')] skip ternary seed$seed"; continue; fi
  rm -rf "$out"
  echo "[$(date '+%F %T')] start ternary seed$seed"
  cd "$STUDY" && "$PY" train_stage.py --source "$SRC" --data-root "$DATA" --data-manifest "$DATA/data-manifest.json" \
    --output "$out" --group qap_b --stage pretrain --seed "$seed" --max-epochs 2 --save-every-epoch \
    --weight-bit-width 2 --weight-quant-scheme ternary --activation-bit-width 8 \
    > "$ART/logs/seed$seed-qap_ternary_a8-pretrain.log" 2>&1
  code=$?
  echo "[$(date '+%F %T')] end ternary seed$seed exit=$code"
  [ $code -ne 0 ] && { echo "[$(date '+%F %T')] QUEUE_STOPPED"; exit $code; }
done
echo "[$(date '+%F %T')] QUEUE_DONE"
