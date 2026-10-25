#!/bin/bash
# Sequential pretraining queue for the formal line (2 epochs each, every epoch saved).
# Entries: group bits seed. Skips entries whose output already has complete.json.
set -u
PY=/root/envs/femba/bin/python
STUDY=/root/work/femba-formal-study
SRC=/root/work/femba-formal-source
DATA=/root/work/femba-artifacts/subject-data-20261025/tuab
ART=/root/work/femba-artifacts/formal-20261025
WAIT_PID=${WAIT_PID:-}
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
QUEUE="qap_b 8 42
qap_b 4 42
fp32 8 43
qap_b 8 43
qap_b 4 43
fp32 8 44
qap_b 8 44
qap_b 4 44"
echo "$QUEUE" | while read -r group bits seed; do
  name=$group; [ "$group" = qap_b ] && name="qap_w${bits}a8"
  out=$ART/seed$seed/$name/pretrain
  if [ -f "$out/complete.json" ]; then echo "[$(date +%F\ %T)] skip $name seed$seed"; continue; fi
  rm -rf "$out"
  extra=(); [ "$group" = qap_b ] && extra=(--weight-bit-width "$bits" --activation-bit-width 8)
  echo "[$(date +%F\ %T)] start $name seed$seed"
  cd "$STUDY" && "$PY" train_stage.py --source "$SRC" --data-root "$DATA" --data-manifest "$DATA/data-manifest.json" \
    --output "$out" --group "$group" --stage pretrain --seed "$seed" --max-epochs 2 --save-every-epoch "${extra[@]}" \
    > "$ART/logs/seed$seed-$name-pretrain.log" 2>&1
  code=$?
  echo "[$(date +%F\ %T)] end $name seed$seed exit=$code"
  [ $code -ne 0 ] && { echo "[$(date +%F\ %T)] QUEUE_STOPPED"; exit $code; }
done
echo "[$(date +%F\ %T)] QUEUE_DONE"
