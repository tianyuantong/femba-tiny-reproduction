#!/bin/bash
# Remaining formal-line fine-tuning queues after seed 42 int8, one batch at a time
# in priority order. A batch starts once the previous batch's last fine-tuning stage
# has completed, so pretraining shares the GPU with one fine-tuning job at a time and
# only the previous batch's PTQ sweeps and evaluations overlap. A failed stage stops
# the chain: its completion marker never appears. Ternary batches start only after
# ternary_smoke.sh passed and are skipped if it failed. A batch whose queue log exists was
# started by an earlier launcher and is only waited for, so the launcher can be restarted.
#   usage: QUEUE=./finetune_queue.sh nohup ./launch_batches.sh > "$ART/logs/launch-batches.log" 2>&1 &
set -u
STUDY=${STUDY:-/root/work/femba-formal-study}
ART=${ART:-/root/work/femba-artifacts/formal-20261025}
QUEUE=${QUEUE:-./finetune_queue.sh}
SMOKE=${SMOKE:-/root/work/femba-artifacts/ternary-smoke}
TASKS=${TASKS:-tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2}
# seed:bits:scheme; ternary seed 42 follows seed 43 int8 so its QAP stages find the
# ternary pretraining (queued after all uniform pretraining) already complete. A machine
# that runs only part of the matrix passes its own list, e.g. BATCHES="42:2:ternary".
BATCHES=${BATCHES:-42:4:uniform 43:8:uniform 42:2:ternary 43:4:uniform 44:8:uniform 44:4:uniform 43:2:ternary 44:2:ternary}

wait_for() { while [ ! -f "$1" ]; do sleep 300; done; }

# the first batch waits for this marker; START_AFTER=none starts it at once
previous=${START_AFTER:-$ART/seed42/qap_w8a8/finetune-tuab/complete.json}
for batch in $BATCHES; do
  IFS=: read -r seed bits scheme <<< "$batch"
  setting=w${bits}a8; [ "$scheme" = ternary ] && setting=ternary_a8
  queue_log=$ART/logs/finetune-queue-seed$seed-$setting-all-v2.log
  if [ -f "$queue_log" ]; then
    echo "[$(date '+%F %T')] seed$seed $setting was started before"
  else
    echo "[$(date '+%F %T')] seed$seed $setting waits for $previous"
    [ "$previous" = none ] || wait_for "$previous"
    if [ "$scheme" = ternary ]; then
      while [ ! -f "$SMOKE/PASSED" ] && [ ! -f "$SMOKE/FAILED" ]; do sleep 300; done
      if [ ! -f "$SMOKE/PASSED" ]; then echo "[$(date '+%F %T')] skip seed$seed $setting: ternary smoke test failed"; continue; fi
    fi
    echo "[$(date '+%F %T')] start seed$seed $setting"
    (cd "$STUDY" && SEED=$seed BITS=$bits SCHEME=$scheme TASKS="$TASKS" LOG_SUFFIX=-all \
        nohup "$QUEUE" > "$queue_log" 2>&1 &)
  fi
  previous=$ART/seed$seed/qap_$setting/finetune-${TASKS##* }/complete.json
done
echo "[$(date '+%F %T')] LAUNCH_BATCHES_DONE"
