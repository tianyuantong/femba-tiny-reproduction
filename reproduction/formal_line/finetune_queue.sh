#!/bin/bash
# Fine-tuning, PTQ calibration and evaluation for one seed and one weight setting
# on the formal line (TUAR and 1/5-subject TUAB tasks, full fine-tuning, 5 epochs).
#   usage: SEED=42 BITS=8 SCHEME=uniform ./finetune_queue.sh
# Waits for the seed's FP32 pretraining (and the matching QAP pretraining) to
# complete, then runs: FP32 fine-tune, C fine-tune, QAP fine-tune, PTQ sweep on
# the FP32 fine-tuned checkpoint, and evaluation of the three fine-tuned models.
set -u
PY=${PY:-/root/envs/femba/bin/python}
STUDY=${STUDY:-/root/work/femba-formal-study}
SRC=${SRC:-/root/work/femba-formal-source}
DATA_ROOT=${DATA_ROOT:-/root/work/femba-artifacts/subject-data-20261025}
ART=${ART:-/root/work/femba-artifacts/formal-20261025}
SEED=${SEED:-42}
BITS=${BITS:-8}
SCHEME=${SCHEME:-uniform}
TASKS=${TASKS:-tuar tuab}
arm=qap_w${BITS}a8; [ "$SCHEME" = ternary ] && arm=qap_ternary_a8
quant=(--weight-bit-width "$BITS" --activation-bit-width 8 --weight-quant-scheme "$SCHEME")
log="$ART/logs/seed$SEED-finetune-${arm#qap_}.log"

field() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
wait_for() { while [ ! -f "$1" ]; do sleep 120; done; }
data_dir() { [ "$1" = tuab ] && echo "$DATA_ROOT/tuab-ft" || echo "$DATA_ROOT/$1"; }
stage() {  # stage <out> <group> <parent complete.json> <task> [quant args...]
  local out=$1 group=$2 parent=$3 task=$4; shift 4
  if [ -f "$out/complete.json" ]; then echo "[$(date '+%F %T')] skip $out"; return 0; fi
  rm -rf "$out"
  echo "[$(date '+%F %T')] finetune $group $task -> $out"
  (cd "$STUDY" && "$PY" train_stage.py --source "$SRC" --data-root "$(data_dir "$task")" \
      --data-manifest "$(data_dir "$task")/data-manifest.json" --output "$out" --group "$group" --stage finetune \
      --seed "$SEED" --pretrained "$(field "$parent" best_checkpoint)" \
      --pretrained-sha256 "$(field "$parent" best_checkpoint_sha256)" "$@") >> "$log" 2>&1 || return 1
}
evaluate() {  # evaluate <stage out> <kind> <task>
  local run=$1 kind=$2 task=$3 out=$1/eval
  if [ -f "$out/results.json" ]; then return 0; fi
  rm -rf "$out"
  echo "[$(date '+%F %T')] evaluate $kind $task"
  (cd "$STUDY" && "$PY" evaluate_one.py --source "$SRC" --data "$(data_dir "$task")" --checkpoint "$(field "$run/complete.json" best_checkpoint)" \
      --checkpoint-sha256 "$(field "$run/complete.json" best_checkpoint_sha256)" --output "$out" --model-kind "$kind" --seed "$SEED") >> "$log" 2>&1 || return 1
}

fp32_pre=$ART/seed$SEED/fp32/pretrain/complete.json
qap_pre=$ART/seed$SEED/$arm/pretrain/complete.json
echo "[$(date '+%F %T')] waiting for $fp32_pre"; wait_for "$fp32_pre"
for task in $TASKS; do
  stage "$ART/seed$SEED/fp32/finetune-$task" fp32 "$fp32_pre" "$task" || exit 2
  stage "$ART/seed$SEED/c_${arm#qap_}/finetune-$task" c "$fp32_pre" "$task" "${quant[@]}" || exit 3
done
echo "[$(date '+%F %T')] waiting for $qap_pre"; wait_for "$qap_pre"
for task in $TASKS; do
  stage "$ART/seed$SEED/$arm/finetune-$task" qap_b "$qap_pre" "$task" "${quant[@]}" || exit 4
done
for task in $TASKS; do
  out=$ART/seed$SEED/ptq_${arm#qap_}/sweep-$task
  if [ ! -f "$out/results.json" ]; then
    rm -rf "$out"; echo "[$(date '+%F %T')] ptq sweep $task"
    (cd "$STUDY" && "$PY" sweep.py --source "$SRC" --data "$(data_dir "$task")" \
        --fp32 "$(field "$ART/seed$SEED/fp32/finetune-$task/complete.json" best_checkpoint)" --output "$out" "${quant[@]}") >> "$log" 2>&1 || exit 5
  fi
  evaluate "$ART/seed$SEED/fp32/finetune-$task" fp32 "$task" || exit 6
  evaluate "$ART/seed$SEED/c_${arm#qap_}/finetune-$task" c "$task" || exit 7
  evaluate "$ART/seed$SEED/$arm/finetune-$task" qap_b "$task" || exit 8
done
echo "[$(date '+%F %T')] FINETUNE_QUEUE_DONE seed$SEED $arm"
