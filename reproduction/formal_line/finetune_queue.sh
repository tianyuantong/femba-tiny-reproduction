#!/bin/bash
# Fine-tuning, PTQ calibration and evaluation for one seed and one weight setting
# on the formal line (TUAR, 1/5-subject TUAB and the four TUSL event folds; full
# fine-tuning, 5 epochs; TUSL folds TUSL_EPOCHS=20, since 5 epochs over ~400
# windows are 60-70 steps and leave the classifier near uniform).
#   usage: SEED=42 BITS=8 SCHEME=uniform TASKS="tuar tuab tusl0 tusl1 tusl2 tusl3" ./finetune_queue.sh
# TUSL folds (tusl<k>) are three-class: evaluation is event-level
# (evaluate_events.py) and the PTQ percentile is chosen by validation macro AUROC.
# Waits for the seed's FP32 pretraining (and the matching QAP pretraining) to
# complete, then runs: FP32 fine-tune, C fine-tune, QAP fine-tune, PTQ sweep on
# the FP32 fine-tuned checkpoint, and evaluation of the three fine-tuned models.
set -u
PY=${PY:-/root/envs/femba/bin/python}
STUDY=${STUDY:-/root/work/femba-formal-study}
SRC=${SRC:-/root/work/femba-formal-source}
DATA_ROOT=${DATA_ROOT:-/root/work/femba-artifacts/subject-data-20261025}
TUSL_ROOT=${TUSL_ROOT:-/root/work/femba-artifacts/tusl-events-20261025}
PERCENTILES="99.999 99.99 99.9 99 98 95 90"
TUSL_EPOCHS=${TUSL_EPOCHS:-20}
ART=${ART:-/root/work/femba-artifacts/formal-20261025}
SEED=${SEED:-42}
BITS=${BITS:-8}
SCHEME=${SCHEME:-uniform}
TASKS=${TASKS:-tuar tuab}
arm=qap_w${BITS}a8; [ "$SCHEME" = ternary ] && arm=qap_ternary_a8
quant=(--weight-bit-width "$BITS" --activation-bit-width 8 --weight-quant-scheme "$SCHEME")
log="$ART/logs/seed$SEED-finetune-${arm#qap_}${LOG_SUFFIX:-}.log"

field() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
wait_for() { while [ ! -f "$1" ]; do sleep 120; done; }
# Shared outputs (FP32 stages, their evaluations) can be claimed by several queues:
# the first to create <out>.lock runs it, the others wait for its completion marker.
claim() { mkdir -p "$(dirname "$1")" && mkdir "$1.lock" 2>/dev/null; }
release() { rmdir "$1.lock" 2>/dev/null; }
wait_shared() {  # wait_shared <out> <marker>
  echo "[$(date '+%F %T')] another queue owns $1; waiting"
  while [ ! -f "$1/$2" ]; do
    if [ ! -d "$1.lock" ] && [ -f "$1/failure.json" ]; then echo "[$(date '+%F %T')] $1 failed elsewhere"; return 1; fi
    sleep 120
  done
}
data_dir() {
  case "$1" in
    tuab) echo "$DATA_ROOT/tuab-ft" ;;
    tusl[0-9]) echo "$TUSL_ROOT/fold${1#tusl}" ;;
    *) echo "$DATA_ROOT/$1" ;;
  esac
}
is_tusl() { case "$1" in tusl[0-9]) return 0 ;; *) return 1 ;; esac; }
metric() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["splits"]["val"]["events"]["macro_auroc"])' "$1"; }
stage() {  # stage <out> <group> <parent complete.json> <task> [quant args...]
  local out=$1 group=$2 parent=$3 task=$4; shift 4
  local epochs=(); is_tusl "$task" && epochs=(--max-epochs "$TUSL_EPOCHS")
  if [ -f "$out/complete.json" ]; then echo "[$(date '+%F %T')] skip $out"; return 0; fi
  claim "$out" || { wait_shared "$out" complete.json; return $?; }
  rm -rf "$out"
  echo "[$(date '+%F %T')] finetune $group $task -> $out"
  (cd "$STUDY" && "$PY" train_stage.py --source "$SRC" --data-root "$(data_dir "$task")" \
      --data-manifest "$(data_dir "$task")/data-manifest.json" --output "$out" --group "$group" --stage finetune \
      --seed "$SEED" --pretrained "$(field "$parent" best_checkpoint)" \
      --pretrained-sha256 "$(field "$parent" best_checkpoint_sha256)" "${epochs[@]}" "$@") >> "$log" 2>&1
  local code=$?; release "$out"; return $code
}
evaluate() {  # evaluate <stage out> <kind> <task>
  local run=$1 kind=$2 task=$3 out=$1/eval
  if [ -f "$out/results.json" ]; then return 0; fi
  claim "$out" || { wait_shared "$out" results.json; return $?; }
  rm -rf "$out"
  echo "[$(date '+%F %T')] evaluate $kind $task"
  local ckpt sha code; ckpt=$(field "$run/complete.json" best_checkpoint); sha=$(field "$run/complete.json" best_checkpoint_sha256)
  if is_tusl "$task"; then
    (cd "$STUDY" && "$PY" evaluate_events.py --source "$SRC" --data "$(data_dir "$task")" --checkpoint "$ckpt" \
        --checkpoint-sha256 "$sha" --output "$out" --model-kind "$kind") >> "$log" 2>&1
  else
    (cd "$STUDY" && "$PY" evaluate_one.py --source "$SRC" --data "$(data_dir "$task")" --checkpoint "$ckpt" \
        --checkpoint-sha256 "$sha" --output "$out" --model-kind "$kind" --seed "$SEED") >> "$log" 2>&1
  fi
  code=$?; release "$out"; return $code
}
ptq_tusl() {  # ptq_tusl <out> <task>: seven percentiles, validation macro AUROC selects, then one test evaluation
  local out=$1 task=$2 fp32=$ART/seed$SEED/fp32/finetune-$task/complete.json best="" best_score="-1"
  mkdir -p "$out"
  for p in $PERCENTILES; do
    local d=$out/p$p
    if [ ! -f "$d/val/results.json" ]; then
      rm -rf "$d"; echo "[$(date '+%F %T')] ptq $task percentile $p"
      (cd "$STUDY" && "$PY" calibrate.py --source "$SRC" --data "$(data_dir "$task")" --checkpoint "$(field "$fp32" best_checkpoint)" \
          --percentile "$p" --calibration-windows 400 --num-classes 3 --classification-type mcc --output "$d" "${quant[@]}" \
        && "$PY" evaluate_events.py --source "$SRC" --data "$(data_dir "$task")" --checkpoint "$(ls "$d"/ptq_*.ckpt | head -1)" \
          --checkpoint-sha256 "$(field "$d/results.json" checkpoint_sha256)" --output "$d/val" --model-kind ptq --splits val) >> "$log" 2>&1 || return 1
    fi
    local score; score=$(metric "$d/val/results.json")
    if "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$score" "$best_score"; then best=$p; best_score=$score; fi
  done
  echo "{\"selected_percentile\": $best, \"validation_macro_auroc\": $best_score, \"candidates\": \"$PERCENTILES\"}" > "$out/selection.json"
  if [ ! -f "$out/test/results.json" ]; then
    (cd "$STUDY" && "$PY" evaluate_events.py --source "$SRC" --data "$(data_dir "$task")" --checkpoint "$(ls "$out/p$best"/ptq_*.ckpt | head -1)" \
        --checkpoint-sha256 "$(field "$out/p$best/results.json" checkpoint_sha256)" --output "$out/test" --model-kind ptq --splits test) >> "$log" 2>&1 || return 1
  fi
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
  if is_tusl "$task"; then
    ptq_tusl "$out" "$task" || exit 5
  elif [ ! -f "$out/results.json" ]; then
    rm -rf "$out"; echo "[$(date '+%F %T')] ptq sweep $task"
    (cd "$STUDY" && "$PY" sweep.py --source "$SRC" --data "$(data_dir "$task")" \
        --fp32 "$(field "$ART/seed$SEED/fp32/finetune-$task/complete.json" best_checkpoint)" --output "$out" "${quant[@]}") >> "$log" 2>&1 || exit 5
  fi
  evaluate "$ART/seed$SEED/fp32/finetune-$task" fp32 "$task" || exit 6
  evaluate "$ART/seed$SEED/c_${arm#qap_}/finetune-$task" c "$task" || exit 7
  evaluate "$ART/seed$SEED/$arm/finetune-$task" qap_b "$task" || exit 8
done
echo "[$(date '+%F %T')] FINETUNE_QUEUE_DONE seed$SEED $arm"
