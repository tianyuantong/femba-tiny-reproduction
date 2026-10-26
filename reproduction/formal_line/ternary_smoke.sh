#!/bin/bash
# Pre-flight check for the ternary QAP pretraining: the whole ternary path of the formal line
# on tiny data. CPU-built 1024/256/256-window subsets of TUAB and TUAR (make_smoke_data.py) and
# TUSL v2 fold 0; one-epoch ternary QAP pretraining on the TUAB subset; then finetune_queue.sh
# with BITS=2 SCHEME=ternary on tuar and tusl0-v2 (FP32, C and QAP fine-tunes, PTQ sweeps,
# evaluations; TUSL one epoch). Passes only if the queue finishes and every quantized evaluation
# reports ternary weights with 2-bit weight quantizers; writes $OUT/PASSED or $OUT/FAILED.
#   usage: OUT=/root/work/femba-artifacts/ternary-smoke QUEUE=./finetune_queue.sh ./ternary_smoke.sh
set -u
PY=${PY:-/root/envs/femba/bin/python}
STUDY=${STUDY:-/root/work/femba-formal-study}
SRC=${SRC:-/root/work/femba-formal-source}
DATA_ROOT=${DATA_ROOT:-/root/work/femba-artifacts/subject-data-20261025}
TUSL_ROOT=${TUSL_ROOT:-/root/work/femba-artifacts/tusl-events-20261026-v2}
FORMAL_ART=${FORMAL_ART:-/root/work/femba-artifacts/formal-20261025}
QUEUE=${QUEUE:-./finetune_queue.sh}
OUT=${OUT:-/root/work/femba-artifacts/ternary-smoke}
art=$OUT/art

fail() { echo "[$(date '+%F %T')] SMOKE_FAILED: $1"; echo "{\"failed\": \"$1\"}" > "$OUT/FAILED"; exit 1; }
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 2; fi
mkdir -p "$OUT/data" "$art/seed42/fp32/pretrain" "$art/logs"
echo "[$(date '+%F %T')] smoke data"
for corpus in tuab tuar; do
  "$PY" "$STUDY/make_smoke_data.py" --source "$DATA_ROOT/$corpus" --output "$OUT/data/$corpus" || fail "$corpus subset"
done
cp "$FORMAL_ART/seed42/fp32/pretrain/complete.json" "$art/seed42/fp32/pretrain/complete.json" || fail "FP32 parent"
echo "[$(date '+%F %T')] ternary QAP pretraining, one epoch"
(cd "$STUDY" && "$PY" train_stage.py --source "$SRC" --data-root "$OUT/data/tuab" --data-manifest "$OUT/data/tuab/data-manifest.json" \
    --output "$art/seed42/qap_ternary_a8/pretrain" --group qap_b --stage pretrain --seed 42 --max-epochs 1 \
    --weight-bit-width 2 --weight-quant-scheme ternary --activation-bit-width 8) > "$art/logs/pretrain.log" 2>&1 || fail "ternary pretraining"
echo "[$(date '+%F %T')] fine-tuning queue, ternary, tuar and tusl0-v2"
(cd "$STUDY" && ART="$art" DATA_ROOT="$OUT/data" TUSL_ROOT="$TUSL_ROOT" TUSL_EPOCHS=1 FP32_PARENTS=run SEED=42 BITS=2 SCHEME=ternary \
    TASKS="tuar tusl0-v2" "$QUEUE") > "$art/logs/queue.log" 2>&1 || fail "fine-tuning queue"
grep -q "FINETUNE_QUEUE_DONE" "$art/logs/queue.log" || fail "fine-tuning queue did not finish"
"$PY" - "$art/seed42" <<'EOF' || fail "quantized evaluations are not ternary"
import json
import sys
from pathlib import Path

seed = Path(sys.argv[1])
files = [seed / arm / f"finetune-{task}" / "eval" / "results.json" for arm in ("c_ternary_a8", "qap_ternary_a8")
         for task in ("tuar", "tusl0-v2")] + [seed / "ptq_ternary_a8" / "sweep-tuar" / "results.json",
                                              seed / "ptq_ternary_a8" / "sweep-tusl0-v2" / "test" / "results.json"]


def values(node, key):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from values(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from values(v, key)


bad = []
for path in files:
    result = json.loads(path.read_text())
    schemes = set(values(result, "weight_quant_scheme"))
    weight_bits = {b for widths in values(result, "actual_proxy_bit_widths") for b in widths["weight"].values()}
    print(path.relative_to(seed), "schemes", sorted(schemes), "weight quantizer bits", sorted(weight_bits))
    if schemes != {"ternary"} or weight_bits - {2}:
        bad.append(str(path))
sys.exit(1 if bad else 0)
EOF
echo "{\"passed\": \"$(date '+%F %T')\"}" > "$OUT/PASSED"
echo "[$(date '+%F %T')] SMOKE_PASSED"
