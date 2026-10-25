#!/bin/bash
# Build subject-independent TUAR/TUAB HDF5 files on the WSL host, split by split.
# preprocess_by_split.py works in chunks of recordings and deletes pickles after
# each chunk, so peak disk use is one chunk of pickles plus the HDF5 files.
# TUAB train drops the pretraining exclusion list (test subjects of every corpus).
set -u -o pipefail
PY=${PY:-/root/envs/femba/bin/python}
UPSTREAM=${UPSTREAM:-/root/work/femba-teammate-ptq-next}
SPLITS_DIR=${SPLITS_DIR:-/root/work/femba-artifacts/subject-splits-20261025}
OUT=${OUT:-/root/work/femba-artifacts/subject-data-20261025}
HOST_MOUNT=${HOST_MOUNT:-/mnt/d}      # Windows drive holding ext4.vhdx
MIN_FREE_GB=${MIN_FREE_GB:-120}
STUDY=$(cd "$(dirname "$0")" && pwd)

free_gb() { df --output=avail -B 1G "$1" 2>/dev/null | tail -1 | tr -d ' '; }
guard() {
  for mount in / "$HOST_MOUNT"; do
    avail=$(free_gb "$mount"); [ -z "$avail" ] && continue
    if [ "$avail" -lt "$MIN_FREE_GB" ]; then
      echo "[$(date '+%F %T')] STOP: $mount has ${avail} GB free (< ${MIN_FREE_GB})"; exit 3
    fi
  done
}

for corpus in ${CORPORA:-tuar tuab}; do
  for split in train val test; do
    guard
    extra=()
    [ "$corpus" = tuab ] && [ "$split" = train ] && extra=(--exclude-subjects "$SPLITS_DIR/pretrain-exclusion.json")
    echo "[$(date '+%F %T')] build $corpus $split ${extra[*]:-}"
    "$PY" -I "$STUDY/preprocess_by_split.py" --upstream "$UPSTREAM" \
      --split-json "$SPLITS_DIR/$corpus-split.json" --split "$split" --output "$OUT/$corpus" "${extra[@]}" \
      2>&1 | grep -v "pick_channels() is a legacy" || exit 4
    echo "[$(date '+%F %T')] $corpus $split done; free / $(free_gb /) GB, $HOST_MOUNT $(free_gb "$HOST_MOUNT") GB"
  done
done
echo "[$(date '+%F %T')] ALL_DONE"
