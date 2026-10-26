"""TUSL as an event-level three-class task with subject-level folds.

Protocol (2026-10-25): the sampling unit is the annotated 10 s event from the
single-event files ``<recording>_NN.csv`` of TUSL v2.0.1 (bckg / seiz / slow,
100 each). Recordings are preprocessed exactly like the upstream pipeline
(0.1-75 Hz band-pass, 60 Hz notch, 256 Hz, 22-channel TCP bipolar); every event
is cut into two 5 s windows aligned to the event start. Events of one
recording whose intervals overlap with a different label are both dropped and
listed. Folds rotate over the subjects that are absent from the TUAB train
split (the pretraining corpus): four groups; fold k tests on group k, validates
on group k+1 and trains on the other two groups plus every subject inside the
pretraining corpus. Each fold is written as ``fold<k>/{train,val,test}.h5`` in
the upstream group layout with a ``source_file`` dataset naming
``<event>_w<0|1>``.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

LABELS = {"bckg": 0, "seiz": 1, "slow": 2}
SAMPLE_RATE, WINDOW, WINDOWS_PER_EVENT = 256, 1280, 2


def subject_of(name: str) -> str:
    return name.split("_", 1)[0]


def read_event(path: Path) -> dict:
    rows = [r for r in csv.DictReader(l for l in path.read_text().splitlines() if not l.startswith("#"))]
    spans = {(float(r["start_time"]), float(r["stop_time"]), r["label"]) for r in rows}
    if len(spans) != 1:
        raise ValueError(f"{path.name}: channels disagree on the event span or label: {spans}")
    start, stop, label = spans.pop()
    if label not in LABELS or abs((stop - start) - 10.0) > 1e-6:
        raise ValueError(f"{path.name}: unexpected event {label} {start}-{stop}")
    return {"event": path.stem, "recording": path.stem.rsplit("_", 1)[0], "subject": subject_of(path.name),
            "start": start, "stop": stop, "label": label, "file": str(path)}


def drop_conflicts(events: list[dict]) -> tuple[list[dict], list[dict]]:
    by_recording = defaultdict(list)
    for e in events:
        by_recording[e["recording"]].append(e)
    dropped = set()
    for group in by_recording.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a["start"] < b["stop"] and b["start"] < a["stop"] and a["label"] != b["label"]:
                    dropped.update({a["event"], b["event"]})
    kept = [e for e in events if e["event"] not in dropped]
    return kept, [e for e in events if e["event"] in dropped]


def preprocess(upstream, edf: Path) -> np.ndarray:
    """Upstream common preprocessing; returns [22, T] float32 at 256 Hz."""
    import mne
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    if any("LE" in ch for ch in raw.ch_names):
        raw.rename_channels(lambda x: x.replace("LE", "REF"))
    raw.pick_channels(upstream.CH_ORDER_STANDARD, ordered=True)
    if raw.ch_names != upstream.CH_ORDER_STANDARD:
        raise ValueError("Channel order mismatch after picking.")
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
    raw.notch_filter(60, verbose=False)
    if int(raw.info["sfreq"]) != SAMPLE_RATE:
        raw.resample(SAMPLE_RATE, npad="auto", n_jobs=1, verbose=False)
    data, channels = upstream.make_bipolar(raw)
    if len(channels) != 22:
        raise ValueError(f"{edf.name}: {len(channels)} bipolar channels, need 22")
    return data.astype(np.float32)


def make_folds(subjects: list[str], pretraining_subjects: set[str], seed: int, groups: int,
               labels_per_subject: dict[str, Counter]) -> dict:
    """Stratified subject groups: shuffled subjects are placed one by one into the group that currently
    holds the fewest events of the subject's dominant label (ties: fewer subjects, then lower index)."""
    pool = sorted(s for s in subjects if s not in pretraining_subjects)
    inside = sorted(s for s in subjects if s in pretraining_subjects)
    np.random.default_rng(seed).shuffle(pool)
    chunks, counts = [[] for _ in range(groups)], [Counter() for _ in range(groups)]
    for subject in pool:
        dominant = labels_per_subject[subject].most_common(1)[0][0]
        g = min(range(groups), key=lambda i: (counts[i][dominant], len(chunks[i]), i))
        chunks[g].append(subject)
        counts[g].update(labels_per_subject[subject])
    folds = []
    for k in range(groups):
        test, val = chunks[k], chunks[(k + 1) % groups]
        train = inside + [s for j, c in enumerate(chunks) if j not in (k, (k + 1) % groups) for s in c]
        folds.append({"fold": k, "train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return {"seed": seed, "groups": groups, "rule": "stratified by dominant event label", "test_pool": pool,
            "groups_subjects": chunks, "always_train": inside, "folds": folds}


def write_split(path: Path, items: list[tuple[str, int, np.ndarray]], group_size: int = 1000) -> dict:
    with h5py.File(path, "w") as h5f:
        for g, start in enumerate(range(0, len(items), group_size)):
            chunk = items[start:start + group_size]
            grp = h5f.create_group(f"data_group_{g}")
            grp.create_dataset("X", data=np.stack([x for _, _, x in chunk]))
            grp.create_dataset("y", data=np.asarray([y for _, y, _ in chunk], dtype=np.int64))
            grp.create_dataset("source_file", data=np.asarray([n for n, _, _ in chunk], dtype=object),
                               dtype=h5py.string_dtype(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = Counter(y for _, y, _ in items)
    return {"file": path.name, "windows": len(items), "events": len(items) // WINDOWS_PER_EVENT,
            "label_counts": {str(k): counts.get(k, 0) for k in range(3)}, "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--tusl", type=Path, required=True, help="TUSL .../edf directory")
    parser.add_argument("--tuab-split", type=Path, required=True, help="tuab-split.json; its train split is the pretraining corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream / "make_datasets"))
    import process_raw_eeg as upstream  # noqa: E402
    args.output.mkdir(parents=True, exist_ok=True)

    events = sorted((read_event(p) for p in args.tusl.rglob("*_[0-9][0-9].csv")), key=lambda e: e["event"])
    kept, dropped = drop_conflicts(events)
    assert len(events) == 300 and len(kept) == 296, (len(events), len(kept))
    assert Counter(e["label"] for e in kept) == {"bckg": 98, "seiz": 100, "slow": 98}  # events; 2 windows each
    (args.output / "exclusions.json").write_text(json.dumps({"rule": "events overlapping a different label on the same recording",
                                                             "dropped": dropped}, indent=2) + "\n")

    windows_by_event, failures = {}, []
    for recording, group in sorted(defaultdict(list, {e["recording"]: [x for x in kept if x["recording"] == e["recording"]]
                                                       for e in kept}).items()):
        edf = next(args.tusl.rglob(f"{recording}.edf"))
        try:
            data = preprocess(upstream, edf)
        except Exception as error:  # recorded, not silently skipped
            failures.append({"recording": recording, "error": f"{type(error).__name__}: {error}"})
            continue
        for e in group:
            first = int(round(e["start"] * SAMPLE_RATE))
            if first + WINDOWS_PER_EVENT * WINDOW > data.shape[1]:
                failures.append({"recording": recording, "event": e["event"], "error": "event runs past the recording"})
                continue
            windows_by_event[e["event"]] = [data[:, first + w * WINDOW:first + (w + 1) * WINDOW] for w in range(WINDOWS_PER_EVENT)]
        print(f"PROCESSED {recording}: {len(group)} events", flush=True)
    usable = [e for e in kept if e["event"] in windows_by_event]
    subjects = sorted({e["subject"] for e in usable})
    pretraining = set(json.loads(args.tuab_split.read_text())["subjects"]["train"])
    labels_per_subject = defaultdict(Counter)
    for e in usable:
        labels_per_subject[e["subject"]][e["label"]] += 1
    folds = make_folds(subjects, pretraining, args.seed, args.groups, labels_per_subject)

    summary = {"events_total": 300, "events_kept": len(kept), "events_usable": len(usable), "failures": failures,
               "subjects": len(subjects), "test_pool": len(folds["test_pool"]), "folds": []}
    for fold in folds["folds"]:
        fold_dir = args.output / f"fold{fold['fold']}"
        fold_dir.mkdir(exist_ok=True)
        record = {"fold": fold["fold"]}
        for part in ("train", "val", "test"):
            chosen = [e for e in usable if e["subject"] in set(fold[part])]
            items = [(f"{e['event']}_w{w}", LABELS[e["label"]], windows_by_event[e["event"]][w])
                     for e in chosen for w in range(WINDOWS_PER_EVENT)]
            record[part] = write_split(fold_dir / f"{part}.h5", items)
            record[part]["subjects"] = len(fold[part])
            if part != "train" and min(record[part]["label_counts"].values()) == 0:
                raise ValueError(f"fold {fold['fold']} {part} lacks a class: {record[part]['label_counts']}")
        (fold_dir / "data-manifest.json").write_text(json.dumps({"corpus": "TUSL-events", "fold": fold["fold"],
            "num_classes": 3, "classification_type": "mcc", "windows_per_event": WINDOWS_PER_EVENT,
            "data": {p: {"bytes": record[p]["bytes"], "sha256": record[p]["sha256"]} for p in ("train", "val", "test")},
            "splits": {p: {k: record[p][k] for k in ("windows", "bytes", "sha256")} for p in ("train", "val", "test")}}, indent=2) + "\n")
        summary["folds"].append(record)
    (args.output / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TUSL_EVENTS_COMPLETE", json.dumps({k: summary[k] for k in ("events_kept", "events_usable", "subjects", "test_pool")}),
          "failures", len(failures))
    for record in summary["folds"]:
        print(f"fold {record['fold']}: " + " | ".join(f"{p} {record[p]['events']} ev {record[p]['label_counts']}" for p in ("train", "val", "test")))


if __name__ == "__main__":
    main()
