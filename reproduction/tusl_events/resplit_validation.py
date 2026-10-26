"""Protocol v2 validation sets for the TUSL event folds.

Revision of 2026-10-26, made after the protocol-v1 test results had been seen: fold k
still tests on group k and every subject inside the pretraining corpus still trains, but
validation no longer takes the whole next group. Candidates are the subjects of the three
other groups with at most 15 events, sorted and shuffled with the split seed; each is taken
while it adds events to a class that is not yet covered (at least 10 events from at least 3
subjects, the largest subject at most half of the class), and the rest train. Constraints
that cannot be met are an error. test.h5 is copied byte for byte from protocol v1; train.h5
and val.h5 are rebuilt from the v1 windows, so no recording is preprocessed again.
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from make_tusl_events import WINDOWS_PER_EVENT, subject_of, write_split

MAX_SUBJECT_EVENTS = 15
MIN_CLASS_EVENTS, MIN_CLASS_SUBJECTS, MAX_CLASS_SHARE = 10, 3, 0.5
NUM_CLASSES = 3


def read_windows(fold_dir: Path) -> dict[str, tuple[int, np.ndarray]]:
    """Every window of one v1 fold (its train, val and test hold each event once): name -> (label, window)."""
    windows = {}
    for part in ("train", "val", "test"):
        with h5py.File(fold_dir / f"{part}.h5", "r") as h5f:
            for group in h5f.values():
                for name, label, x in zip(group["source_file"][:], group["y"][:], group["X"][:]):
                    windows[name.decode()] = (int(label), x)
    return windows


def class_profile(subjects: list[str], events: dict[str, Counter], label: int) -> tuple[int, int, float]:
    """Events of one class among the subjects, how many subjects hold them, and the largest subject's share."""
    counts = [events[s][label] for s in subjects if events[s][label]]
    return sum(counts), len(counts), max(counts) / sum(counts) if counts else 0.0


def class_covered(subjects: list[str], events: dict[str, Counter], label: int) -> bool:
    total, holders, share = class_profile(subjects, events, label)
    return total >= MIN_CLASS_EVENTS and holders >= MIN_CLASS_SUBJECTS and share <= MAX_CLASS_SHARE


def pick_validation(pool: list[str], events: dict[str, Counter], seed: int) -> list[str]:
    candidates = sorted(s for s in pool if sum(events[s].values()) <= MAX_SUBJECT_EVENTS)
    np.random.default_rng(seed).shuffle(candidates)
    chosen = []
    for subject in candidates:
        open_classes = [c for c in range(NUM_CLASSES) if not class_covered(chosen, events, c)]
        if not open_classes:
            break
        if any(events[subject][c] for c in open_classes):
            chosen.append(subject)
    if not all(class_covered(chosen, events, c) for c in range(NUM_CLASSES)):
        raise ValueError(f"validation constraints cannot be met from {len(candidates)} candidates: "
                         f"{[class_profile(chosen, events, c) for c in range(NUM_CLASSES)]}")
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True, help="make_tusl_events.py output: folds.json and fold<k>/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="split seed, shared by every training seed")
    args = parser.parse_args()

    v1 = json.loads((args.v1 / "folds.json").read_text())
    groups, corpus = v1["groups_subjects"], v1["always_train"]
    windows = read_windows(args.v1 / "fold0")
    events = defaultdict(Counter)
    for name, (label, _) in windows.items():
        if name.endswith("_w0"):
            events[subject_of(name)][label] += 1
    assert len(windows) == WINDOWS_PER_EVENT * sum(sum(c.values()) for c in events.values())
    args.output.mkdir(parents=True, exist_ok=False)

    folds, records = [], []
    for k, test in enumerate(groups):
        pool = [s for j, group in enumerate(groups) if j != k for s in group]
        val = pick_validation(pool, events, args.seed)
        train = sorted(corpus + [s for s in pool if s not in val])
        assert set(train) | set(val) | set(test) == set(events) and len(train) + len(val) + len(test) == len(events)
        fold_dir = args.output / f"fold{k}"
        fold_dir.mkdir()
        v1_manifest = json.loads((args.v1 / f"fold{k}" / "data-manifest.json").read_text())
        shutil.copyfile(args.v1 / f"fold{k}" / "test.h5", fold_dir / "test.h5")
        assert hashlib.sha256((fold_dir / "test.h5").read_bytes()).hexdigest() == v1_manifest["data"]["test"]["sha256"]
        record = {"fold": k, "test": dict(v1_manifest["splits"]["test"])}
        for part, subjects in (("train", train), ("val", val)):
            members = set(subjects)
            names = sorted(n for n in windows if subject_of(n) in members)
            record[part] = write_split(fold_dir / f"{part}.h5", [(n, windows[n][0], windows[n][1]) for n in names])
            record[part]["subjects"] = len(subjects)
        record["val"]["classes"] = {str(c): dict(zip(("events", "subjects", "largest_share"), class_profile(val, events, c)))
                                    for c in range(NUM_CLASSES)}
        (fold_dir / "data-manifest.json").write_text(json.dumps({"corpus": "TUSL-events", "protocol": "v2-validation",
            "fold": k, "num_classes": NUM_CLASSES, "classification_type": "mcc", "windows_per_event": WINDOWS_PER_EVENT,
            "data": {p: {"bytes": record[p]["bytes"], "sha256": record[p]["sha256"]} for p in ("train", "val", "test")},
            "splits": {p: {key: record[p][key] for key in ("windows", "bytes", "sha256")} for p in ("train", "val", "test")}},
            indent=2) + "\n")
        folds.append({"fold": k, "train": train, "val": val, "test": sorted(test)})
        records.append(record)
        print(f"fold {k}: train {record['train']['events']} ev | val {record['val']['events']} ev from {len(val)} subjects "
              f"{record['val']['label_counts']} | test {record['test']['windows'] // WINDOWS_PER_EVENT} ev (copied)", flush=True)

    rule = {"max_subject_events": MAX_SUBJECT_EVENTS, "min_class_events": MIN_CLASS_EVENTS,
            "min_class_subjects": MIN_CLASS_SUBJECTS, "max_class_share": MAX_CLASS_SHARE}
    (args.output / "folds.json").write_text(json.dumps({"protocol": "v2-validation", "derived_from": str(args.v1),
        "split_seed": args.seed, "rule": rule, "groups_subjects": groups, "always_train": corpus, "folds": folds}, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"protocol": "v2-validation", "rule": rule, "folds": records}, indent=2) + "\n")
    print("TUSL_V2_COMPLETE")


if __name__ == "__main__":
    main()
