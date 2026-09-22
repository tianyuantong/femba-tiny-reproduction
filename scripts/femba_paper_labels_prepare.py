"""Prepare a 13-artifact, any-coverage label control without changing EEG inputs.

CSV rounding, channel checks and the parent 18-artifact >=30% rule reproduce
process_raw_eeg.py at d885965. The label-only audit was first implemented in
femba_fp32_label_audit.py; this script adds a verified pickle-to-HDF5 mapping.
Run only on the project's trusted, locally generated pickle artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np


UPSTREAM_COMMIT = "d88596590f3bd3fce573be07646b7d3977ce7bcc"
PROCESSOR_SHA256 = "5ba11aa9f06db19d7c821e4507dc3656391bbbcf362c9b9c39a4d15e1901728b"
SAMPLE_RATE = 256
WINDOW_SAMPLES = 1280
PREFIX = "upstream2025_"
SPLITS = ("train", "val", "test")
ARTIFACT_LABELS = frozenset((
    "chew", "chew_elec", "chew_musc", "elec", "eyem", "eyem_chew",
    "eyem_elec", "eyem_musc", "eyem_shiv", "musc", "musc_elec", "shiv", "shiv_elec",
))
EXTRA_LABELS = frozenset(("cpsz", "elpp", "fnsz", "gnsz", "tcsz"))
CHANNELS = frozenset((
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6",
    "T6-O2", "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
))
LABEL_RULE = {
    "name": "paper13_any", "labels": sorted(ARTIFACT_LABELS),
    "interpretation": "Paper 13-class any-artifact semantics with upstream 256Hz rounded, half-open sample intervals and complete 5s windows",
    "coverage": "union_over_accepted_channels", "comparison": ">",
    "artifact_fraction_threshold": 0.0, "sample_rate": SAMPLE_RATE,
    "window_samples": WINDOW_SAMPLES, "time_rounding": "Python round ties-to-even",
    "accepted_channels": sorted(CHANNELS),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signal_sha256(signal: np.ndarray) -> str:
    """Hash shape, exact dtype and all signal bytes; reject invalid EEG arrays."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array")
    if signal.shape != (22, WINDOW_SAMPLES) or signal.dtype.kind != "f":
        raise ValueError("Signal must be a floating [22,1280] array")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains nonfinite values")
    metadata = {"dtype": signal.dtype.str, "shape": list(signal.shape)}
    digest = hashlib.sha256(_canonical(metadata) + b"\n")
    digest.update(np.ascontiguousarray(signal).tobytes())
    return digest.hexdigest()


def _binary_label(value: Any) -> int:
    label = np.asarray(value)
    if label.ndim or label.dtype.kind not in "biu" or int(label) not in (0, 1):
        raise ValueError("Expected a scalar binary integer label")
    return int(label)


def labels_from_csv(path: Path, windows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return parent 18-class >=30% and candidate 13-class any labels."""
    if not isinstance(windows, int) or windows <= 0:
        raise ValueError("Window count must be a positive integer")
    old_mask = np.zeros(windows * WINDOW_SAMPLES, dtype=np.bool_)
    new_mask = np.zeros_like(old_mask)
    with path.open(newline="") as handle:
        rows = csv.DictReader(line for line in handle if line.strip() and not line.startswith("#"))
        if not {"channel", "start_time", "stop_time", "label"}.issubset(rows.fieldnames or ()):
            raise ValueError("Annotation CSV is missing required columns")
        for row in rows:
            start, stop = float(row["start_time"]), float(row["stop_time"])
            if not math.isfinite(start) or not math.isfinite(stop) or stop < 0 or stop < start:
                raise ValueError("Invalid annotation time range")
            first = max(int(round(start * SAMPLE_RATE)), 0)
            last = min(int(round(stop * SAMPLE_RATE)), len(old_mask))
            if row["channel"] not in CHANNELS or first >= last:
                continue
            if row["label"] in ARTIFACT_LABELS | EXTRA_LABELS:
                old_mask[first:last] = True
            if row["label"] in ARTIFACT_LABELS:
                new_mask[first:last] = True
    old = old_mask.reshape(windows, WINDOW_SAMPLES).mean(axis=1) >= 0.3
    new = new_mask.reshape(windows, WINDOW_SAMPLES).any(axis=1)
    return old.astype(np.int64), new.astype(np.int64)


def _read_parent(path: Path) -> dict:
    parent = json.loads(path.read_text())
    expected = {"status": "complete", "upstream_commit": UPSTREAM_COMMIT,
                "source_sha256": PROCESSOR_SHA256, "artifact_fraction_threshold": 0.3}
    if any(parent.get(key) != value for key, value in expected.items()):
        raise ValueError("Parent manifest does not identify the verified 2025 label protocol")
    if not isinstance(parent.get("records"), list) or not parent["records"]:
        raise ValueError("Parent manifest must contain source records")
    for split in SPLITS:
        counts = parent["outputs"][split]
        if not all(type(counts.get(key)) is int for key in ("windows", "positive")):
            raise ValueError("Invalid parent window or positive count")
        if not 0 <= counts["positive"] <= counts["windows"] or not counts["windows"]:
            raise ValueError("Invalid parent count range")
        if not Path(counts["path"]).is_file():
            raise ValueError("Missing parent HDF5 file")
    return parent


def _record_windows(parent: dict, processed: Path) -> dict:
    records = {}
    for number, record in enumerate(parent["records"]):
        source, split = Path(record["source"]), record["split"]
        if split not in SPLITS or not source.is_file() or not source.with_suffix(".csv").is_file():
            raise ValueError(f"Missing/invalid source at manifest index {number}")
        stem = source.name.split(".")[0]
        if stem in records:
            raise ValueError("Duplicate source basename makes window mapping ambiguous")
        records[stem] = {"source": source, "split": split, "windows": {}}
    for split in SPLITS:
        folder = processed / split
        if not folder.is_dir():
            raise ValueError(f"Missing processed directory for {split}")
        for path in folder.glob(PREFIX + "*.pkl"):
            stem, text = path.stem[len(PREFIX):].rsplit("_", 1)
            if stem not in records or records[stem]["split"] != split or not text.isdigit():
                raise ValueError("Unmapped, misplaced or invalid pickle filename")
            index = int(text)
            if index in records[stem]["windows"]:
                raise ValueError("Duplicate pickle window index")
            records[stem]["windows"][index] = path
    for record in records.values():
        indices = record["windows"]
        if not indices or set(indices) != set(range(len(indices))):
            raise ValueError("Source has missing or noncontinuous pickle windows")
    return records


@dataclass
class _SignalEntry:
    old_label: int
    new_label: int
    source_ids: list[str]


def _add_signal(index: dict, digest: str, old: int, new: int, source_id: str) -> None:
    existing = index.get(digest)
    if existing is None:
        index[digest] = _SignalEntry(old, new, [source_id])
    elif (existing.old_label, existing.new_label) != (old, new):
        raise ValueError("Duplicate signal has ambiguous parent/candidate labels")
    else:
        existing.source_ids.append(source_id)


def _index_pickles(records: dict, split: str, expected: dict) -> tuple[dict, dict]:
    index: dict[str, _SignalEntry] = {}
    windows = old_positive = new_positive = 0
    for record in records.values():
        if record["split"] != split:
            continue
        old, new = labels_from_csv(record["source"].with_suffix(".csv"), len(record["windows"]))
        for number, path in sorted(record["windows"].items()):
            with path.open("rb") as handle:
                sample = pickle.load(handle)
            if not isinstance(sample, dict) or not {"X", "y"}.issubset(sample):
                raise ValueError("Invalid source pickle schema")
            original = _binary_label(sample["y"])
            if original != int(old[number]):
                raise ValueError("CSV reconstruction disagrees with an original pickle label")
            source_id = hashlib.sha256(f"{split}/{path.name}".encode()).hexdigest()
            _add_signal(index, signal_sha256(sample["X"]), original, int(new[number]), source_id)
        windows += len(old)
        old_positive += int(old.sum())
        new_positive += int(new.sum())
    if (windows, old_positive) != (expected["windows"], expected["positive"]):
        raise ValueError(f"Reconstructed parent counts do not match manifest for {split}")
    counts = {"windows": windows, "positive": new_positive,
              "duplicate_signal_rows": windows - len(index)}
    return index, counts


def _consume_signal(index: dict, signal: np.ndarray, label: Any) -> tuple[str, str, int]:
    digest = signal_sha256(signal)
    entry = index.get(digest)
    if entry is None or not entry.source_ids:
        raise ValueError("HDF5 signal has no unused matching pickle")
    if _binary_label(label) != entry.old_label:
        raise ValueError("HDF5 label disagrees with its matching pickle")
    return digest, entry.source_ids.pop(), entry.new_label


def _map_group(group: Any, index: dict, mapping: Any, digests: dict) -> tuple[np.ndarray, np.ndarray]:
    if set(group.keys()) != {"X", "y"}:
        raise ValueError("Unexpected parent HDF5 group schema")
    x, old = group["X"], group["y"][:]
    if x.shape != (len(old), 22, WINDOW_SAMPLES) or old.ndim != 1:
        raise ValueError("Invalid parent HDF5 sample/label shapes")
    new = np.empty_like(old)
    for row in range(len(old)):
        digest, source_id, new[row] = _consume_signal(index, x[row], old[row])
        digests["sample_identity"].update(bytes.fromhex(digest))
        item = {"group": group.name[1:], "row": row, "signal_sha256": digest,
                "pickle_identity_sha256": source_id}
        mapping.write(json.dumps(item, sort_keys=True) + "\n")
    digests["old_labels"].update(old.astype("<i8").tobytes())
    digests["new_labels"].update(new.astype("<i8").tobytes())
    return old, new


def _write_hdf5(source: Path, target: Path, index: dict, mapping_path: Path) -> dict:
    import h5py

    digests = {name: hashlib.sha256() for name in ("sample_identity", "old_labels", "new_labels")}
    transitions = np.zeros((2, 2), dtype=np.int64)
    initial_stat = source.stat()
    with h5py.File(source, "r") as parent, h5py.File(target, "x", track_order=True) as output:
        output.attrs.update(dict(parent.attrs))
        with mapping_path.open("x") as mapping:
            for name in parent.keys():
                old, new = _map_group(parent[name], index, mapping, digests)
                group = output.create_group(name)
                group.attrs.update(dict(parent[name].attrs))
                group["X"] = h5py.ExternalLink(os.path.relpath(source, target.parent), f"/{name}/X")
                labels = group.create_dataset("y", data=new)
                labels.attrs.update(dict(parent[name]["y"].attrs))
                transitions += np.bincount(old.astype(np.int64) * 2 + new, minlength=4).reshape(2, 2)
        if list(output.keys()) != list(parent.keys()):
            raise ValueError("New HDF5 loader group order differs from parent")
    if any(entry.source_ids for entry in index.values()):
        raise ValueError("Some original pickle windows were not consumed by HDF5 mapping")
    source_hash = file_sha256(source)
    final_stat = source.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Parent HDF5 changed during preparation")
    return {"source_h5": {"path": str(source), "sha256": source_hash},
            "sha256": file_sha256(target), "path": str(target),
            "mapping_sha256": file_sha256(mapping_path),
            "changed_labels": int(transitions[0, 1] + transitions[1, 0]),
            "transition_matrix": transitions.tolist(),
            **{name + "_sha256": digest.hexdigest() for name, digest in digests.items()}}


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def prepare(parent_manifest: Path, processed: Path, output_root: Path) -> dict:
    """Create new artifacts exclusively; parent EEG/labels stay read-only."""
    import h5py  # Fail before creating output directories if unavailable.

    parent_manifest, processed, output_root = (path.resolve() for path in (parent_manifest, processed, output_root))
    if output_root.exists():
        raise FileExistsError("Output root must not exist; previous evidence is preserved")
    parent_digest = file_sha256(parent_manifest)
    parent = _read_parent(parent_manifest)
    records = _record_windows(parent, processed)
    output_root.mkdir(parents=True, exist_ok=False)
    results = output_root / "results"
    results.mkdir()
    try:
        return _prepare_outputs(parent_manifest, parent, records, output_root, parent_digest)
    except Exception as error:
        _write_json(results / "preparation-failed.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise


def _prepare_outputs(parent_path: Path, parent: dict, records: dict, root: Path, parent_digest: str) -> dict:
    results, data = root / "results", root / "data/TUAR_data"
    data.mkdir(parents=True)
    snapshot = results / "parent-upstream2025-data.json"
    with snapshot.open("xb") as handle:
        handle.write(parent_path.read_bytes())
    if file_sha256(snapshot) != parent_digest:
        raise ValueError("Parent manifest changed before its immutable snapshot")
    script_snapshot = results / f"prepare-source-{file_sha256(Path(__file__))}.py"
    with script_snapshot.open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    outputs = {}
    for split in SPLITS:
        print(f"PAPER_LABEL_PREP_START {split}", flush=True)
        index, counts = _index_pickles(records, split, parent["outputs"][split])
        mapping = results / f"{split}-label-mapping.jsonl"
        source = Path(parent["outputs"][split]["path"]).resolve()
        outputs[split] = {**counts, **_write_hdf5(source, data / f"{PREFIX}{split}.h5", index, mapping),
                          "mapping_file": str(mapping.relative_to(root))}
        assert sum(map(sum, outputs[split]["transition_matrix"])) == counts["windows"]
        print(f"PAPER_LABEL_PREP_DONE {split} {counts['windows']} {counts['positive']}", flush=True)
    manifest = _candidate_manifest(parent, parent_path, snapshot, script_snapshot, outputs)
    if file_sha256(parent_path) != manifest["parent_manifest"]["sha256"]:
        raise ValueError("Parent manifest changed during preparation")
    _write_json(results / "upstream2025-data.json", manifest)
    return manifest


def _candidate_manifest(parent: dict, parent_path: Path, snapshot: Path, script: Path, outputs: dict) -> dict:
    return {
        "status": "complete", "upstream_commit": UPSTREAM_COMMIT,
        "seed": parent["seed"], "split_unit": parent["split_unit"], "records": parent["records"],
        "artifact_fraction_threshold": 0.0, "artifact_fraction_comparison": ">",
        "label_rule": LABEL_RULE, "label_rule_sha256": hashlib.sha256(_canonical(LABEL_RULE)).hexdigest(),
        "source_sha256": file_sha256(script), "source_snapshot": "results/" + script.name,
        "parent_manifest": {"path": str(parent_path), "sha256": file_sha256(snapshot),
                            "snapshot": "results/" + snapshot.name},
        "parent_processor_sha256": PROCESSOR_SHA256,
        "input_identity": "Exact dtype/shape/signal SHA256 and original label; full multiset consumed in HDF5 loader order",
        "duplicate_identity_policy": "Equal signals require equal old and new labels; otherwise reject; same-label duplicate occurrences are interchangeable",
        "storage": "X uses relative HDF5 ExternalLink to parent; y is local. Parent HDF5 files must remain unchanged and accessible.",
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent_manifest, args.processed, args.output_root)
    print("PAPER_LABEL_DATA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
