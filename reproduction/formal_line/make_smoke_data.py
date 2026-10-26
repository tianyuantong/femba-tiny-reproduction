"""Tiny copy of a formal-line data directory for the ternary smoke test.

Per split, the first windows of each class in file order (train 1024, val 256, test 256,
split evenly over the classes of the split manifest) are written in the same HDF5 layout,
with the manifests the runners read: data-manifest.json and <split>-manifest.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

WINDOWS = {"train": 1024, "val": 256, "test": 256}


def take(path: Path, quota: dict[int, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, names, left = [], [], [], dict(quota)
    with h5py.File(path, "r") as h5f:
        for group in sorted(h5f.keys(), key=lambda g: int(g.rsplit("_", 1)[1])):
            labels = h5f[group]["y"][:]
            rows = []
            for i, label in enumerate(labels):
                if left.get(int(label), 0) > 0:
                    rows.append(i)
                    left[int(label)] -= 1
            if rows:
                xs.append(h5f[group]["X"][rows])
                ys.append(labels[rows])
                names.extend(n.decode() for n in h5f[group]["source_file"][rows])
            if not any(left.values()):
                break
    if any(left.values()):
        raise ValueError(f"{path}: too few windows, still missing {left}")
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64), names


def write(path: Path, x: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("data_group_0")
        group.create_dataset("X", data=x)
        group.create_dataset("y", data=y)
        group.create_dataset("source_file", data=np.asarray(names, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
    return {"windows": len(y), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "label_counts": {str(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, total in WINDOWS.items():
        classes = sorted(json.loads((args.source / f"{split}-manifest.json").read_text())["label_counts"])
        quota = {int(c): total // len(classes) for c in classes}
        splits[split] = write(args.output / f"{split}.h5", *take(args.source / f"{split}.h5", quota))
        (args.output / f"{split}-manifest.json").write_text(json.dumps({"split": split, "file": f"{split}.h5", **splits[split]}, indent=2) + "\n")
    (args.output / "data-manifest.json").write_text(json.dumps({"corpus": f"smoke subset of {args.source}",
        "data": {s: {"bytes": r["bytes"], "sha256": r["sha256"]} for s, r in splits.items()},
        "splits": {s: {k: r[k] for k in ("windows", "bytes", "sha256")} for s, r in splits.items()}}, indent=2) + "\n")
    print("SMOKE_DATA_COMPLETE", args.output, {s: r["label_counts"] for s, r in splits.items()})


if __name__ == "__main__":
    main()
