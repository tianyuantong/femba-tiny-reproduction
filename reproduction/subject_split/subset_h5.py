"""Subject-level subsample of an HDF5 split built by preprocess_by_split.py.

Subjects are read from the ``source_file`` names, sorted, shuffled with
``numpy.random.default_rng(seed)`` and the first ``ceil(fraction * n)`` are
kept. Windows of the kept subjects are copied in their original order into a
new file with the same group layout (``X``, ``y``, ``source_file`` per
``data_group_N``), so fine-tuning on a fraction of the training subjects uses
the same files and loaders as the full split.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def subject_of(name: str) -> str:
    return name.split("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=1000)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if args.target.exists():
        raise FileExistsError(args.target)

    with h5py.File(args.source, "r") as src:
        groups = sorted(src.keys(), key=lambda g: int(g.rsplit("_", 1)[1]))
        names = [n.decode() for g in groups for n in src[g]["source_file"][:]]
        subjects = sorted({subject_of(n) for n in names})
        np.random.default_rng(args.seed).shuffle(subjects)
        kept = set(subjects[:math.ceil(args.fraction * len(subjects))])
        labels, buffer_x, buffer_y, buffer_n = [], [], [], []
        with h5py.File(args.target, "w") as dst:
            dst.attrs.update(dict(src.attrs))
            dst.attrs.update({"subset_of": args.source.name, "subset_fraction": args.fraction,
                              "subset_seed": args.seed, "subset_unit": "subject"})

            def flush() -> None:
                if not buffer_n:
                    return
                group = dst.create_group(f"data_group_{len(dst.keys())}")
                group.create_dataset("X", data=np.stack(buffer_x))
                group.create_dataset("y", data=np.asarray(buffer_y, dtype=np.int64))
                group.create_dataset("source_file", data=np.asarray(buffer_n, dtype=object),
                                     dtype=h5py.string_dtype(encoding="utf-8"))
                buffer_x.clear(), buffer_y.clear(), buffer_n.clear()

            for g in groups:
                x, y = src[g]["X"], src[g]["y"][:]
                for i, name in enumerate(src[g]["source_file"][:]):
                    name = name.decode()
                    if subject_of(name) not in kept:
                        continue
                    buffer_x.append(x[i]), buffer_y.append(int(y[i])), buffer_n.append(name)
                    labels.append(int(y[i]))
                    if len(buffer_n) == args.group_size:
                        flush()
            flush()
    digest = hashlib.sha256()
    with args.target.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    summary = {"source": str(args.source), "file": args.target.name, "fraction": args.fraction, "seed": args.seed,
               "subjects_total": len(subjects), "subjects_kept": len(kept), "windows": len(labels),
               "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
               "kept_subjects": sorted(kept), "sha256": digest.hexdigest()}
    args.target.with_name(f"{args.target.stem}-manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUBSET_COMPLETE", json.dumps({k: summary[k] for k in ("file", "subjects_kept", "windows", "label_counts")}))


if __name__ == "__main__":
    main()
