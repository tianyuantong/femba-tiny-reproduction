"""Build subject-independent HDF5 splits for TUAR / TUAB from a subject manifest.

Recordings listed in ``<corpus>-split.json`` (from ``make_subject_splits.py``)
are processed with the pristine upstream
``make_datasets/process_raw_eeg.process_and_dump_file`` (0.1-75 Hz band-pass,
60 Hz notch, 256 Hz, 22-channel TCP bipolar, 5 s windows, upstream labels) in
chunks of recordings. After every chunk the windows are appended to the split's
HDF5 file, which keeps the upstream group layout (``data_group_N`` with ``X``
and ``y``) and adds a ``source_file`` dataset per group, and the chunk's
pickles are deleted. ``--exclude-subjects`` drops the listed subjects, e.g. the
test subjects of the other corpora when the split feeds pretraining.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def load_upstream(upstream: Path):
    sys.path.insert(0, str(upstream / "make_datasets"))
    import process_raw_eeg  # noqa: E402  (upstream module, unchanged)
    return process_raw_eeg


def tuab_label(relative_path: str) -> int:
    parts = Path(relative_path).parts
    if "abnormal" in parts:
        return 1
    if "normal" in parts:
        return 0
    raise ValueError(f"Cannot infer TUAB label from path: {relative_path}")


def subject_of(name: str) -> str:
    return Path(name).name.split("_", 1)[0]


def append_pickles(h5f: h5py.File, source: Path, dtype: str, group_size: int) -> tuple[list[int], set[str]]:
    files = sorted(f for f in os.listdir(source) if f.endswith(".pkl"))
    labels, subjects = [], set()
    for start in range(0, len(files), group_size):
        chunk = files[start:start + group_size]
        samples = []
        for name in chunk:
            with open(source / name, "rb") as handle:
                samples.append(pickle.load(handle))
        group = h5f.create_group(f"data_group_{len(h5f.keys())}")
        group.create_dataset("X", data=np.stack([s["X"] for s in samples]).astype(dtype))
        y = np.asarray([s["y"] for s in samples], dtype=np.int64)
        group.create_dataset("y", data=y)
        group.create_dataset("source_file", data=np.asarray(chunk, dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))
        labels.extend(y.tolist())
        subjects.update(subject_of(n) for n in chunk)
    return labels, subjects


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args) -> None:
    upstream = load_upstream(args.upstream)
    split = json.loads(args.split_json.read_text())
    root, corpus = Path(split["root"]), split["corpus"].lower()
    excluded = set(json.loads(args.exclude_subjects.read_text())["subjects"]) if args.exclude_subjects else set()
    files = [f for f in split["files"][args.split] if subject_of(f) not in excluded]
    dropped = sorted({subject_of(f) for f in split["files"][args.split]} & excluded)
    target = args.output / f"{args.name or args.split}.h5"
    if target.exists():
        raise FileExistsError(target)
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = args.output / f"scratch-{target.stem}"
    labels, subjects, errors = [], set(), []
    with h5py.File(target, "w") as h5f, multiprocessing.Pool(args.processes) as pool:
        h5f.attrs.update({"corpus": split["corpus"], "split": args.split, "split_seed": split["seed"],
                          "split_unit": "subject", "dtype": args.dtype, "group_size": args.group_size,
                          "excluded_subjects": len(dropped), "source_order": "lexicographic pickle filename per chunk"})
        for start in range(0, len(files), args.chunk_files):
            chunk = files[start:start + args.chunk_files]
            shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir()
            params = [(str(root / f), str(scratch), tuab_label(f) if corpus == "tuab" else args.mode, corpus)
                      for f in chunk]
            errors.extend(r for r in pool.imap_unordered(upstream.process_and_dump_file, params) if r is not None)
            chunk_labels, chunk_subjects = append_pickles(h5f, scratch, args.dtype, args.group_size)
            labels.extend(chunk_labels)
            subjects |= chunk_subjects
            print(f"CHUNK {min(start + args.chunk_files, len(files))}/{len(files)} recordings, "
                  f"{len(labels)} windows, errors={len(errors)}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    counts = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    summary = {"corpus": split["corpus"], "split": args.split, "file": target.name, "recordings": len(files),
               "windows": len(labels), "subjects": len(subjects), "label_counts": counts,
               "excluded_subjects": dropped, "errors": errors, "dtype": args.dtype, "sha256": sha256(target)}
    (args.output / f"{target.stem}-manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BUILD_COMPLETE", json.dumps({k: summary[k] for k in ("file", "recordings", "windows", "subjects", "label_counts")}),
          f"errors={len(errors)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None, help="output stem, default = split name")
    parser.add_argument("--exclude-subjects", type=Path, default=None)
    parser.add_argument("--mode", default="Binary", help="TUAR label mode passed to upstream")
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--chunk-files", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=1000)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
