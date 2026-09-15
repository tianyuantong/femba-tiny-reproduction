"""Run the pinned upstream TUAR processor without replacing prior outputs.

Only IO names are adapted: new pickle/HDF5 files use an upstream2025 prefix
inside existing directories. Filtering, labels and splitting are upstream code.
"""
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import types

import h5py
import numpy as np

REF = "d88596590f3bd3fce573be07646b7d3977ce7bcc"
REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FEMBA_RUN_ROOT", "/root/work/femba-artifacts/tuar-2025-seed42"))
RAW = Path(os.environ.get("FEMBA_RAW_ROOT", "/root/data/tuh_eeg/tuh_eeg_artifact/v3.0.1/edf"))
DATA = ROOT / "data/TUAR_data"
PREFIX = "upstream2025_"


def source(path):
    return subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=REPO, text=True)


def main():
    paths = [DATA / "processed" / split for split in ("train", "val", "test")]
    assert RAW.is_dir() and all(p.is_dir() for p in paths)
    outputs = {s: DATA / f"{PREFIX}{s}.h5" for s in ("train", "val", "test")}
    assert not any(p.exists() for p in outputs.values()), "Prior upstream outputs exist; inspect before resuming"
    assert not any(list(p.glob(PREFIX + "*.pkl")) for p in paths), "Partial preparation exists; do not restart blindly"
    code = source("make_datasets/process_raw_eeg.py")
    packing_code = source("make_datasets/make_hdf5.py")
    processor = types.ModuleType("upstream_processor")
    exec(compile(code, f"{REF}/process_raw_eeg.py", "exec"), processor.__dict__)
    packer = types.ModuleType("upstream_packer")
    exec(compile(packing_code, f"{REF}/make_hdf5.py", "exec"), packer.__dict__)
    errors = []

    def output_open(path, mode="r", *args, **kwargs):
        path = Path(path)
        if mode == "wb" and path.suffix == ".pkl":
            path = path.with_name(PREFIX + path.name)
            mode = "xb"
        elif "a" in mode and "process-errors" in str(path):
            errors.append(str(path))
            path = ROOT / "logs/upstream2025-process-errors.txt"
        return builtins.open(path, mode, *args, **kwargs)

    processor.open = output_open
    # Use the original records and their actual os.listdir order. The upstream
    # function uses numpy's global RNG; seed it explicitly for repeatability.
    np.random.seed(42)
    parameters = processor.get_tuar_parameters(str(RAW), str(DATA), "Binary")
    manifest = {
        "upstream_commit": REF,
        "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "packer_sha256": hashlib.sha256(packing_code.encode()).hexdigest(),
        "seed": 42,
        "split_unit": "record",
        "artifact_fraction_threshold": 0.3,
        "io_adaptation": "prefix new files inside existing directories; originals preserved",
        "records": [{"source": p[0], "split": Path(p[1]).name} for p in parameters],
    }
    manifest_path = ROOT / "results/upstream2025-data.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    # Sequential processing keeps memory bounded and original function behavior.
    for i, params in enumerate(parameters, 1):
        processor.process_and_dump_file(params)
        if errors:
            raise RuntimeError("Upstream processor reported an error; inspect preserved log")
        print(f"UPSTREAM_RECORD {i}/{len(parameters)}", flush=True)

    # The packer remains byte-for-byte upstream. Restrict its directory listing
    # to files from this run so previous labels cannot enter the new dataset.
    packer.os = types.SimpleNamespace(**{name: getattr(os, name) for name in dir(os)})
    packer.os.listdir = lambda path: [n for n in os.listdir(path) if n.startswith(PREFIX)]
    summary = {}
    for split, target in outputs.items():
        folder = DATA / "processed" / split
        expected = len(list(folder.glob(PREFIX + "*.pkl")))
        assert expected > 0
        packer.create_hdf5(str(folder), str(target), finetune=True)
        count, positives = 0, 0
        with h5py.File(target, "r") as h:
            for group in h.values():
                assert group["X"].shape[1:] == (22, 1280)
                labels = group["y"][:]
                assert len(labels) == len(group["X"])
                assert set(np.unique(labels)).issubset({0, 1})
                count += len(labels)
                positives += int(labels.sum())
        assert count == expected, (split, count, expected)
        summary[split] = {"windows": count, "positive": positives, "path": str(target)}
    manifest["outputs"] = summary
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("UPSTREAM_DATA_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
