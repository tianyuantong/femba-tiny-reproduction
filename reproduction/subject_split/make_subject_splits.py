"""Subject-independent train/val/test splits for TUAR, TUAB and TUSL.

Reads only EDF file names. The subject id is the file name up to the first
underscore (``aaaaaaju_s005_t000.edf`` -> ``aaaaaaju``). Subjects are sorted,
shuffled once with ``numpy.random.default_rng(seed)`` and cut 80/10/10; TUAB
keeps the official ``eval`` directory as the test split and cuts its ``train``
directory 80/20 into train/val. TUAR uses the ``01_tcp_ar`` and ``02_tcp_le``
montages only, which reproduces the 2025 reproduction split (167/21/21
subjects for seed 42).

Outputs, per corpus, ``<name>-split.json`` with subjects and relative file
paths per split, plus ``pretrain-exclusion.json`` (union of all test subjects)
and ``summary.json`` (counts only). The per-subject files contain licensed
identifiers and stay outside git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TUAR_MONTAGES = ("01_tcp_ar", "02_tcp_le")


def subject_id(path: Path) -> str:
    return path.name.split("_", 1)[0]


def edf_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.edf") if p.is_file())


def shuffled_subjects(files: list[Path], seed: int) -> list[str]:
    subjects = sorted({subject_id(p) for p in files})
    np.random.default_rng(seed).shuffle(subjects)
    return subjects


def by_subject(files: list[Path], root: Path, subjects: list[str]) -> list[str]:
    wanted = set(subjects)
    return [str(p.relative_to(root)) for p in files if subject_id(p) in wanted]


def split_80_10_10(name: str, root: Path, files: list[Path], seed: int) -> dict:
    subjects = shuffled_subjects(files, seed)
    train_end, val_end = int(len(subjects) * 0.8), int(len(subjects) * 0.9)
    parts = {"train": subjects[:train_end], "val": subjects[train_end:val_end],
             "test": subjects[val_end:]}
    return {"corpus": name, "seed": seed, "unit": "subject", "rule": "sorted subjects, rng shuffle, 80/10/10",
            "root": str(root), "subjects": parts,
            "files": {k: by_subject(files, root, v) for k, v in parts.items()}}


def split_tuab(root: Path, seed: int) -> dict:
    train_files, eval_files = edf_files(root / "train"), edf_files(root / "eval")
    train_subjects = shuffled_subjects(train_files, seed)
    cut = int(len(train_subjects) * 0.8)
    parts = {"train": train_subjects[:cut], "val": train_subjects[cut:],
             "test": sorted({subject_id(p) for p in eval_files})}
    overlap = sorted(set(parts["test"]) & set(train_subjects))
    if overlap:
        raise ValueError(f"TUAB eval subjects also appear in train: {len(overlap)}")
    return {"corpus": "TUAB", "seed": seed, "unit": "subject",
            "rule": "official eval = test; train subjects rng shuffle 80/20", "root": str(root),
            "subjects": parts,
            "files": {"train": by_subject(train_files, root, parts["train"]),
                      "val": by_subject(train_files, root, parts["val"]),
                      "test": by_subject(eval_files, root, parts["test"])}}


def check_disjoint(split: dict) -> None:
    seen: dict[str, str] = {}
    for part, subjects in split["subjects"].items():
        for subject in subjects:
            if subject in seen:
                raise ValueError(f"{split['corpus']}: {subject} in both {seen[subject]} and {part}")
            seen[subject] = part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuar", type=Path, required=True, help="TUAR .../edf directory")
    parser.add_argument("--tuab", type=Path, required=True, help="TUAB .../edf directory with train/ and eval/")
    parser.add_argument("--tusl", type=Path, default=None, help="TUSL .../edf directory (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tuar_files = [p for m in TUAR_MONTAGES for p in edf_files(args.tuar / m)]
    splits = [split_80_10_10("TUAR", args.tuar, tuar_files, args.seed), split_tuab(args.tuab, args.seed)]
    if args.tusl is not None:
        splits.append(split_80_10_10("TUSL", args.tusl, edf_files(args.tusl), args.seed))

    summary, test_subjects, all_subjects = {}, set(), {}
    for split in splits:
        check_disjoint(split)
        (args.output / f"{split['corpus'].lower()}-split.json").write_text(json.dumps(split, indent=2) + "\n")
        summary[split["corpus"]] = {part: {"subjects": len(split["subjects"][part]), "files": len(split["files"][part])}
                                    for part in ("train", "val", "test")}
        test_subjects |= set(split["subjects"]["test"])
        all_subjects[split["corpus"]] = {s for part in split["subjects"].values() for s in part}

    names = sorted(all_subjects)
    overlap = {f"{a}&{b}": len(all_subjects[a] & all_subjects[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    exclusion = {"seed": args.seed, "rule": "union of test subjects of every corpus", "subjects": sorted(test_subjects)}
    (args.output / "pretrain-exclusion.json").write_text(json.dumps(exclusion, indent=2) + "\n")
    summary.update({"seed": args.seed, "cross_corpus_subject_overlap": overlap,
                    "pretrain_exclusion_subjects": len(test_subjects)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
