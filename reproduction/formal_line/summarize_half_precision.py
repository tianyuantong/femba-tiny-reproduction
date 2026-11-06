"""Test metrics of the FP16, BF16 and float32 replays, per task and seed, with the windows that overflowed.

run_half_precision.sh leaves each type under <artifacts>/<dtype> in the layout of the formal FP32
arm, so the readers of summarize_formal.py and calibration_error.py apply unchanged.
    usage: summarize_half_precision.py --artifacts <dir> --output <json>
"""
import argparse
import json
from pathlib import Path
import statistics

from calibration_error import binary_scores, ece, tusl_scores
from summarize_formal import BINARY_TASKS, TUSL_FOLDS, TUSL_METRICS, binary_row, load, tusl_row

DTYPES = ("float32", "bfloat16", "float16")
SPLITS = ("val", "test")


def tusl_seed(root: Path) -> dict:
    results = [load(root / f"finetune-tusl{fold}-v2" / "eval" / "results.json") for fold in range(TUSL_FOLDS)]
    rows = [tusl_row(result) for result in results]
    return {**{metric: statistics.mean(row[metric] for row in rows) for metric in TUSL_METRICS},
            "ece": ece(*tusl_scores("fp32", root)),
            "windows": {split: sum(result["windows"][split] for result in results) for split in SPLITS},
            "nonfinite_windows": {split: sum(result["nonfinite_windows"][split] for result in results) for split in SPLITS}}


def binary_seed(root: Path, task: str) -> dict:
    result = load(root / f"finetune-{task}" / "eval" / "results.json")
    return {**binary_row(result, False), "ece": ece(*binary_scores("fp32", root, task)),
            "windows": result["windows"], "nonfinite_windows": result["nonfinite_windows"]}


def collect(artifacts: Path) -> dict:
    summary = {}
    for dtype in DTYPES:
        for seed_dir in sorted((artifacts / dtype).glob("seed*")):
            seed, root = seed_dir.name[4:], seed_dir / "fp32"
            tasks = summary.setdefault(dtype, {})
            for task in BINARY_TASKS:
                tasks.setdefault(task, {})[seed] = binary_seed(root, task)
            tasks.setdefault("tusl-v2", {})[seed] = tusl_seed(root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.artifacts)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
