"""Expected calibration error of the formal-line arms from the stored test scores.

Top-label ECE with 15 equal-width confidence bins (Guo et al., ICML 2017). Binary tasks read the
logit captures. TUSL pools the event probabilities of its four folds within a seed, because a
single fold is too small for 15 bins. Seeds are the unit of variation.
    usage: calibration_error.py --artifacts <dir> --output <json> [--settings w8a8 w4a8 ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_formal import BINARY_TASKS, SETTINGS, TUSL_FOLDS, arm_dirs, load

BINS = 15
CAPTURE = {"fp32": "fp32", "c": "c", "qap": "qap_b"}


def ece(probs: np.ndarray, labels: np.ndarray) -> float:
    confidence, predictions = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        members = (confidence > low) & (confidence <= high)
        if members.any():
            total += members.mean() * abs((predictions[members] == labels[members]).mean() - confidence[members].mean())
    return float(total)


def binary_scores(arm: str, root: Path, task: str):
    if arm == "ptq":
        sweep = root / f"sweep-{task}"
        result = load(sweep / "results.json")
        path = sweep / f"{result['selected']['candidate']}_test_logits.npz" if result else None
    else:
        path = root / f"finetune-{task}" / "eval" / f"{CAPTURE[arm]}_test_logits.npz"
    if path is None or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        logits, labels = data["logits"].astype(np.float64), data["labels"]
    logits -= logits.max(axis=1, keepdims=True)
    return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True), labels


def tusl_scores(arm: str, root: Path):
    folds = []
    for fold in range(TUSL_FOLDS):
        task = f"tusl{fold}-v2"
        path = root / f"sweep-{task}" / "test" / "test_scores.npz" if arm == "ptq" else root / f"finetune-{task}" / "eval" / "test_scores.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            folds.append((data["event_probs"].astype(np.float64), data["event_labels"]))
    return np.concatenate([f[0] for f in folds]), np.concatenate([f[1] for f in folds])


def collect(artifacts: Path, settings) -> dict:
    out = {}
    for seed_dir in sorted(artifacts.glob("seed*")):
        seed = seed_dir.name[4:]
        for setting in settings:
            for arm, root in arm_dirs(seed_dir, setting).items():
                for task in (*BINARY_TASKS, "tusl-v2"):
                    scores = tusl_scores(arm, root) if task == "tusl-v2" else binary_scores(arm, root, task)
                    if scores is not None:
                        out.setdefault(setting, {}).setdefault(task, {}).setdefault(arm, {})[seed] = ece(*scores)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=SETTINGS)
    args = parser.parse_args()
    summary = collect(args.artifacts, args.settings)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for setting, tasks in summary.items():
        for task, arms in tasks.items():
            print(setting, task, {arm: round(sum(s.values()) / len(s), 4) for arm, s in arms.items()})


if __name__ == "__main__":
    main()
