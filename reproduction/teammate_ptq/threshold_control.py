#*----------------------------------------------------------------------------*
#* Copyright (C) 2026.                                                        *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");              *
#* you may not use this file except in compliance with the License.             *
#* You may obtain a copy of the License at                                      *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                   *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software          *
#* distributed under the License is distributed on an "AS IS" BASIS,            *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    *
#* See the License for the specific language governing permissions and         *
#* limitations under the License.                                              *
#*----------------------------------------------------------------------------*
"""Choose binary thresholds on validation captures, then evaluate frozen test logits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


MODEL_IDS = ("fp32", "ptq_original", "ptq_p99", "qap_b")
PROTOCOL = {
    "score": "float64(logit_1) - float64(logit_0)",
    "decision": "positive iff score > threshold; ties are class 0",
    "default_threshold": 0.0,
    "selection_split": "val",
    "objective": "maximum validation balanced accuracy",
    "candidates": "unique validation scores, 0, and nextafter(minimum score, -inf)",
    "tie_break": ["smallest absolute threshold", "lowest threshold"],
    "execution": "save all selections exclusively before reading any test capture",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Expected nonempty one-dimensional labels and matching scores")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary integers")
    if not np.isfinite(scores).all() or len(np.unique(labels)) != 2:
        raise ValueError("Scores must be finite and labels must contain both classes")


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    validate(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predictions = (scores > threshold).astype(np.int64)
    matrix = np.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = matrix.ravel().tolist()
    return {
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "accuracy": (tn + tp) / len(labels),
        "balanced_accuracy": 0.5 * (tn / (tn + fp) + tp / (tp + fn)),
        "predicted_positive_rate": float(predictions.mean()),
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    validate(labels, scores)
    order = np.argsort(scores, kind="stable")
    ends = np.flatnonzero(np.r_[np.diff(scores[order]) != 0, True])
    counts = np.diff(np.r_[0, ends + 1])
    positives = np.diff(np.r_[0, np.cumsum(labels[order])[ends]])
    negatives = counts - positives
    total_positive, total_negative = int(positives.sum()), int(negatives.sum())
    concordant = positives * (np.cumsum(negatives) - 0.5 * negatives)
    precision = np.cumsum(positives[::-1]) / np.cumsum(counts[::-1])
    return {
        "auroc": float(concordant.sum() / (total_positive * total_negative)),
        "ap": float(np.sum(precision * positives[::-1] / total_positive)),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    validate(labels, scores)
    order = np.argsort(scores, kind="stable")
    below_minimum = np.nextafter(scores.min(), -np.inf)
    if not np.isfinite(below_minimum):
        raise ValueError("Minimum score has no finite lower threshold")
    candidates = np.unique(np.r_[scores, 0.0, below_minimum])
    ends = np.searchsorted(scores[order], candidates, side="right")
    positive_prefix = np.r_[0, np.cumsum(labels[order])]
    negative_prefix = np.arange(len(labels) + 1) - positive_prefix
    positives, negatives = positive_prefix[-1], negative_prefix[-1]
    true_positive = positives - positive_prefix[ends]
    true_negative = negative_prefix[ends]
    # All candidates share denominator 2 * positives * negatives, so ties are exact.
    objective = true_negative * positives + true_positive * negatives
    best = np.flatnonzero(objective == objective.max())
    best = best[np.lexsort((candidates[best], np.abs(candidates[best])))][0]
    threshold = float(candidates[best])
    return {
        "threshold": threshold,
        "candidate_count": int(len(candidates)),
        "best_candidate_count": int(np.count_nonzero(objective == objective.max())),
        "objective_numerator": int(objective[best]),
        "objective_denominator": int(2 * positives * negatives),
        "validation_default": binary_metrics(labels, scores, 0.0),
        "validation_selected": binary_metrics(labels, scores, threshold),
        "validation_ranking": ranking_metrics(labels, scores),
    }


def load_capture(captures: Path, model_id: str, split: str) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    metadata_path = captures / f"{model_id}_{split}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model"] != model_id or metadata["split"] != split:
        raise ValueError("Capture metadata does not match requested model and split")
    array_path = captures / f"{model_id}_{split}_logits.npz"
    hashes = {metadata_path.name: digest(metadata_path), array_path.name: digest(array_path)}
    if hashes[array_path.name] != metadata["array_sha256"]:
        raise ValueError("Capture array hash differs from recorded evidence")
    with np.load(array_path, allow_pickle=False) as arrays:
        logits = arrays["logits"].astype(np.float64)
        labels = arrays["labels"].copy()
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all():
        raise ValueError("Expected finite two-class logits")
    scores = logits[:, 1] - logits[:, 0]
    validate(labels, scores)
    if len(labels) != metadata["windows"]:
        raise ValueError("Capture length differs from recorded evidence")
    if hashlib.sha256(labels.tobytes()).hexdigest() != metadata["labels_sha256"]:
        raise ValueError("Capture labels differ from recorded evidence")
    return metadata, labels, scores, hashes


def run(captures: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", {"time_utc": timestamp(), **PROTOCOL})
    selections, inputs, validation_label_hashes = [], {}, set()
    for model_id in MODEL_IDS:
        metadata, labels, scores, hashes = load_capture(captures, model_id, "val")
        inputs.update(hashes)
        validation_label_hashes.add(metadata["labels_sha256"])
        selections.append({"model": model_id, "checkpoint_sha256": metadata["checkpoint_sha256"],
                           **select_threshold(labels, scores)})
    assert len(validation_label_hashes) == 1, "Validation ordering must match across models"
    selection_path = output / "selection.json"
    write_json(selection_path, {"time_utc": timestamp(), "protocol": PROTOCOL,
                               "validation_inputs_sha256": inputs.copy(), "models": selections})
    selection_sha = digest(selection_path)
    write_json(output / "test_started.json", {"time_utc": timestamp(), "selection_sha256": selection_sha})
    rows, test_label_hashes = [], set()
    for selected in selections:
        metadata, labels, scores, hashes = load_capture(captures, selected["model"], "test")
        inputs.update(hashes)
        test_label_hashes.add(metadata["labels_sha256"])
        assert metadata["checkpoint_sha256"] == selected["checkpoint_sha256"]
        rows.append({"model": selected["model"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                     "windows": len(labels), "class_counts": np.bincount(labels, minlength=2).tolist(),
                     **ranking_metrics(labels, scores),
                     "default": binary_metrics(labels, scores, 0.0),
                     "validation_selected": binary_metrics(labels, scores, selected["threshold"])})
    assert len(test_label_hashes) == 1, "Test ordering must match across models"
    assert digest(selection_path) == selection_sha, "Selection must remain immutable"
    assert all(digest(captures / name) == expected for name, expected in inputs.items())
    result = {"time_utc": timestamp(), "protocol": PROTOCOL, "models": rows,
              "selection_sha256": selection_sha, "inputs_sha256": inputs,
              "runner_sha256": digest(Path(__file__)), "captures_unchanged": True}
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.captures, args.output)
    print(json.dumps({"status": "complete", "models": result["models"]}, allow_nan=False))


if __name__ == "__main__":
    main()
