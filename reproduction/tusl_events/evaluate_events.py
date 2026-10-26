"""Event-level evaluation of a fine-tuned three-class TUSL classifier.

Every window of ``val.h5`` and ``test.h5`` (two per event, ``<event>_w0/_w1``)
is scored once in file order with the checkpoint's own normalization and
quantization state; the two window probabilities of an event are averaged and
the event is the evaluation unit. Metrics: macro one-vs-rest AUROC (constant
scores give 0.5), macro average precision, accuracy and balanced accuracy of
the argmax (constant predictions give 1/3). Window-level values are kept for
reference. No threshold or checkpoint is selected here.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    from scipy.stats import rankdata
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, positives: np.ndarray) -> float:
    if positives.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = positives[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    predictions = probs.argmax(axis=1)
    per_class = {}
    for c in range(num_classes):
        positives = labels == c
        per_class[str(c)] = {"auroc": rank_auroc(probs[:, c], positives), "ap": average_precision(probs[:, c], positives),
                             "recall": float((predictions[positives] == c).mean()) if positives.any() else float("nan"),
                             "count": int(positives.sum())}
    recalls = [per_class[str(c)]["recall"] for c in range(num_classes)]
    return {"n": int(len(labels)), "macro_auroc": float(np.nanmean([per_class[str(c)]["auroc"] for c in range(num_classes)])),
            "macro_ap": float(np.nanmean([per_class[str(c)]["ap"] for c in range(num_classes)])),
            "accuracy": float((predictions == labels).mean()), "balanced_accuracy": float(np.nanmean(recalls)),
            "confusion_matrix": np.bincount(labels * num_classes + predictions, minlength=num_classes ** 2)
            .reshape(num_classes, num_classes).tolist(), "per_class": per_class}


def score_split(task, path: Path, device: str, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch
    probs, labels, names = [], [], []
    with h5py.File(path, "r") as h5f, torch.no_grad():
        for group in sorted(h5f.keys(), key=lambda g: int(g.rsplit("_", 1)[1])):
            x_all, y_all = h5f[group]["X"], h5f[group]["y"][:]
            names.extend(n.decode() for n in h5f[group]["source_file"][:])
            labels.extend(int(v) for v in y_all)
            for start in range(0, len(y_all), batch_size):
                x = torch.from_numpy(x_all[start:start + batch_size].astype(np.float32)).to(device)
                if getattr(task, "normalize", False):
                    x = task.normalize_fct(x)
                logits, _ = task.model(x, torch.zeros_like(x, dtype=torch.bool))
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs), np.asarray(labels, dtype=np.int64), names


def to_events(probs: np.ndarray, labels: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped = defaultdict(list)
    for i, name in enumerate(names):
        grouped[name.rsplit("_w", 1)[0]].append(i)
    events, event_probs, event_labels = [], [], []
    for event, rows in grouped.items():
        if len({int(labels[r]) for r in rows}) != 1:
            raise ValueError(f"{event}: windows carry different labels")
        events.append(event)
        event_probs.append(probs[rows].mean(axis=0))
        event_labels.append(int(labels[rows[0]]))
    return np.stack(event_probs), np.asarray(event_labels, dtype=np.int64), events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="fold directory with val.h5, test.h5, data-manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("fp32", "ptq", "qap_b", "c"), required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=("val", "test"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    from evaluate import load_task

    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the recorded value")
    manifest = json.loads((args.data / "data-manifest.json").read_text())
    num_classes, kind = int(manifest["num_classes"]), str(manifest["classification_type"])
    task, contract = load_task(args.checkpoint, "ptq_original" if args.model_kind == "ptq" else args.model_kind,
                               device=args.device, num_classes=num_classes, classification_type=kind)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"model": args.model_kind, "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
              "data": {s: {"path": str(args.data / f"{s}.h5"), "sha256": manifest["data"][s]["sha256"]} for s in args.splits},
              "num_classes": num_classes, "quantization": contract, "baselines": {"macro_auroc": 0.5,
              "balanced_accuracy": 1 / num_classes}, "splits": {}}
    for split in args.splits:
        probs, labels, names = score_split(task, args.data / f"{split}.h5", args.device)
        if len(labels) != manifest["splits"][split]["windows"]:
            raise ValueError(f"{split}: {len(labels)} windows, manifest says {manifest['splits'][split]['windows']}")
        event_probs, event_labels, events = to_events(probs, labels, names)
        result["splits"][split] = {"windows": metrics(probs, labels, num_classes), "events": metrics(event_probs, event_labels, num_classes)}
        np.savez_compressed(args.output / f"{split}_scores.npz", window_probs=probs, window_labels=labels,
                            window_names=np.asarray(names), event_probs=event_probs, event_labels=event_labels,
                            events=np.asarray(events))
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print("EVENT_EVALUATION_COMPLETE", json.dumps({s: {k: round(result["splits"][s]["events"][k], 4)
          for k in ("macro_auroc", "macro_ap", "balanced_accuracy")} for s in args.splits}))


if __name__ == "__main__":
    main()
