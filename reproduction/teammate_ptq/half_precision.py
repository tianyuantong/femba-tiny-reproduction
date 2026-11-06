"""Replay an FP32 fine-tuned checkpoint with weights, activations and inputs in FP16 or BF16.

Binary tasks report what evaluate_one.py reports, with the threshold chosen on validation; TUSL
reports what evaluate_events.py reports. float32 runs the same path as a check against the stored
FP32 results. Windows with a non-finite logit are counted and left out of the metrics.
    usage: half_precision.py --source <dir> --data <dir> --checkpoint <file> --checkpoint-sha256 <hex>
                             --dtype bfloat16 --output <new dir>
"""
import argparse
from pathlib import Path
import sys

import numpy as np

import evaluate as capture
from evaluate_events import metrics, score_split, to_events
from threshold_control import binary_metrics, ranking_metrics, select_threshold, timestamp

DTYPES = ("float16", "bfloat16", "float32")
SPLITS = ("val", "test")


def replay(task, path: Path, device: str, dtype) -> dict:
    """Score one split through evaluate_events, casting at the model's input and keeping raw logits."""
    logits = []
    hooks = (task.model.register_forward_pre_hook(lambda _, inputs: (inputs[0].to(dtype), *inputs[1:])),
             task.model.register_forward_hook(lambda _, __, output: logits.append(output[0].float().cpu().numpy())))
    probs, labels, names = score_split(task, path, device)
    for hook in hooks:
        hook.remove()
    return {"logits": np.concatenate(logits), "probs": probs, "labels": labels, "names": np.asarray(names)}


def margin(logits: np.ndarray) -> np.ndarray:
    return logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)


def binary_results(val: dict, test: dict, output: Path) -> dict:
    threshold = select_threshold(val["labels"], margin(val["logits"]))["threshold"]
    scores = margin(test["logits"])
    np.savez_compressed(output / "fp32_test_logits.npz", logits=test["logits"], labels=test["labels"])
    return {**ranking_metrics(test["labels"], scores), "threshold": threshold,
            "default": binary_metrics(test["labels"], scores, 0.0),
            "validation_selected": binary_metrics(test["labels"], scores, threshold)}


def event_results(splits: dict, num_classes: int, output: Path) -> dict:
    results = {}
    for split, row in splits.items():
        event_probs, event_labels, _ = to_events(row["probs"], row["labels"], row["names"])
        results[split] = {"windows": metrics(row["probs"], row["labels"], num_classes),
                          "events": metrics(event_probs, event_labels, num_classes)}
        np.savez_compressed(output / f"{split}_scores.npz", event_probs=event_probs, event_labels=event_labels)
    return {"splits": results}


def run(args) -> dict:
    sys.path.insert(0, str(args.source))
    import torch

    if capture.sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the recorded value")
    model_config = capture.safe_load_checkpoint(args.checkpoint)["hyper_parameters"]["model"]
    num_classes = model_config["num_classes"]
    runtime = capture.configure_runtime()
    task, _ = capture.load_task(args.checkpoint, "fp32", device=args.device, num_classes=num_classes,
                                classification_type=model_config["classification_type"])
    dtype = getattr(torch, args.dtype)
    task.model.to(dtype)
    torch.set_default_dtype(dtype)  # tensors the scan creates inside forward, as Lightning's true-precision modes do
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"dtype": args.dtype, "checkpoint_sha256": args.checkpoint_sha256, "device": runtime["gpu"],
              "windows": {}, "nonfinite_windows": {}}
    splits = {}
    for split in SPLITS:
        row = replay(task, args.data / f"{split}.h5", args.device, dtype)
        is_finite = np.isfinite(row["logits"]).all(axis=1)
        result["windows"][split], result["nonfinite_windows"][split] = len(is_finite), int((~is_finite).sum())
        splits[split] = {key: value[is_finite] for key, value in row.items()}
    result.update(binary_results(splits["val"], splits["test"], args.output) if num_classes == 2
                  else event_results(splits, num_classes, args.output))
    result["completed_at"] = timestamp()
    capture.write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dtype", choices=DTYPES, required=True)
    parser.add_argument("--device", default="cuda")
    result = run(parser.parse_args())
    print({key: value for key, value in result.items() if key not in ("splits", "default", "validation_selected")}, flush=True)


if __name__ == "__main__":
    main()
