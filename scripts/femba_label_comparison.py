"""Compare frozen best checkpoints under one label protocol using saved logits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_run(root: Path) -> dict:
    results = root / "results"
    run = _read_json(results / "upstream2025-run.json")
    training = _read_json(results / "upstream2025-training-results.json")
    audit = _read_json(results / "upstream2025-audit.json")
    manifest_path = Path(run["data_manifest_path"])
    if _digest(manifest_path) != run["data_manifest_sha256"]:
        raise ValueError("Run data manifest has changed")
    manifest = _read_json(manifest_path)
    if manifest["status"] != "complete" or manifest["upstream_commit"] != run["upstream_commit"]:
        raise ValueError("Run data preparation is incomplete or has a different upstream revision")
    best = audit["checkpoints"]["best"]
    prediction_path = results / "upstream2025-best-predictions.pt"
    if _digest(prediction_path) != best["predictions_sha256"]:
        raise ValueError("Saved best predictions have changed")
    if _digest(Path(training["best_checkpoint"])) != best["checkpoint_sha256"]:
        raise ValueError("Validation-selected best checkpoint has changed")
    _require_exact_metrics(best["native_metrics"], {
        key: training["best_test"][0][key] for key in best["native_metrics"]
    })
    return {"root": root, "run": run, "training": training, "audit": audit,
            "manifest": manifest, "prediction_path": prediction_path}


def _normalized_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    for split in ("train", "val", "test"):
        config["data_module"][split]["hdf5_file"] = f"{split}.h5"
    for name in ("base_output_path", "checkpoint_dirpath"):
        config["io"][name] = name
    return config


def _require_same_recipe(control: dict, paper: dict) -> None:
    for key in ("upstream_commit", "released_sha256", "source_hashes", "precision", "training_budget"):
        if control[key] != paper[key]:
            raise ValueError(f"Non-label training change: {key}")
    if control["executed_adapter"]["sha256"] != paper["executed_adapter"]["sha256"]:
        raise ValueError("The two runs executed different training adapters")
    if _normalized_config(control["config"]) != _normalized_config(paper["config"]):
        raise ValueError("Training configurations differ beyond data/output paths")
    model = control["config"]["model"]
    if model["classification_type"] != "bc" or model["num_classes"] != 2:
        raise ValueError("This evaluator requires the pinned two-class BC task")


def _label_digest(labels) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest()


def _paired_data_paths(control: dict, paper: dict) -> tuple[Path, Path]:
    manifest = paper["manifest"]
    output = manifest["outputs"]["test"]
    parent = manifest["parent_manifest"]
    if control["manifest"]["artifact_fraction_threshold"] != 0.3:
        raise ValueError("Expected the frozen >=30% control-label protocol")
    if parent["sha256"] != control["run"]["data_manifest_sha256"]:
        raise ValueError("Paper labels do not derive from the control data manifest")
    if _digest(paper["root"] / parent["snapshot"]) != parent["sha256"]:
        raise ValueError("Parent data manifest snapshot has changed")
    if manifest["records"] != control["manifest"]["records"]:
        raise ValueError("Record identities or train/validation/test assignments changed")
    if manifest["label_rule"]["name"] != "paper13_any":
        raise ValueError("Expected the declared paper any-artifact label rule")
    old_path = Path(control["run"]["config"]["data_module"]["test"]["hdf5_file"]).resolve()
    new_path = Path(paper["run"]["config"]["data_module"]["test"]["hdf5_file"]).resolve()
    if Path(output["source_h5"]["path"]).resolve() != old_path or Path(output["path"]).resolve() != new_path:
        raise ValueError("Manifest HDF5 paths do not match the evaluated runs")
    old_hash, new_hash = _digest(old_path), _digest(new_path)
    if old_hash != output["source_h5"]["sha256"] or old_hash != control["audit"]["test_sha256"]:
        raise ValueError("Control test HDF5 identity changed")
    if new_hash != output["sha256"] or new_hash != paper["audit"]["test_sha256"]:
        raise ValueError("Paper test HDF5 identity changed")
    if _digest(paper["root"] / output["mapping_file"]) != output["mapping_sha256"]:
        raise ValueError("Sample mapping evidence changed")
    return old_path, new_path


def _paired_labels(control: dict, paper: dict):
    import h5py
    import numpy as np

    old_path, new_path = _paired_data_paths(control, paper)
    output = paper["manifest"]["outputs"]["test"]
    old_labels, new_labels = [], []
    with h5py.File(old_path, "r") as old, h5py.File(new_path, "r") as new:
        if list(old.keys()) != list(new.keys()):
            raise ValueError("HDF5 loader group order changed")
        for group in old.keys():
            link = new[group].get("X", getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                raise ValueError("Paper inputs must link to the frozen control inputs")
            if (new_path.parent / link.filename).resolve() != old_path or link.path != f"/{group}/X":
                raise ValueError("Paper input link changes the source or sample order")
            old_y, new_y = old[group]["y"][:], new[group]["y"][:]
            if old_y.shape != new_y.shape or old_y.shape != (len(old[group]["X"]),):
                raise ValueError("Label shape does not match paired input rows")
            old_labels.extend(old_y)
            new_labels.extend(new_y)
    old_labels, new_labels = np.asarray(old_labels), np.asarray(new_labels)
    if not np.isin(old_labels, (0, 1)).all() or not np.isin(new_labels, (0, 1)).all():
        raise ValueError("Labels must be binary")
    transitions = np.bincount(old_labels.astype(np.int64) * 2 + new_labels.astype(np.int64),
                             minlength=4).reshape(2, 2).tolist()
    if transitions != output["transition_matrix"]:
        raise ValueError("Observed label transitions differ from preparation evidence")
    if _label_digest(old_labels) != output["old_labels_sha256"] or _label_digest(new_labels) != output["new_labels_sha256"]:
        raise ValueError("Ordered label hashes do not match preparation evidence")
    if len(new_labels) != output["windows"] or any(
            len(new_labels) != artifact["audit"]["test_windows"] for artifact in (control, paper)):
        raise ValueError("Test window counts differ between the prepared and evaluated datasets")
    if int(new_labels.sum()) != output["positive"] or int((old_labels != new_labels).sum()) != output["changed_labels"]:
        raise ValueError("Prepared positive/changed label counts do not match HDF5 labels")
    if len(np.unique(old_labels)) != 2 or len(np.unique(new_labels)) != 2:
        raise ValueError("Both original and paper test labels must contain both classes for AUROC")
    return old_labels, new_labels


def _load_predictions(artifact: dict, expected_labels):
    import torch

    payload = torch.load(artifact["prediction_path"], map_location="cpu", weights_only=True)
    logits, labels = payload["logits"], payload["labels"]
    expected = torch.as_tensor(expected_labels, dtype=torch.long)
    if logits.shape != (len(expected), 2) or not torch.equal(labels, expected):
        raise ValueError("Saved prediction rows/labels do not match the frozen HDF5 loader order")
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError("Saved predictions must contain finite FP32 logits")
    return logits


def _native_metrics(logits, labels, batch_size: int, device: str) -> dict:
    import torch
    from torchmetrics import MetricCollection
    from torchmetrics.classification import Accuracy, AUROC, AveragePrecision, CohenKappa, F1Score, Precision, Recall

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Native replay requires a positive original batch size")
    label_metrics = MetricCollection([
        Accuracy(task="binary", num_classes=2, average="macro"),
        Recall(task="multiclass", num_classes=2, average="macro"),
        Precision(task="binary", num_classes=2, average="macro"),
        F1Score(task="binary", num_classes=2, average="macro"),
        CohenKappa(task="binary", num_classes=2),
    ], prefix="test_").to(device)
    logit_metrics = MetricCollection([
        AUROC(task="binary", num_classes=2, average="macro"),
        AveragePrecision(task="binary", num_classes=2, average="macro"),
    ], prefix="test_").to(device)
    with torch.inference_mode():
        for start in range(0, len(logits), batch_size):
            batch = logits[start:start + batch_size].to(device)
            target = torch.as_tensor(labels[start:start + batch_size], dtype=torch.long, device=device)
            label_metrics(batch.softmax(1).argmax(1), target)
            # Preserve upstream per-batch auto-sigmoid behavior and squeeze.
            logit_metrics(batch[:, 1].squeeze(), target)
        return {key: float(value.cpu()) for metrics in (label_metrics, logit_metrics)
                for key, value in metrics.compute().items()}


def _require_exact_metrics(actual: dict, expected: dict) -> None:
    if actual.keys() != expected.keys():
        raise ValueError("Native replay metric keys differ from the recorded audit")
    errors = {key: abs(actual[key] - expected[key]) for key in actual if actual[key] != expected[key]}
    if errors:
        raise ValueError(f"Native replay did not exactly reproduce original-label metrics: {errors}")


def _softmax_diagnostics(logits, labels) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = logits.softmax(1)[:, 1].numpy()
    return {"auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    control_root, paper_root = args.control_root.resolve(), args.paper_root.resolve()
    output_path = paper_root / "results/paper-label-comparison.json"
    if output_path.exists():
        raise FileExistsError(f"Comparison evidence already exists: {output_path}")
    control, paper = _load_run(control_root), _load_run(paper_root)
    _require_same_recipe(control["run"], paper["run"])
    old_labels, paper_labels = _paired_labels(control, paper)

    import torch
    import torchmetrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    for name, artifact, original_labels in (("control_best", control, old_labels),
                                             ("paper_best", paper, paper_labels)):
        logits = _load_predictions(artifact, original_labels)
        batch_size = artifact["run"]["config"]["batch_size"]
        original = _native_metrics(logits, original_labels, batch_size, device)
        best = artifact["audit"]["checkpoints"]["best"]
        _require_exact_metrics(original, best["native_metrics"])
        rows[name] = {
            "checkpoint_sha256": best["checkpoint_sha256"],
            "predictions_sha256": best["predictions_sha256"],
            "selected_epoch": best["checkpoint_epoch"], "selected_step": best["checkpoint_step"],
            "selection": "min validation loss under original >=30% labels" if name == "control_best"
                         else "min validation loss under paper13_any labels",
            "native_replay_batch_size": batch_size, "original_label_native": original,
            "paper_label_native": _native_metrics(logits, paper_labels, batch_size, device),
            "paper_label_softmax_diagnostic": _softmax_diagnostics(logits, paper_labels),
        }
    result = {
        "status": "complete", "comparison_script_sha256": _digest(Path(__file__)),
        "torch": torch.__version__, "torchmetrics": torchmetrics.__version__, "metric_device": device,
        "samples": len(paper_labels), "old_positive": int(old_labels.sum()),
        "paper_positive": int(paper_labels.sum()), "changed_labels": int((old_labels != paper_labels).sum()),
        "paper_data_manifest_sha256": paper["run"]["data_manifest_sha256"],
        "paper_label_rule": paper["manifest"]["label_rule"],
        "input_identity": "Original HDF5 SHA and identical loader-ordered ExternalLink targets verified",
        "scope": "Training and validation label-protocol intervention; checkpoints were not reselected",
        "rows": rows,
    }
    with output_path.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print("PAPER_LABEL_COMPARISON_OK", json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
