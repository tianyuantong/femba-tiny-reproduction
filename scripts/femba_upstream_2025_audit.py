"""Reload both upstream checkpoints and verify full-test metrics and predictions."""
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score

from femba_upstream_2025_train import ROOT, DATA, REF, upstream_module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = ROOT / "results"
    final = json.loads((results / "upstream2025-training-results.json").read_text())
    run = json.loads((results / "upstream2025-run.json").read_text())
    data = json.loads((results / "upstream2025-data.json").read_text())
    assert data["status"] == "complete" and run["upstream_commit"] == REF
    torch.set_num_threads(2)
    upstream_module("models.FEMBA", "models/FEMBA.py")
    upstream_module("util.train_utils", "util/train_utils.py")
    upstream_module("datasets.hdf5_dataset", "datasets/hdf5_dataset.py")
    upstream_module("data_module.finetune_data_module", "data_module/finetune_data_module.py")
    module = upstream_module("tasks.finetune_task", "tasks/finetune_task.py")
    cfg = OmegaConf.create(run["config"])
    cfg.num_workers = 0  # Evaluation IO only; preserve sample and batch ordering.
    dm = hydra.utils.instantiate(cfg.data_module)
    dm.setup("test")
    expected = data["outputs"]["test"]["windows"]
    audit = {"upstream_commit": REF, "test_windows": expected,
             "test_sha256": digest(DATA / "upstream2025_test.h5"), "checkpoints": {}}
    subjects = {split: {Path(row["source"]).name.split("_")[0]
                       for row in data["records"] if row["split"] == split}
                for split in ("train", "val", "test")}
    audit["subject_counts"] = {k: len(v) for k, v in subjects.items()}
    audit["subject_overlap"] = {a + "_" + b: len(subjects[a] & subjects[b])
                                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    previous_labels = None
    for kind, ckpt_key, metric_key in [("final", "final_checkpoint", "upstream_final_test"),
                                        ("best", "best_checkpoint", "best_test")]:
        path = Path(final[ckpt_key])
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = module.FinetuneTask(cfg)
        model.load_state_dict(state["state_dict"], strict=True)
        model.cuda().eval()
        model.test_label_metrics.reset()
        model.test_logit_metrics.reset()
        logits, labels = [], []
        with torch.inference_mode():
            for x, y in dm.test_dataloader():
                x, y = model.normalize_fct(x.cuda()), y.cuda()
                mask = model.generate_fake_mask(len(x), x.shape[1], x.shape[2])
                pred = model._step(x, mask)
                assert torch.isfinite(pred["logits"]).all()
                model.test_label_metrics(pred["label"], y)
                model.test_logit_metrics(model._handle_binary(pred["logits"]), y)
                logits.append(pred["logits"].cpu())
                labels.append(y.cpu())
        logits, labels = torch.cat(logits), torch.cat(labels)
        assert len(labels) == expected
        if previous_labels is not None:
            assert torch.equal(labels, previous_labels)
        previous_labels = labels
        native = {k: float(v.cpu()) for metrics in (model.test_label_metrics, model.test_logit_metrics)
                  for k, v in metrics.compute().items()}
        reference = final[metric_key][0]
        errors = {k: abs(value - reference[k]) for k, value in native.items()}
        assert max(errors.values()) < 1e-5, errors
        y = labels.numpy()
        p = logits.softmax(1)[:, 1].numpy()
        z = logits[:, 1].numpy()
        cls = logits.argmax(1).numpy()
        row = {"checkpoint_sha256": digest(path), "checkpoint_epoch": state["epoch"],
               "checkpoint_step": state["global_step"], "native_metrics": native,
               "native_reload_max_abs_error": max(errors.values()),
               "diagnostic_metrics": {"accuracy": accuracy_score(y, cls),
                   "balanced_accuracy": balanced_accuracy_score(y, cls),
                   "global_raw_logit_auroc": roc_auc_score(y, z),
                   "global_raw_logit_ap": average_precision_score(y, z),
                   "softmax_auroc": roc_auc_score(y, p),
                   "softmax_ap": average_precision_score(y, p)},
               "labels": np.bincount(y, minlength=2).tolist(),
               "predictions": np.bincount(cls, minlength=2).tolist()}
        # Raw global-score metrics and native batch-updated TorchMetrics are
        # separate: auto-sigmoid handling can depend on the input batch range.
        pred_path = results / f"upstream2025-{kind}-predictions.pt"
        with pred_path.open("xb") as f:
            torch.save({"logits": logits, "labels": labels}, f)
        row["predictions_sha256"] = digest(pred_path)
        audit["checkpoints"][kind] = row
        del model, state
        torch.cuda.empty_cache()
    with (results / "upstream2025-audit.json").open("x") as f:
        json.dump(audit, f, indent=2)
    print("UPSTREAM_FULL_TEST_AUDIT_OK", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
