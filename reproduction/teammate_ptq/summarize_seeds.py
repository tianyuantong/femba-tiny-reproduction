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
"""Combine retained seed records with the historical seed42 replay.

No model or dataset is loaded. Matplotlib is optional unless --figures is used.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = ("auroc", "ap", "accuracy", "balanced_accuracy",
           "validation_threshold_accuracy", "validation_threshold_balanced_accuracy")
MODELS = ("fp32", "ptq", "qap_b")
COMPARISONS = (("ptq", "fp32"), ("qap_b", "ptq"), ("qap_b", "fp32"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Records:
    """Read originals without mutation and retain their relative names and hashes."""

    def __init__(self, root: Path):
        self.root = root
        self.hashes: dict[str, str] = {}

    def read(self, name: str, *, lines: bool = False):
        path = self.root / name
        self.hashes[name] = sha256(path)
        if lines:
            return [json.loads(line) for line in path.read_text().splitlines() if line]
        return json.loads(path.read_text())


def summary(values: list[float]) -> dict:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("A sample summary needs at least two finite observations")
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def metric(row: dict, name: str) -> float:
    if name.startswith("validation_threshold_"):
        return row["validation_threshold"][name.removeprefix("validation_threshold_")]
    return row[name] if name in ("auroc", "ap") else row["default"][name]


def aggregate(rows: list[dict], seeds: list[int]) -> dict:
    indexed = {(row["seed"], row["model"]): row for row in rows}
    assert len(indexed) == len(rows), "Duplicate seed/model rows"
    return {model: {"n": len(seeds), "seeds": seeds,
                    "metrics": {name: summary([metric(indexed[seed, model], name)
                                                for seed in seeds]) for name in METRICS}}
            for model in MODELS}


def differences(rows: list[dict], seeds: list[int]) -> dict:
    indexed = {(row["seed"], row["model"]): row for row in rows}
    result = {}
    for left, right in COMPARISONS:
        metrics = {}
        for name in METRICS:
            values = [metric(indexed[seed, left], name) - metric(indexed[seed, right], name)
                      for seed in seeds]
            metrics[name] = {"by_seed": dict(zip(map(str, seeds), values)), **summary(values)}
        result[f"{left}_minus_{right}"] = {"n": len(seeds), "metrics": metrics}
    return result


def historical_rows(history: dict) -> list[dict]:
    rows = []
    for old in history["results"]:
        if old["model"] not in ("fp32", "qap_b"):
            continue
        rows.append({"seed": 42, "model": old["model"],
                     "origin": "teammate checkpoint replay" if old["model"] == "qap_b"
                     else "local FP32 training, historical patch",
                     "auroc": old["AUROC"], "ap": old["AP"],
                     "checkpoint_sha256": old["checkpoint_sha256"],
                     "default": old["default"],
                     "validation_threshold": old["test_with_validation_threshold"]})
    expanded = history["expanded_calibration"]
    rows.append({"seed": 42, "model": "ptq", "origin": "historical FP32 parent, extended sweep",
                 "auroc": expanded["test"]["auroc"], "ap": expanded["test"]["ap"],
                 "checkpoint_sha256": expanded["test"]["checkpoint_sha256"],
                 "selected_percentile": expanded["selected_percentile"],
                 "default": expanded["default"],
                 "validation_threshold": expanded["validation_selected"]})
    assert {row["model"] for row in rows} == set(MODELS)
    return sorted(rows, key=lambda row: MODELS.index(row["model"]))


def evaluation_row(records: Records, prefix: str, model: str, seed: int,
                   test_data: dict) -> dict:
    result = records.read(f"{prefix}/results.json")
    test = result["test"]
    assert test["strict_reload"] and test["weights_and_scales_unchanged"]
    assert test["windows"] == test_data["windows"]
    assert test["data_sha256"] == test_data["sha256"]
    assert test["model_contract"]["dt_proj_shapes"] == [[1540, 96]] * 4
    if model != "fp32":
        contract = test["model_contract"]
        assert contract["activation_quantizers"] == 35 and contract["weight_quantizers"] == 18
    selection = records.read(f"{prefix}/selection.json")
    started = records.read(f"{prefix}/test_started.json")
    assert selection["time_utc"] < started["time_utc"]
    assert started["selection_sha256"] == records.hashes[f"{prefix}/selection.json"]
    scores = test["scores"]["logit_margin"]
    row = {"seed": seed, "model": model, "origin": "local original-recipe training",
           **scores, "checkpoint_sha256": test["checkpoint_sha256"],
           "default": {key: test[key] for key in ("threshold", "confusion_matrix", "accuracy",
                                                   "balanced_accuracy", "predicted_positive_rate")},
           "validation_threshold": (test if model == "ptq" else result)["validation_selected"]}
    if model == "ptq":
        integrity = records.read(f"{prefix}/integrity.json")
        assert integrity["source_data_weights_runners_unchanged"]
        row["origin"] = "PTQ from same-seed FP32 parent"
        row["selected_percentile"] = result["selected"]["percentile"]
    return row


def training_runs(records: Records) -> list[dict]:
    runs = []
    for path in sorted((records.root / "seeds").glob("*/*/*/epochs.jsonl")):
        prefix = path.parent.relative_to(records.root).as_posix()
        _, seed, model, stage = prefix.split("/")
        complete = records.read(f"{prefix}/complete.json")
        manifest = records.read(f"{prefix}/run_manifest.json")
        epochs = records.read(f"{prefix}/epochs.jsonl", lines=True)
        curves = records.read(f"{prefix}/curves/summary.json")
        count = 10 if stage == "pretrain" else 5
        assert complete["status"] == "complete" and complete["epochs"] == count
        assert complete["global_step"] == count * (1561 if stage == "pretrain" else 1735)
        assert [row["epoch"] for row in epochs] == list(range(count))
        config = manifest["config"]
        assert config["batch_size"] == 32 and config["seed"] == int(seed)
        assert curves["nonfinite_events"] == 0
        runs.append({"seed": int(seed), "model": model, "stage": stage,
                     "epochs": count, "global_step": complete["global_step"],
                     "best_checkpoint_sha256": complete["best_checkpoint_sha256"],
                     "best_validation_loss": complete["best_validation_loss"],
                     "optimizer": config["optimizer"],
                     "curve_scalar_events": curves["scalar_events"],
                     "loss": [{"epoch": row["epoch"] + 1,
                               "train": row["metrics"]["train_loss_epoch"],
                               "val": row["metrics"]["val_loss"]} for row in epochs]})
    assert len(runs) == 9, "Expected four pretraining and five fine-tuning runs"
    return runs


def c_handoff(records: Records, seed: int) -> dict:
    prefix = f"seeds/{seed}/c/finetune"
    handoff = records.read(f"{prefix}/pretrained_handoff.json")
    first = records.read(f"{prefix}/first_training_batch.json")
    before, after = handoff["observers"], first["observers"]
    assert len(handoff["source_fresh_observers"]) == 34
    assert set(handoff["source_fresh_observers"]).issubset(before)
    assert handoff["target_fresh_observers"] == len(before) == len(after) == 35
    assert all(observer["counter"] == 0 for observer in before.values())
    assert all(observer["buffer"] == observer["value"] == 1 for observer in before.values())
    assert all(0 < observer["counter"] <= observer["collect_stats_steps"]
               for observer in after.values())
    assert all(after[name]["buffer"] != before[name]["buffer"] for name in before)
    assert first["rng_unchanged"] and math.isfinite(first["loss"])
    return {"pretrained_checkpoint_sha256": handoff["pretrained_sha256"],
            "fresh_shared_observers_after_load": 34, "fresh_total_observers_after_load": 35,
            "first_batch_observers_updated": len(after), "first_batch_loss": first["loss"]}


def source_record(raw: Path, history: dict, manifest: dict, protocol: dict) -> dict:
    patch = Path(__file__).with_name("source.patch")
    tracked = (raw / "executed-source.patch").read_bytes()
    assert patch.read_bytes().startswith(tracked)
    return {"repository": history["source"]["repository"],
            "scope": "New seed43/44 and C42 runs; historical seed42 provenance is in results.json",
            "base_commit": manifest["source_base_commit"], "dirty": True,
            "tracked_diff_sha256": sha256(raw / "executed-source.patch"),
            "reconstruction_patch": "source.patch", "reconstruction_patch_sha256": sha256(patch),
            "executed_files_sha256": manifest["executed_files_sha256"],
            "runners_sha256": protocol["source_snapshot"]["runners"]}


def verification_record(records: Records, runs: list[dict]) -> dict:
    independent = records.read("independent-final-metrics.json")
    assert independent["result"] == "PASS" and independent["library"] == "sklearn.metrics"
    assert independent["captures"] == len(independent["records"]) == 26
    for row in independent["records"]:
        original = records.read(row["metadata"])
        assert row["array_sha256"] == original["array_sha256"]
        assert row["windows"] == original["windows"]
        ranking = original["scores"]["logit_margin"]
        assert all(abs(row[name] - ranking[name]) <= 1e-12 for name in ("auroc", "ap"))
    stages = records.read("seed-stages.jsonl", lines=True)
    assert len(stages) == 25 and all(row["returncode"] == 0 for row in stages)
    assert sum(run["epochs"] for run in runs) == 65
    assert sum(run["stage"] == "pretrain" for run in runs) == 4
    assert sum(run["curve_scalar_events"] for run in runs) == 92307
    closeout = records.read("live-closeout.json")
    process_rows = [line.split(maxsplit=3) for line in closeout["processes"][1:]]
    queue_pid = str(records.read("seed-status.json")["pid"])
    queue_exited = all(row[0] != queue_pid for row in process_rows)
    markers = ("seed_queue.py", "train_stage.py", "evaluate_one.py", "run_train.py",
               "tools/calibrate.py", "tools/sweep.py")
    training_exited = all(not any(marker in row[3] for marker in markers)
                          for row in process_rows if len(row) == 4)
    gpu = [int(field.split()[0]) for field in closeout["gpu"].strip().split(",")]
    assert queue_exited and training_exited and gpu == [0, 53, 15998]
    return {"independent_metrics": {"result": independent["result"], "library": independent["library"],
                                    "captures": independent["captures"], "checks": independent["checks"],
                                    "ranking_tolerance": 1e-12},
            "training_stages": 9, "pretraining_runs": 4, "finetuning_runs": 5,
            "total_training_epochs": 65, "successful_queue_stages": len(stages),
            "curve_scalar_events": 92307, "nonfinite_curve_events": 0,
            "closeout": {"time_utc": closeout["time"], "queue_exited": queue_exited,
                         "training_and_evaluation_exited": training_exited,
                         "gpu_utilization_percent": gpu[0], "gpu_used_mib": gpu[1],
                         "gpu_free_mib": gpu[2]}}


def provenance_record(records: Records, history_path: Path) -> dict:
    return {"normalization": "Selected fields only; relative record identifiers replace local paths. Original bytes retained outside the repository.",
            "historical_results_file": "results.json",
            "historical_training_source": "results.json#/source",
            "historical_ptq_source": "results.json#/expanded_calibration/source",
            "historical_results_sha256": sha256(history_path),
            "summary_script_sha256": sha256(Path(__file__)),
            "original_records_sha256": records.hashes}


def build(raw: Path, history_path: Path) -> dict:
    records = Records(raw)
    history = json.loads(history_path.read_text())
    protocol = records.read("seed-protocol.json")
    data = records.read("seed-data-manifest.json")["data"]
    assert records.read("seed-data-after.json")["data"] == data
    assert records.read("seed-status.json")["phase"] == "complete"
    manifest = records.read("seeds/43/fp32/pretrain/run_manifest.json")
    assert manifest["source_dirty"] is True
    rows = historical_rows(history)
    pairs = []
    for seed in protocol["seeds"]:
        for model in MODELS:
            suffix = "ptq-sweep" if model == "ptq" else f"{model}/evaluation"
            rows.append(evaluation_row(records, f"seeds/{seed}/{suffix}", model, seed, data["test"]))
        pre = records.read(f"seeds/{seed}/paired-pretraining.json")
        classifier = records.read(f"seeds/{seed}/paired-classifier.json")
        assert pre["initial_encoder_and_decoder_equal"] and pre["pretrain_splits_equal"]
        assert classifier["initial_classifier_equal"]
        pairs.append({"seed": seed, "encoder_and_decoder_initialization_equal": True,
                      "pretraining_split_equal": True, "classifier_initialization_equal": True,
                      "initial_encoder_sha256": pre["initial_encoder_sha256"],
                      "initial_classifier_sha256": classifier["classifier_sha256"]})
    c_seed = protocol["c_seed"]
    c = evaluation_row(records, f"seeds/{c_seed}/c/evaluation", "c", c_seed, data["test"])
    c["handoff"] = c_handoff(records, c_seed)
    runs = training_runs(records)
    result = {"study": "TUAR FP32, validation-selected INT8 PTQ, QAP B and C",
              "score": "float64(z1)-float64(z0)", "data": data,
              "source": source_record(raw, history, manifest, protocol),
              "training": protocol["training"],
              "calibration": {"windows": protocol["calibration_windows"],
                              "indices_seed": protocol["calibration_indices_seed"],
                              "percentiles": protocol["calibration_percentiles"],
                              "selection": protocol["selection"],
                              "activation_quantizers": 35, "weight_quantizers": 18},
              "per_seed": rows, "three_seed_aggregate": aggregate(rows, [42, 43, 44]),
              "paired_differences": differences(rows, [42, 43, 44]),
              "pairing_scope": "Seed42 QAP B is a teammate checkpoint replay; seeds43/44 verify matched initialization and pretraining splits locally.",
              "matched_seed43_44": {"checks": pairs, "aggregate": aggregate(rows, [43, 44]),
                                     "paired_differences": differences(rows, [43, 44])},
              "c_seed42": c, "training_runs": runs, "runtime": manifest["runtime"],
              "verification": verification_record(records, runs),
              "provenance": provenance_record(records, history_path)}
    encoded = json.dumps(result, allow_nan=False)
    assert not any(token in encoded for token in ("/root/", "/Users/", "XNG-"))
    return result


def plot_results(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                                "font.size": 10, "axes.spines.top": False,
                                "axes.spines.right": False, "svg.hashsalt": "femba-seeds"})
    colors = ("#287d68", "#ac7711", "#3659a0")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, seed, color in zip((-0.07, 0, 0.07), (42, 43, 44), colors):
        rows = {(row["seed"], row["model"]): row for row in result["per_seed"]}
        ax.plot([x + offset for x in range(3)], [rows[seed, m]["auroc"] for m in MODELS],
                "o--", color=color, linewidth=1, label=f"Seed {seed}")
    c = result["c_seed42"]
    ax.scatter([3], [c["auroc"]], marker="D", color="#8b5285", label="C, seed42")
    for x, model in enumerate(MODELS):
        stats = result["three_seed_aggregate"][model]["metrics"]["auroc"]
        ax.errorbar(x, stats["mean"], yerr=stats["sample_std"], color="#202a30", capsize=6)
        ax.text(x, 0.921, f"{stats['mean']:.4f} ± {stats['sample_std']:.4f}", ha="center", fontsize=9)
    ax.text(3, 0.921, f"{c['auroc']:.4f}", ha="center", fontsize=9)
    ax.set(xticks=range(4), xticklabels=("FP32", "PTQ INT8", "QAP B", "FP32 pretrain\n+ QAT (C)"),
           ylim=(0.85, 0.93), ylabel="Test AUROC", title="Same test set: individual seeds and mean ± sample SD")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", frameon=False, ncols=2)
    fig.supxlabel("PTQ percentile chosen on validation. Seed42 QAP: teammate checkpoint replay; seeds43/44: matched local runs.", fontsize=8)
    fig.savefig(output / "seed-auroc.svg", metadata={"Date": None})
    plt.close(fig)
    plot_losses(result, output, plt)


def plot_losses(result: dict, output: Path, plt) -> None:
    runs = sorted(result["training_runs"], key=lambda row: (row["seed"], row["model"], row["stage"]))
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), layout="constrained")
    for ax, run in zip(axes.flat, runs):
        loss = run["loss"]
        x = [row["epoch"] for row in loss]
        ax.plot(x, [row["train"] for row in loss], "o--", ms=3, color="#6c8791", label="Train")
        ax.plot(x, [row["val"] for row in loss], "o-", ms=3, color="#287d68", label="Validation")
        ax.set(title=f"Seed {run['seed']} · {run['model'].upper()} · {run['stage']}",
               xlabel="Epoch", ylabel="Loss", xticks=range(1, run["epochs"] + 1, 2 if run["epochs"] == 10 else 1))
        ax.grid(alpha=0.18)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("New runs: epoch training and validation loss", fontsize=14)
    fig.supxlabel("Pretraining: Smooth L1 reconstruction; fine-tuning: cross-entropy. Each panel has its own loss scale.", fontsize=9)
    fig.savefig(output / "seed-training-loss.svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    result = build(args.raw, args.history)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        plot_results(result, args.figures)


if __name__ == "__main__":
    main()
