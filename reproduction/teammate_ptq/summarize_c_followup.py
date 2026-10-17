# SPDX-License-Identifier: Apache-2.0
"""Normalize audited C43/44 records without changing historical seed-results.json.

This consumes retained metadata and capture bytes, not model weights or datasets.
Independent metric computation is recorded by the queue; normalization verifies
its captures and metadata, then recomputes aggregates and paired differences.
Matplotlib is needed only with --figures.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from summarize_seeds import METRICS, c_handoff, metric, sha256, summary


SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
MODELS = ("fp32", "ptq", "qap_b", "c")
STAGES = tuple(f"seed{seed}-c-{stage}" for seed in NEW_SEEDS
               for stage in ("finetune", "evaluation", "curves"))
RAW_MARKER = "c-followup-20261016/"
PAIRING_SCOPE = ("Seed42 QAP B is a teammate-checkpoint full WSL replay; "
                 "only seeds43/44 have verified matched initialization.")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Records:
    """Hash original bytes; publish only relative identifiers, never raw paths."""

    def __init__(self, root: Path):
        self.root = root
        self.hashes: dict[str, str] = {}
        self.expected: dict[str, str] = {}

    def fingerprint(self, name: str, expected: str | None = None) -> str:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe record identifier")
        digest = sha256(self.root / name)
        target = expected or self.expected.get(name)
        require(target is None or digest == target, f"SHA256 mismatch: {name}")
        self.hashes[name] = digest
        return digest

    def read(self, name: str, *, lines: bool = False):
        self.fingerprint(name)
        content = (self.root / name).read_text()
        if lines:
            return [json.loads(line) for line in content.splitlines() if line]
        return json.loads(content)

    def bind_queue_hashes(self, hashes: dict[str, str]) -> None:
        for name, digest in hashes.items():
            if RAW_MARKER in name and not name.endswith(".ckpt"):
                relative = name.split(RAW_MARKER, 1)[1]
                require(relative not in self.expected, "Duplicate original record")
                self.expected[relative] = digest
        require(bool(self.expected), "Queue contains no follow-up record hashes")
        for name, digest in self.expected.items():
            self.fingerprint(name, digest)


def aggregate(rows: list[dict], seeds: tuple[int, ...] = SEEDS) -> dict:
    indexed = {(row["seed"], row["model"]): row for row in rows}
    require(len(indexed) == len(rows), "Duplicate seed/model row")
    require(set(indexed) == {(seed, model) for seed in SEEDS for model in MODELS},
            "Expected all twelve seed/model rows")
    return {model: {"n": len(seeds), "seeds": list(seeds),
                    "metrics": {name: summary([metric(indexed[seed, model], name)
                                                for seed in seeds]) for name in METRICS}}
            for model in MODELS}


def qap_minus_c(rows: list[dict], seeds: tuple[int, ...]) -> dict:
    indexed = {(row["seed"], row["model"]): row for row in rows}
    metrics = {}
    for name in METRICS:
        values = [metric(indexed[seed, "qap_b"], name) - metric(indexed[seed, "c"], name)
                  for seed in seeds]
        metrics[name] = {"by_seed": dict(zip(map(str, seeds), values)), **summary(values)}
    return {"seeds": list(seeds), "metrics": metrics}


def validate_stages(stages: list[dict]) -> list[dict]:
    require(tuple(row["stage"] for row in stages) == STAGES, "Missing, repeated or reordered stage")
    previous_end = None
    for row in stages:
        start, end = (datetime.fromisoformat(row[key]) for key in ("started_at", "ended_at"))
        require(row["returncode"] == 0 and start <= end, "Stage failed or has invalid timestamps")
        require(previous_end is None or previous_end <= start, "Queue stages overlap")
        previous_end = end
    return [{key: row[key] for key in ("stage", "started_at", "ended_at", "returncode")}
            for row in stages]


def validate_budget(complete: dict, config: dict, epochs: list[dict], seed: int) -> None:
    require(complete["status"] == "complete" and complete["epochs"] == 5
            and complete["global_step"] == 8675, "C fine-tuning must complete 5 epochs / 8675 steps")
    require(config["seed"] == seed and config["batch_size"] == 32
            and config["trainer"]["max_epochs"] == 5 and config["resume"] is False,
            "Unexpected seed, batch size, epoch budget or resume")
    require(config["model"]["quantization_enabled"] is True, "C must enable quantization")
    require(config["model_checkpoint"]["monitor"] == "val_loss"
            and config["model_checkpoint"]["mode"] == "min", "Checkpoint selection changed")
    require([row["epoch"] for row in epochs] == list(range(5)), "Missing training epoch")
    require([row["global_step"] for row in epochs] == [1735 * n for n in range(1, 6)],
            "Training step budget changed")
    require(all(math.isfinite(value) for row in epochs for value in row["metrics"].values()),
            "Nonfinite training metric")
    best = min(epochs, key=lambda row: row["metrics"]["val_loss"])
    require(complete["best_validation_loss"] == best["metrics"]["val_loss"], "Wrong best validation loss")
    expected = f"epoch={best['epoch']}-step={best['global_step']}.ckpt"
    require(Path(complete["best_checkpoint"]).name == expected, "Checkpoint is not native val_loss best")


def check_source(records: Records, protocol: dict, previous: dict) -> dict:
    snapshot = protocol["source_snapshot"]
    require(snapshot["source_dirty"] is True, "Executed source dirty state changed")
    require(snapshot["source_base_commit"] == previous["source"]["base_commit"], "Source base differs")
    for name, digest in snapshot["runners"].items():
        records.fingerprint(name, digest)
    patch = records.fingerprint("runners/source.patch")
    require(patch == previous["source"]["reconstruction_patch_sha256"], "Reconstruction patch differs")
    deployment = records.read("deployment-manifest.json")
    require(deployment["entrypoint_dirty"] is True, "Runner deployment dirty state changed")
    for name, digest in deployment["files"].items():
        records.fingerprint(name, digest)
    require(records.hashes["baseline-results.json"] ==
            previous["provenance"]["historical_results_sha256"], "Historical replay changed")
    source_bytes = json.dumps(snapshot["source"], sort_keys=True, separators=(",", ":")).encode()
    return {"repository": previous["source"]["repository"],
            "base_commit": snapshot["source_base_commit"], "dirty": True,
            "source_inventory_files": len(snapshot["source"]),
            "source_inventory_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_inventory_hash_encoding": "UTF-8 JSON, sorted keys, compact separators",
            "reconstruction_patch": "source.patch", "reconstruction_patch_sha256": patch,
            "executed_files_sha256": {}, "runners_sha256": snapshot["runners"],
            "entrypoint_deployment": {"base_commit": deployment["base_commit"], "dirty": True,
                                      "patch_sha256": records.hashes["entrypoint-changes.patch"]}}


def training_run(records: Records, seed: int, protocol: dict, source: dict) -> dict:
    prefix = f"seeds/{seed}/c/finetune"
    complete = records.read(f"{prefix}/complete.json")
    manifest = records.read(f"{prefix}/run_manifest.json")
    epochs = records.read(f"{prefix}/epochs.jsonl", lines=True)
    curves = records.read(f"{prefix}/curves/summary.json")
    validate_budget(complete, manifest["config"], epochs, seed)
    fit = records.read(f"{prefix}/fit_summary.json")
    require(all(fit[key] == complete[key] for key in fit), "Fit and completion disagree")
    require(curves["nonfinite_events"] == 0 and curves["scalar_events"] == 16155,
            "Incomplete or nonfinite curve capture")
    require(manifest["source_dirty"] is True and manifest["source_base_commit"] == source["base_commit"],
            "Training source provenance changed")
    require(manifest["wrapper_sha256"] == source["runners_sha256"]["runners/train_stage.py"],
            "Training runner fingerprint differs")
    executed = manifest["executed_files_sha256"]
    snapshot = protocol["source_snapshot"]["source"]
    require(all(snapshot[name] == digest for name, digest in executed.items()), "Executed source differs")
    require(not source["executed_files_sha256"] or source["executed_files_sha256"] == executed,
            "Seeds used different source files")
    source["executed_files_sha256"] = executed
    parent = protocol["parents"][str(seed)]
    require(parent["status"] == "complete" and parent["epochs"] == 10
            and parent["global_step"] == 15610, "FP32 parent budget differs")
    return {"seed": seed, "model": "c", "stage": "finetune", "source_id": "c_followup",
            "epochs": 5, "global_step": 8675, "best_checkpoint_sha256": complete["best_checkpoint_sha256"],
            "best_validation_loss": complete["best_validation_loss"],
            "best_epoch": min(epochs, key=lambda row: row["metrics"]["val_loss"])["epoch"] + 1,
            "parent": {"model": "fp32", "seed": seed, "stage": "pretrain", "epochs": 10,
                       "global_step": 15610, "checkpoint_sha256": parent["best_checkpoint_sha256"]},
            "optimizer": manifest["config"]["optimizer"], "curve_scalar_events": curves["scalar_events"],
            "loss": [{"epoch": row["epoch"] + 1, "train": row["metrics"]["train_loss_epoch"],
                      "val": row["metrics"]["val_loss"]} for row in epochs]}


def validate_pair(records: Records, seed: int, previous: dict, run: dict) -> dict:
    prefix = f"seeds/{seed}/c/finetune"
    handoff = c_handoff(records, seed)
    paired = records.read(f"{prefix}/paired-c-handoff.json")
    initial = records.read(f"{prefix}/initialization.json")
    loaded = records.read(f"{prefix}/pretrained_handoff.json")
    old = next(row for row in previous["matched_seed43_44"]["checks"] if row["seed"] == seed)
    require(all(old[name] for name in ("encoder_and_decoder_initialization_equal",
                                      "pretraining_split_equal", "classifier_initialization_equal")),
            "Old FP32/QAP initialization pairing is absent")
    require(paired["classifier_matches_fp32_qap"] and paired["loaded_encoder_matches_fp32"]
            and loaded["classifier_initialization_preserved"], "C handoff pairing failed")
    classifier = initial["hashes"]["classifier"]["sha256"]
    require(classifier == paired["classifier_sha256"] == old["initial_classifier_sha256"]
            == loaded["hashes_after_load"]["classifier"]["sha256"], "Classifier initialization differs")
    parent = next(row for row in previous["training_runs"]
                  if row["seed"] == seed and row["model"] == "fp32" and row["stage"] == "pretrain")
    require(parent["epochs"] == 10 and parent["global_step"] == 15610, "Historical FP32 parent budget differs")
    require(handoff["pretrained_checkpoint_sha256"] == paired["pretrained_checkpoint_sha256"]
            == run["parent"]["checkpoint_sha256"] == parent["best_checkpoint_sha256"],
            "C used a different FP32 parent")
    return {"seed": seed, **handoff, "classifier_initialization_sha256": classifier,
            "loaded_encoder_sha256": loaded["hashes_after_load"]["encoder"]["sha256"],
            "fp32_qap_initial_encoder_sha256": old["initial_encoder_sha256"],
            "classifier_matches_fp32_qap": True, "loaded_encoder_matches_fp32": True,
            "fp32_qap_pretraining_split_equal": True}


def validate_evaluation_protocol(protocol: dict, source: dict, seed: int) -> None:
    require(protocol["training_seed"] == seed and protocol["model"] == "c", "Evaluation identity differs")
    runtime = protocol["runtime"]
    require(runtime["batch_size"] == 32 and runtime["seed"] == 42
            and runtime["precision"] == "32-true", "Evaluator conditions differ")
    executed = protocol["source"]
    require(executed["dirty"] is True and executed["base_commit"] == source["base_commit"],
            "Evaluation source provenance changed")
    require(all(source["executed_files_sha256"][name] == digest
                for name, digest in executed["files_sha256"].items()), "Evaluation source files differ")
    require(protocol["runner_sha256"] == source["runners_sha256"]["runners/evaluate_one.py"]
            and executed["capture_script_sha256"] == source["runners_sha256"]["runners/evaluate.py"],
            "Evaluation runners differ")


def evaluation_row(records: Records, seed: int, data: dict, run: dict, captures: dict, source: dict) -> dict:
    prefix = f"seeds/{seed}/c/evaluation"
    result = records.read(f"{prefix}/results.json")
    validate_evaluation_protocol(records.read(f"{prefix}/protocol.json"), source, seed)
    require(result["seed"] == seed and result["model"] == "c", "Evaluation result identity differs")
    selection = records.read(f"{prefix}/selection.json")
    started = records.read(f"{prefix}/test_started.json")
    require(selection["time_utc"] < started["time_utc"], "Threshold was not locked before test")
    require(started["selection_sha256"] == result["selection_sha256"]
            == records.hashes[f"{prefix}/selection.json"], "Selection fingerprint differs")
    require(result["source_data_checkpoint_unchanged"] and result["strict_reload_each_split"],
            "Evaluation integrity failed")
    require(result["checkpoint_sha256"] == run["best_checkpoint_sha256"], "Wrong evaluation checkpoint")
    require(result["validation_selected"]["threshold"] == selection["threshold"]
            and selection["checkpoint_sha256"] == result["checkpoint_sha256"]
            and selection["validation_array_sha256"] == result["validation"]["array_sha256"],
            "Threshold selection does not match the evaluated checkpoint and validation capture")
    for split, key in (("val", "validation"), ("test", "test")):
        capture = records.read(f"{prefix}/c_{split}.json")
        require(capture == result[key] and capture["checkpoint_sha256"] == result["checkpoint_sha256"],
                "Capture metadata differs from evaluation result")
        require(capture["windows"] == data[split]["windows"] and capture["data_sha256"] == data[split]["sha256"],
                "Evaluation dataset differs")
        require(capture["strict_reload"] and capture["weights_and_scales_unchanged"], "Strict reload failed")
        contract = capture["model_contract"]
        require(contract == {"enabled": True, "activation_quantizers": 35, "weight_quantizers": 18,
                             "dt_proj_shapes": [[1540, 96]] * 4, "initialized_frozen_scales": True},
                "Quantization coverage, rank or scale contract differs")
        independent = captures[seed, split]
        records.fingerprint(f"{prefix}/c_{split}_logits.npz", capture["array_sha256"])
        require(independent["npz_sha256"] == capture["array_sha256"]
                and independent["labels_sha256"] == capture["labels_sha256"]
                and independent["windows"] == capture["windows"]
                and independent["confusion_matrices_match"], "Independent capture verification differs")
        require(all(abs(independent[name] - capture["scores"]["logit_margin"][name]) <= 1e-12
                    for name in ("auroc", "ap")), "Independent margin ranking differs")
    return {"seed": seed, "model": "c", "origin": "same-seed FP32 pretrain, quantized fine-tuning",
            "source_id": "c_followup", "checkpoint_sha256": result["checkpoint_sha256"],
            **result["test"]["scores"]["logit_margin"], "default": result["default"],
            "validation_threshold": result["validation_selected"]}


def initial_records(previous: dict) -> tuple[list[dict], list[dict]]:
    require(previous["score"] == "float64(z1)-float64(z0)", "Historical ranking score changed")
    rows = deepcopy(previous["per_seed"] + [previous["c_seed42"]])
    for row in rows:
        row["source_id"] = "historical_seed42" if row["seed"] == 42 and row["model"] != "c" else "previous_seed_runs"
        if row["seed"] == 42 and row["model"] == "qap_b":
            require(row["origin"] == "teammate checkpoint replay", "Seed42 QAP origin changed")
            row["origin"] = "teammate checkpoint, full WSL replay"
    runs = [deepcopy(row) for row in previous["training_runs"] if row["model"] == "c"]
    require(len(runs) == 1 and runs[0]["seed"] == 42, "Missing historical C42 training record")
    runs[0]["source_id"] = "previous_seed_runs"
    return rows, runs


def validate_queue_summary(queue: dict, rows: list[dict], totals: dict, differences: dict) -> None:
    indexed = {row["seed"]: row for row in rows if row["model"] == "c"}
    require(len(queue["per_seed"]) == 3 and {row["seed"] for row in queue["per_seed"]} == set(SEEDS),
            "Incomplete or repeated queue C rows")
    for row in queue["per_seed"]:
        require(row["checkpoint_sha256"] == indexed[row["seed"]]["checkpoint_sha256"], "Queue checkpoint differs")
        require(all(row[name] == metric(indexed[row["seed"]], name) for name in METRICS), "Queue metrics differ")
    require(queue["aggregate"] == totals["c"]["metrics"], "Queue aggregate differs")
    require(queue["qap_b_minus_c"] == differences, "Queue paired differences differ")


def build(raw: Path, previous_path: Path) -> dict:
    records = Records(raw)
    previous = json.loads(previous_path.read_text())
    queue = records.read("c-results.json")
    records.bind_queue_hashes(queue["original_records_sha256"])
    protocol = records.read("c-protocol.json")
    require(protocol["seeds"] == list(NEW_SEEDS) and protocol["pretrain_epochs"] == 0
            and protocol["finetune_epochs"] == 5 and protocol["batch_size"] == 32
            and protocol["dt_rank"] == 96 and protocol["resume"] is False, "Follow-up protocol changed")
    require(records.read("c-status.json")["phase"] == "complete"
            and queue["source_data_runners_unchanged"], "Queue not complete or integrity failed")
    stages = validate_stages(records.read("c-stages.jsonl", lines=True))
    require(queue["completed_stage_count"] == len(stages), "Queue stage count differs")
    data = records.read("c-data-before.json")["data"]
    require(data == records.read("c-data-after.json")["data"] == previous["data"], "Dataset fingerprints differ")
    source = check_source(records, protocol, previous)
    independent = records.read("independent-metrics.json")
    captures = {(row["seed"], row["split"]): row for row in independent["captures"]}
    require(independent["sklearn_verified"] and independent["count"] == len(captures) == len(independent["captures"]) == 4
            and set(captures) == {(seed, split) for seed in NEW_SEEDS for split in ("val", "test")},
            "Independent metric audit is incomplete")
    rows, runs = initial_records(previous)
    pairs = []
    for seed in NEW_SEEDS:
        run = training_run(records, seed, protocol, source)
        pair = validate_pair(records, seed, previous, run)
        rows.append({**evaluation_row(records, seed, data, run, captures, source), "handoff": pair})
        runs.append(run)
        pairs.append(pair)
    rows.sort(key=lambda row: (row["seed"], MODELS.index(row["model"])))
    totals = aggregate(rows)
    differences = {"42_43_44": qap_minus_c(rows, SEEDS), "43_44": qap_minus_c(rows, NEW_SEEDS)}
    validate_queue_summary(queue, rows, totals, differences)
    gpu = records.read("c-gpu-exit.json")
    require(gpu == queue["gpu_release"] and not gpu["training_evaluation_processes"]
            and gpu["gpu_utilization_percent"] <= 5 and gpu["gpu_used_mib"] < 512, "GPU was not released")
    return assemble(previous, previous_path, records, source, rows, runs, pairs, totals,
                    differences, stages, independent, gpu, queue["completed_at"])


def assemble(previous: dict, previous_path: Path, records: Records, source: dict,
             rows: list[dict], runs: list[dict], pairs: list[dict], totals: dict,
             differences: dict, stages: list[dict], independent: dict, gpu: dict, completed: str) -> dict:
    evaluation = records.read("seeds/43/c/evaluation/protocol.json")
    return {"study": previous["study"], "dataset_scope": "One TUAR dataset; recording-level split",
            "score": previous["score"], "decision": "positive iff margin > threshold; ties are class 0",
            "data": previous["data"], "training": previous["training"], "calibration": previous["calibration"],
            "per_seed": rows, "three_seed_aggregate": totals, "qap_b_minus_c": differences,
            "pairing_scope": PAIRING_SCOPE, "matched_seed43_44": {"checks": pairs, "aggregate": aggregate(rows, NEW_SEEDS)},
            "c_training_runs": runs, "runtime": previous["runtime"],
            "evaluation": {"threshold_protocol": evaluation["threshold_protocol"], "runtime": evaluation["runtime"]},
            "sources": {"historical_seed42": {"training": "results.json#/source",
                                               "ptq": "results.json#/expanded_calibration/source",
                                               "qap_b": "results.json: teammate checkpoint full WSL replay",
                                               "results_sha256": previous["provenance"]["historical_results_sha256"]},
                        "previous_seed_runs": previous["source"], "c_followup": source},
            "verification": {"completed_at": completed, "stages": stages, "new_training_epochs": 10,
                             "new_pretraining_epochs": 0, "new_training_steps": 17350,
                             "new_curve_scalar_events": sum(run["curve_scalar_events"] for run in runs if run["seed"] in NEW_SEEDS),
                             "nonfinite_curve_events": 0, "independent_metrics": independent,
                             "closeout": {key: value for key, value in gpu.items() if key != "training_evaluation_processes"},
                             "training_and_evaluation_exited": True},
            "provenance": {"normalization": "Selected fields only; raw bytes remain outside the repository. All original names are relative to the retained follow-up bundle.",
                           "previous_results_file": "seed-results.json", "previous_results_sha256": sha256(previous_path),
                           "summary_script_sha256": sha256(Path(__file__)),
                           "summary_dependency_sha256": sha256(Path(__file__).with_name("summarize_seeds.py")),
                           "original_records_sha256": dict(sorted(records.hashes.items()))}}


def plot_results(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10,
                                "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "femba-c-seeds"})
    colors = ("#287d68", "#ac7711", "#3659a0")
    rows = {(row["seed"], row["model"]): row for row in result["per_seed"]}
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for offset, seed, color in zip((-.07, 0, .07), SEEDS, colors):
        ax.plot([x + offset for x in range(4)], [rows[seed, model]["auroc"] for model in MODELS],
                "o--", color=color, linewidth=1, label=f"Seed {seed}")
    for x, model in enumerate(MODELS):
        stats = result["three_seed_aggregate"][model]["metrics"]["auroc"]
        ax.errorbar(x, stats["mean"], yerr=stats["sample_std"], color="#202a30", capsize=6)
        ax.text(x, .923, f"{stats['mean']:.4f} ± {stats['sample_std']:.4f}", ha="center", fontsize=9)
    ax.set(xticks=range(4), xticklabels=("FP32", "PTQ INT8", "QAP B", "FP32 pretrain\n+ QAT (C)"),
           ylim=(.85, .932), ylabel="Test margin AUROC", title="TUAR: per-seed results and mean ± sample SD")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="lower right", frameon=False, ncols=3)
    fig.supxlabel("One dataset; recording-level split. Seed42 QAP: teammate replay. Strict initialization pairing: seeds43/44.", fontsize=8)
    fig.savefig(output / "c-seed-auroc.svg", metadata={"Date": None})
    plt.close(fig)
    plot_losses(result["c_training_runs"], output, plt)


def plot_losses(runs: list[dict], output: Path, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout="constrained")
    for ax, run in zip(axes, runs):
        losses = run["loss"]
        x = [row["epoch"] for row in losses]
        ax.plot(x, [row["train"] for row in losses], "o--", color="#6c8791", label="Train")
        ax.plot(x, [row["val"] for row in losses], "o-", color="#287d68", label="Validation")
        best = min(losses, key=lambda row: row["val"])
        ax.scatter(best["epoch"], best["val"], marker="*", s=180, color="#ac7711", zorder=4, label="Selected checkpoint")
        ax.set(title=f"C · Seed {run['seed']}", xlabel="Fine-tuning epoch", xticks=range(1, 6))
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("FP32 pretraining → five epochs of quantized fine-tuning", fontsize=13)
    fig.supxlabel("Native validation loss selects the checkpoint; test data is not used for checkpoint selection.", fontsize=9)
    fig.savefig(output / "c-training-loss.svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path(__file__).with_name("seed-results.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    result = build(args.raw, args.previous)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        plot_results(result, args.figures)


if __name__ == "__main__":
    main()
