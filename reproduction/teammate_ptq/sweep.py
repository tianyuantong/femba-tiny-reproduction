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
"""Extend a fixed PTQ percentile sweep, select on validation, then test once."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import calibrate
import evaluate as capture
from threshold_control import binary_metrics, ranking_metrics, select_threshold, timestamp


CANDIDATES = (("original", 99.999), ("p99_99", 99.99), ("p99_9", 99.9),
              ("p99", 99.0), ("p98", 98.0), ("p95", 95.0), ("p90", 90.0))
REUSE_OPTIONS = ("ptq_original", "ptq_p99_99", "ptq_p99_9", "ptq_p99")
PROTOCOL = {
    "activation_percentiles": [value for _, value in CANDIDATES],
    "calibration": "same FP32 parent; train512 seed42 batch32; all 35A/18W enabled",
    "reuse": "supply all four existing checkpoints to calibrate only98/95/90; otherwise calibrate all seven",
    "score": "float64(z1)-float64(z0)",
    "selection": ["maximum validation AUROC", "maximum validation AP",
                  "minimum validation cross-entropy", "preset candidate order"],
    "threshold": "maximum validation balanced accuracy; ties use smallest absolute then lowest threshold",
    "test": "lock candidate and threshold before opening test; one selected checkpoint only",
    "scope": "best among these seven percentiles; no retraining",
}


def validation_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    result = capture.decision_metrics(labels, logits)
    margin = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    loss = np.logaddexp(0.0, -(2 * labels - 1) * margin).mean()
    return {**result["scores"]["logit_margin"], "loss": float(loss),
            "threshold_selection": select_threshold(labels, margin)}


def choose_candidate(rows: list[dict]) -> dict:
    if len(rows) != len(CANDIDATES):
        raise ValueError("Expected all seven preset validation candidates")
    for row, (name, percentile) in zip(rows, CANDIDATES):
        if row["candidate"] != name or row["percentile"] != percentile or row["split"] != "val":
            raise ValueError("Candidate identity, order, or validation split differs from protocol")
        if not all(np.isfinite(row[key]) for key in ("auroc", "ap", "loss")):
            raise ValueError("Validation selection requires finite metrics")
    best = min(range(len(rows)), key=lambda index: (
        -rows[index]["auroc"], -rows[index]["ap"], rows[index]["loss"], index))
    return rows[best]


def finish(rows: list[dict], output: Path, test_selected) -> dict:
    """The test callback is invoked only after an exclusive immutable selection."""
    selected = choose_candidate(rows)
    selection = {"time_utc": timestamp(), "protocol": PROTOCOL, "selected": selected,
                 "validation_candidates": rows}
    selection_path = output / "selection.json"
    capture.write_json(selection_path, selection)
    selection_sha = capture.sha256(selection_path)
    capture.write_json(output / "test_started.json", {
        "time_utc": timestamp(), "selection_sha256": selection_sha})
    test = test_selected(selected)
    assert capture.sha256(selection_path) == selection_sha, "Selection changed during test"
    result = {"time_utc": timestamp(), "selected": selected, "test": test,
              "selection_sha256": selection_sha, "protocol": PROTOCOL}
    capture.write_json(output / "results.json", result)
    return result


def runner_hashes() -> dict:
    return {name: capture.sha256(Path(__file__).with_name(name)) for name in
            ("sweep.py", "calibrate.py", "evaluate.py", "threshold_control.py")}


def record_inputs(args, checkpoints: dict) -> dict:
    # Test data is deliberately absent until finish() has locked the selection.
    return {"source": capture.source_record(args.source), "runners": runner_hashes(),
            "checkpoints": {name: capture.sha256(path) for name, path in checkpoints.items()},
            "data": {split: capture.sha256(args.data / f"{split}.h5") for split in ("train", "val")}}


def check_parent(path: Path, parent_sha: str, percentile: float, expected_config=None) -> None:
    record = capture.safe_load_checkpoint(path)
    config = capture.checkpoint_quantization_config(record, expected_config)
    if record["hyper_parameters"]["model"].get("quantization_enabled", True) is not True:
        raise ValueError("Reused PTQ checkpoint must have quantization enabled")
    if record.get("parent_sha256") != parent_sha:
        raise ValueError("PTQ checkpoint does not identify the fixed FP32 parent")
    recipe = record.get("ptq_recipe", {})
    if recipe.get("quantization_config", config) != config:
        raise ValueError("PTQ checkpoint recipe and model configurations differ")
    model_config = record["hyper_parameters"]["model"]
    legacy_original = (percentile == 99.999 and "activation_percentile" not in recipe
                       and "quantization_config" not in recipe
                       and not any(field in model_config for field in capture.QUANTIZATION_FIELDS))
    if not legacy_original and recipe.get("activation_percentile") != percentile:
        raise ValueError("PTQ checkpoint recipe differs from its preset percentile")


def weight_scales(model) -> dict:
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase

    values = {name: proxy.scale().detach().cpu().clone() for name, proxy in model.named_modules()
              if isinstance(proxy, WeightQuantProxyFromInjectorBase) and proxy.is_quant_enabled}
    assert len(values) == 18
    return values


def assert_same_tensors(actual: dict, expected: dict) -> None:
    import torch

    assert actual.keys() == expected.keys()
    assert all(torch.equal(value, expected[name]) for name, value in actual.items())


def existing_checkpoints(args) -> dict:
    paths = [getattr(args, name, None) for name in REUSE_OPTIONS]
    supplied = [path is not None for path in paths]
    if any(supplied) and not all(supplied):
        raise ValueError("Supply all four existing PTQ checkpoints, or none")
    return {name: path for (name, _), path in zip(CANDIDATES, paths) if path is not None}


def calibrate_new(args, checkpoints: dict) -> dict:
    manifests = []
    reusing = "original" in checkpoints
    candidates = CANDIDATES[4:] if reusing else CANDIDATES
    for name, percentile in candidates:
        directory = args.output / name
        print(f"CALIBRATE {name}", flush=True)
        # binary classifier calibrated on the protocol's 512 seed-42 training windows
        result = calibrate.run(SimpleNamespace(source=args.source, data=args.data,
            checkpoint=args.fp32, output=directory, percentile=percentile, calibration_windows=512,
            num_classes=2, classification_type="bc", **capture.argument_quantization_config(args)))
        checkpoints[name] = directory / result["checkpoint_file"]
        manifests.append(result["recipe"]["indices_sha256"])
    assert len(set(manifests)) == 1, "Calibration subsets or order changed across percentiles"
    if reusing:
        # Same seed42 manifest used by the original four frozen pilot checkpoints.
        assert manifests[0] == "58886a53d3250e3d803f044aa3bd412bdb5a6a4b49aec38c196f071ff9c804bf"
    return {"indices_sha256": manifests[0], "new_calibrations": len(candidates)}


def validate_candidates(args, checkpoints: dict, inputs: dict, expected_float: dict) -> list[dict]:
    from datasets.hdf5_dataset import HDF5Loader

    dataset = HDF5Loader(str(args.data / "val.h5"), use_cache=False)
    rows, expected_labels, expected_weights = [], None, None
    try:
        for name, percentile in CANDIDATES:
            print(f"VALIDATE {name}", flush=True)
            config = capture.argument_quantization_config(args)
            check_parent(checkpoints[name], inputs["checkpoints"]["fp32"], percentile, config)
            task, contract = capture.load_task(checkpoints[name], "ptq_original", expected_config=config)
            assert_same_tensors(calibrate.floating_parameters(task.model), expected_float)
            actual_weights = weight_scales(task.model)
            if expected_weights is None:
                expected_weights = actual_weights
            assert_same_tensors(actual_weights, expected_weights)
            native, arrays = capture.replay(task, dataset)
            if expected_labels is not None:
                assert np.array_equal(arrays["labels"], expected_labels)
            expected_labels = arrays["labels"].copy()
            metadata = capture.save_capture(args.output, name, "val", arrays, native, inputs, contract)
            rows.append({"candidate": name, "percentile": percentile, "split": "val",
                         "quantization_config": config,
                         "checkpoint_sha256": inputs["checkpoints"][name],
                         "array_sha256": metadata["array_sha256"],
                         "labels_sha256": metadata["labels_sha256"],
                         **validation_metrics(arrays["labels"], arrays["logits"])})
            del task
    finally:
        dataset.data.close()
    capture.write_json(args.output / "validation.json", {"candidates": rows, "test_accessed": False})
    return rows


def test_candidate(args, checkpoints: dict, inputs: dict, selected: dict) -> dict:
    from datasets.hdf5_dataset import HDF5Loader

    name = selected["candidate"]
    assert capture.sha256(checkpoints[name]) == selected["checkpoint_sha256"]
    inputs["data"]["test"] = capture.sha256(args.data / "test.h5")
    task, contract = capture.load_task(checkpoints[name], "ptq_original",
                                      expected_config=capture.argument_quantization_config(args))
    dataset = HDF5Loader(str(args.data / "test.h5"), use_cache=False)
    try:
        native, arrays = capture.replay(task, dataset)
        row = capture.save_capture(args.output, name, "test", arrays, native, inputs, contract)
    finally:
        dataset.data.close()
    margin = arrays["logits"][:, 1].astype(np.float64) - arrays["logits"][:, 0].astype(np.float64)
    threshold = selected["threshold_selection"]["threshold"]
    return {**row, **ranking_metrics(arrays["labels"], margin),
            "validation_selected": binary_metrics(arrays["labels"], margin, threshold)}


def verify_inputs(args, checkpoints: dict, inputs: dict) -> None:
    assert runner_hashes() == inputs["runners"], "Runner changed during sweep"
    assert capture.source_record(args.source) == inputs["source"], "Source changed during sweep"
    assert {name: capture.sha256(path) for name, path in checkpoints.items()} == inputs["checkpoints"]
    assert {split: capture.sha256(args.data / f"{split}.h5") for split in inputs["data"]} == inputs["data"]


def test_and_verify(args, checkpoints: dict, inputs: dict, calibration: dict, selected: dict) -> dict:
    verify_inputs(args, checkpoints, inputs)
    test = test_candidate(args, checkpoints, inputs, selected)
    verify_inputs(args, checkpoints, inputs)
    capture.write_json(args.output / "integrity.json", {
        "source_data_weights_runners_unchanged": True, "inputs": inputs, "calibration": calibration})
    return test


def run(args) -> dict:
    checkpoints = {"fp32": args.fp32, **existing_checkpoints(args)}
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    config = capture.argument_quantization_config(args)
    parent_sha = capture.sha256(args.fp32)
    for name, percentile in CANDIDATES:
        if name in checkpoints:
            check_parent(checkpoints[name], parent_sha, percentile, config)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, runtime = record_inputs(args, checkpoints), capture.configure_runtime()
    capture.write_json(args.output / "protocol.json", {
        "time_utc": timestamp(), **PROTOCOL, "inputs": inputs, "runtime": runtime,
        "quantization_config": config})
    parent, _ = capture.load_task(args.fp32, "fp32")
    expected_float = calibrate.floating_parameters(parent.model)
    del parent
    calibration = calibrate_new(args, checkpoints)
    inputs["checkpoints"].update({name: capture.sha256(checkpoints[name]) for name, _ in CANDIDATES
                                   if name not in inputs["checkpoints"]})
    rows = validate_candidates(args, checkpoints, inputs, expected_float)
    return finish(rows, args.output, partial(test_and_verify, args, checkpoints, inputs, calibration))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "fp32", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in REUSE_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            help="Supply all four existing PTQ checkpoints, or none")
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps({"status": "complete", "selected": result["selected"]["candidate"],
                      "auroc": result["test"]["auroc"]}, allow_nan=False))


if __name__ == "__main__":
    main()
