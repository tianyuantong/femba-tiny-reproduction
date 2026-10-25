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
"""Freeze one classifier, select its threshold on validation, then test once.

Portable publication of the experiment runner. Historical runs used SHA256
1f04e599a0b7ddf86d8ab76b67f163190626bb665201bf748a6e5180ad569f02;
the public wrapper has a different file hash. Training/evaluation logic is retained.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import evaluate as capture
from threshold_control import PROTOCOL, binary_metrics, ranking_metrics, select_threshold, timestamp


def verify_inputs(args, source_record, checkpoint_sha, hashes):
    assert capture.source_record(args.source) == source_record, "Executed source changed"
    assert capture.sha256(args.checkpoint) == checkpoint_sha, "Checkpoint changed"
    assert {split: capture.sha256(args.data / f"{split}.h5") for split in hashes} == hashes


def replay_split(args, split, inputs):
    from datasets.hdf5_dataset import HDF5Loader

    # Reload independently for each split; replay() checks every parameter/buffer/counter.
    task, contract = capture.load_task(args.checkpoint, args.model_kind,
                                     expected_config=getattr(args, "expected_quantization_config", None))
    dataset = HDF5Loader(str(args.data / f"{split}.h5"), use_cache=False)
    try:
        native, arrays = capture.replay(task, dataset)
        row = capture.save_capture(args.output, args.model_kind, split, arrays, native, inputs, contract)
    finally:
        dataset.data.close()
    margin = arrays["logits"][:, 1].astype(np.float64) - arrays["logits"][:, 0].astype(np.float64)
    return row, arrays["labels"], margin


def run(args):
    sys.path.insert(0, str(args.source))
    checkpoint_sha = capture.sha256(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from the explicit training-stage handoff")
    checkpoint = capture.safe_load_checkpoint(args.checkpoint)
    if checkpoint["hyper_parameters"].get("seed") != args.seed:
        raise ValueError("Checkpoint training seed differs from requested seed")
    args.expected_quantization_config = capture.checkpoint_quantization_config(
        checkpoint, capture.argument_quantization_config(args, infer=True))
    del checkpoint
    source_record = capture.source_record(args.source)
    runtime = capture.configure_runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {"checkpoints": {args.model_kind: checkpoint_sha},
              "data": {"val": capture.sha256(args.data / "val.h5")}}
    capture.write_json(args.output / "protocol.json", {
        "time_utc": timestamp(), "model": args.model_kind, "training_seed": args.seed,
        "threshold_protocol": PROTOCOL, "source": source_record, "runtime": runtime,
        "quantization_config": args.expected_quantization_config,
        "inputs_before_selection": inputs, "runner_sha256": capture.sha256(Path(__file__))})
    validation, labels, margin = replay_split(args, "val", inputs)
    selected = {"time_utc": timestamp(), "model": args.model_kind, "seed": args.seed,
                "checkpoint_sha256": checkpoint_sha, "validation_array_sha256": validation["array_sha256"],
                **select_threshold(labels, margin)}
    selection_path = args.output / "selection.json"
    capture.write_json(selection_path, selected)
    selection_sha = capture.sha256(selection_path)
    verify_inputs(args, source_record, checkpoint_sha, inputs["data"])
    capture.write_json(args.output / "test_started.json", {
        "time_utc": timestamp(), "selection_sha256": selection_sha})
    inputs["data"]["test"] = capture.sha256(args.data / "test.h5")
    test, labels, margin = replay_split(args, "test", inputs)
    verify_inputs(args, source_record, checkpoint_sha, inputs["data"])
    assert capture.sha256(selection_path) == selection_sha, "Validation selection changed during test"
    result = {"model": args.model_kind, "seed": args.seed, "windows": len(labels),
              "quantization_config": args.expected_quantization_config,
              "checkpoint_sha256": checkpoint_sha, "selection_sha256": selection_sha,
              **ranking_metrics(labels, margin), "default": binary_metrics(labels, margin, 0.0),
              "validation_selected": binary_metrics(labels, margin, selected["threshold"]),
              "test": test, "validation": validation, "inputs": inputs,
              "source_data_checkpoint_unchanged": True, "strict_reload_each_split": True,
              "completed_at": timestamp()}
    capture.write_json(args.output / "results.json", result)
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Pinned teammate checkout with source.patch applied")
    parser.add_argument("--data-root", "--data", dest="data", type=Path, required=True,
                        help="Directory containing the fixed val.h5 and test.h5")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Selected fine-tuned classifier checkpoint")
    parser.add_argument("--output", type=Path, required=True,
                        help="New artifact directory outside the source checkout")
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="SHA256 recorded by the completed fine-tuning stage")
    parser.add_argument("--model-kind", choices=("fp32", "qap_b", "c"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    capture.add_quantization_arguments(parser, infer=True)
    args = parser.parse_args(argv)
    for name in ("source", "data", "checkpoint", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output == args.source or args.source in args.output.parents:
        raise ValueError("Evaluation artifacts must stay outside the source checkout")
    if args.output.exists():
        raise ValueError("Output already exists; evaluation must not overwrite a prior selection")
    digest = args.checkpoint_sha256
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Checkpoint SHA256 must contain 64 lowercase hexadecimal characters")
    return args


def main():
    args = arguments()
    result = run(args)
    print("CLASSIFIER_EVALUATION_COMPLETE", json.dumps({key: result[key] for key in
          ("model", "seed", "auroc", "ap", "default", "validation_selected")}), flush=True)


if __name__ == "__main__":
    main()
