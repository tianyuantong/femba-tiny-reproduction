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
"""Where a binary-task PTQ model loses accuracy, and whether MSE or KL clipping recovers it.

For one FP32 fine-tuned parent: calibrates MSE and KL activation clipping on the protocol's
512 training windows, then evaluates on validation and test the formal sweep's selected
percentile checkpoint with all quantizers, with weight quantizers only and with activation
quantizers only, and the MSE and KL checkpoints. Each variant picks its threshold on
validation. Supplementary to the formal sweep: nothing here selects among methods.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import calibrate
import evaluate as capture
from sweep import validation_metrics
from threshold_control import binary_metrics, ranking_metrics, timestamp

VARIANTS = (("percentile", "all"), ("percentile", "weights"), ("percentile", "activations"),
            ("mse", "all"), ("kl", "all"))


def quantize_only(model, role: str) -> None:
    """Keep one role's quantizers; with float activations no layer returns quant tensors."""
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase

    if role == "all":
        return
    disabled = ActQuantProxyFromInjectorBase if role == "weights" else WeightQuantProxyFromInjectorBase
    for module in model.modules():
        if isinstance(module, disabled):
            module.disable_quant = True
        if role == "weights" and hasattr(module, "return_quant_tensor"):
            module.return_quant_tensor = False


def replay_split(task, data: Path, split: str) -> dict:
    from datasets.hdf5_dataset import HDF5Loader

    dataset = HDF5Loader(str(data / f"{split}.h5"), use_cache=False)
    try:
        return capture.replay(task, dataset)[1]
    finally:
        dataset.data.close()


def evaluate_variant(checkpoint: Path, prepare, data: Path, config: dict) -> dict:
    """Validation picks the threshold; prepare(model) adjusts the loaded quantizers first."""
    task, _ = capture.load_task(checkpoint, "ptq_original", expected_config=config)
    prepare(task.model)
    val, test = replay_split(task, data, "val"), replay_split(task, data, "test")
    selection = validation_metrics(val["labels"], val["logits"])
    threshold = selection["threshold_selection"]["threshold"]
    margin = test["logits"][:, 1].astype(np.float64) - test["logits"][:, 0].astype(np.float64)
    return {"validation": {key: selection[key] for key in ("auroc", "ap", "loss")}, "threshold": threshold,
            "test": {**ranking_metrics(test["labels"], margin),
                     "default": binary_metrics(test["labels"], margin, 0.0),
                     "validation_selected": binary_metrics(test["labels"], margin, threshold)}}


def calibrate_method(args, method: str, percentile: float = 99.999) -> tuple[Path, dict]:
    directory = args.output / (f"p{percentile:g}" if method == "percentile" else method)
    result = calibrate.run(SimpleNamespace(source=args.source, data=args.data, checkpoint=args.fp32,
        output=directory, method=method, percentile=percentile, calibration_windows=512,
        num_classes=2, classification_type="bc", **capture.argument_quantization_config(args)))
    return directory / result["checkpoint_file"], result


def run(args) -> dict:
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    config = capture.argument_quantization_config(args)
    parent_sha = capture.sha256(args.fp32)
    if capture.safe_load_checkpoint(args.percentile_checkpoint).get("parent_sha256") != parent_sha:
        raise ValueError("The percentile checkpoint was not calibrated from this FP32 parent")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = capture.configure_runtime()
    checkpoints, scales = {"percentile": args.percentile_checkpoint}, {}
    for method in ("mse", "kl"):
        print(f"CALIBRATE {method}", flush=True)
        checkpoints[method], result = calibrate_method(args, method)
        scales[method] = result["calibration"]["scales"]
    rows = []
    for method, role in VARIANTS:
        print(f"EVALUATE {method} {role}", flush=True)
        rows.append({"calibration": method, "quantized": role, "checkpoint_sha256": capture.sha256(checkpoints[method]),
                     **evaluate_variant(checkpoints[method], lambda model: quantize_only(model, role), args.data, config)})
    result = {"time_utc": timestamp(), "fp32_sha256": parent_sha, "quantization_config": config,
              "data": {split: capture.sha256(args.data / f"{split}.h5") for split in ("val", "test")},
              "variants": rows, "scales": scales, "runtime": runtime, "source": capture.source_record(args.source)}
    capture.write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "fp32", "percentile-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps({row["calibration"] + "/" + row["quantized"]: round(row["test"]["auroc"], 4)
                      for row in result["variants"]}))


if __name__ == "__main__":
    main()
