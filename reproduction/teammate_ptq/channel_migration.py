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
"""Move the outlier channels of the second Mamba block's scan outputs into out_proj before PTQ.

For one FP32 parent of a binary task, on the protocol's 512 calibration windows with
quantization off: the per-channel maximum of each migrated scan output y (the gated input of
its activation quantizer, which out_proj consumes) and of the matching input column of
out_proj. For a strength alpha, channel c of y is divided by
s_c = max|y_c|^alpha / max|W_c|^(1 - alpha) before the quantizer and column c of out_proj is
multiplied by s_c, which leaves the float model unchanged. Each candidate (no migration, or
one alpha) is calibrated as in the formal sweep, at one percentile for every activation
quantizer, and scored by validation AUROC; the best is evaluated on test with all quantizers
and with weight quantizers only.
"""

import argparse
import json
from pathlib import Path
import sys

import calibrate
import evaluate as capture
from compare_calibration import evaluate_task, quantize_only, replay_split
from sweep import validation_metrics
from threshold_control import timestamp

ALPHAS = (None, 0.25, 0.5, 0.75, 0.85, 1.0)  # None: no migration
MIGRATED = ("mamba_blocks.1.mamba_fwd", "mamba_blocks.1.mamba_rev")
FLOAT_TOLERANCE = 1e-2  # largest |logit change| allowed in float; TF32 rounding alone gives about 3e-4


def load_parent(fp32: Path, config: dict):
    task, _ = capture.load_task(fp32, "fp32", target_config=config, num_classes=2, classification_type="bc")
    return task


def float_logits(task, signals):
    import torch
    from models.femba_ptq import set_quantization_enabled

    set_quantization_enabled(task.model, False)
    with torch.no_grad():
        return task.model(signals, torch.zeros_like(signals, dtype=torch.bool))[0]


def channel_maxima(task, batches) -> dict:
    """Per-channel max |y| at the input of each migrated scan-output quantizer, quantization off."""
    import torch
    from models.femba_ptq import set_quantization_enabled

    modules, maxima = dict(task.model.named_modules()), {}

    def keep(prefix: str, y) -> None:
        current = y.detach().abs().amax(dim=(0, 1))
        maxima[prefix] = torch.maximum(maxima[prefix], current) if prefix in maxima else current

    hooks = [modules[prefix + ".ssm.output_quant"].register_forward_pre_hook(
        lambda module, inputs, prefix=prefix: keep(prefix, inputs[0])) for prefix in MIGRATED]
    set_quantization_enabled(task.model, False)
    try:
        with torch.no_grad():
            for signals in batches:
                task.model(signals, torch.zeros_like(signals, dtype=torch.bool))
    finally:
        for hook in hooks:
            hook.remove()
    return maxima


def migrate(model, maxima: dict, alpha: float) -> dict:
    """Divide each channel by s before its scan-output quantizer and multiply its out_proj column by s."""
    import torch

    modules, factors = dict(model.named_modules()), {}
    for prefix in MIGRATED:
        weight = modules[prefix + ".out_proj"].weight
        factors[prefix] = maxima[prefix] ** alpha / weight.detach().abs().amax(dim=0) ** (1 - alpha)
        with torch.no_grad():
            weight.mul_(factors[prefix])  # in place: this model instance stays migrated
        modules[prefix + ".ssm.output_quant"].register_forward_pre_hook(
            lambda module, inputs, factor=factors[prefix]: (inputs[0] / factor,))
    return factors


def candidate(args, config: dict, dataset, indices: list[int], maxima: dict, reference: dict, alpha) -> tuple:
    """A fresh parent, migrated when alpha is set, calibrated and scored on validation."""
    from models.femba_ptq import calibrate_quantized

    task, record = load_parent(args.fp32, config), {"alpha": alpha}
    if alpha is not None:
        factors = migrate(task.model, maxima, alpha)
        change = (float_logits(task, reference["signals"]) - reference["logits"]).abs().max().item()
        if change > FLOAT_TOLERANCE:
            raise RuntimeError(f"Migration with alpha={alpha} changed the float logits by {change}")
        record.update(float_logit_change=change, factors={
            prefix: {"min": factor.min().item(), "median": factor.median().item(), "max": factor.max().item()}
            for prefix, factor in factors.items()})
    calibrate.set_percentile(task.model, args.percentile)
    report = calibrate_quantized(task.model, calibrate.batches(task, dataset, indices))
    record["migrated_scales"] = {prefix: report["scales"][prefix + ".ssm.output_quant.act_quant"] for prefix in MIGRATED}
    val = replay_split(task, args.data, "val")
    record["validation_auroc"] = validation_metrics(val["labels"], val["logits"])["auroc"]
    return task, record


def run(args) -> dict:
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    from datasets.hdf5_dataset import HDF5Loader

    config = capture.argument_quantization_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = capture.configure_runtime()
    dataset = HDF5Loader(str(args.data / "train.h5"), use_cache=False)
    try:
        indices = calibrate.calibration_indices(len(dataset))
        parent = load_parent(args.fp32, config)
        maxima = channel_maxima(parent, calibrate.batches(parent, dataset, indices))
        signals = next(calibrate.batches(parent, dataset, indices[:32]))
        reference = {"signals": signals, "logits": float_logits(parent, signals)}
        tasks, candidates = {}, []
        for alpha in ALPHAS:
            print(f"CANDIDATE alpha={alpha}", flush=True)
            tasks[alpha], record = candidate(args, config, dataset, indices, maxima, reference, alpha)
            candidates.append(record)
    finally:
        dataset.data.close()
    chosen = max(candidates, key=lambda record: record["validation_auroc"])["alpha"]
    print(f"TEST alpha={chosen}", flush=True)
    best = tasks[chosen]
    evaluation = {"all": evaluate_task(best, args.data)}
    quantize_only(best.model, "weights")
    evaluation["weights_only"] = evaluate_task(best, args.data)
    result = {"time_utc": timestamp(), "fp32_sha256": capture.sha256(args.fp32), "quantization_config": config,
              "activation_percentile": args.percentile, "migrated": list(MIGRATED),
              "calibration": {"split": "train", "windows": len(indices), "train_sha256": capture.sha256(args.data / "train.h5")},
              "selection_rule": "highest validation AUROC among no migration and alpha in "
                                + str([alpha for alpha in ALPHAS if alpha is not None]),
              "candidates": candidates, "chosen_alpha": chosen, "chosen": evaluation,
              "data": {split: capture.sha256(args.data / f"{split}.h5") for split in ("val", "test")},
              "runtime": runtime, "source": capture.source_record(args.source)}
    capture.write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "fp32", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Activation percentile for every quantizer (the formal sweep's selection)")
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps({"chosen_alpha": result["chosen_alpha"],
                      "validation": {str(record["alpha"]): round(record["validation_auroc"], 4)
                                     for record in result["candidates"]},
                      "test_auroc": round(result["chosen"]["all"]["test"]["auroc"], 4),
                      "weights_only_test_auroc": round(result["chosen"]["weights_only"]["test"]["auroc"], 4)}))


if __name__ == "__main__":
    main()
