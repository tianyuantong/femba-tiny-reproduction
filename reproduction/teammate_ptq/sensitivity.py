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
"""Which activation quantizers cost a binary-task W8A8 PTQ model its accuracy, and how many need FP16.

On validation only: the AUROC with each activation quantizer alone enabled (weights stay INT8,
the other activations float) ranks the quantizers by their drop from the weights-only model.
The top k then run in FP16 and the rest in INT8 for each k in FALLBACK_K; the smallest k whose
validation AUROC is within TOLERANCE of the weights-only model is chosen (otherwise the best k),
and only that configuration is tested. Also reports the share of activation elements in FP16.
"""

import argparse
import json
from pathlib import Path
import sys

import calibrate
import evaluate as capture
from compare_calibration import evaluate_variant, quantize_only, replay_split
from ptq_ablation import float_activations, tail_ranking
from sweep import validation_metrics
from threshold_control import timestamp

FALLBACK_K = (1, 2, 3, 4, 6, 8, 12)
TOLERANCE = 0.01


def fp16_activations(model, names: list[str]) -> None:
    """Replace the named activation quantizers by an FP16 round trip; the rest stay INT8."""
    float_activations(model, names)
    modules = dict(model.named_modules())
    for name in names:
        modules[name].register_forward_hook(lambda module, inputs, output: output.half().float())


def validation_auroc(checkpoint: Path, prepare, data: Path, config: dict) -> float:
    task, _ = capture.load_task(checkpoint, "ptq_original", expected_config=config)
    prepare(task.model)
    arrays = replay_split(task, data, "val")
    return validation_metrics(arrays["labels"], arrays["logits"])["auroc"]


def activation_elements(checkpoint: Path, data: Path, config: dict, names: list[str]) -> dict:
    """Elements each activation quantizer produces per input window."""
    import torch
    from brevitas.quant_tensor import QuantTensor
    from datasets.hdf5_dataset import HDF5Loader

    def count(name: str, output) -> None:
        value = output.value if isinstance(output, QuantTensor) else output
        counts[name] = value.numel() // value.shape[0]

    task, _ = capture.load_task(checkpoint, "ptq_original", expected_config=config)
    modules, counts = dict(task.model.named_modules()), {}
    hooks = [modules[name].register_forward_hook(lambda module, inputs, output, name=name: count(name, output))
             for name in names]
    dataset = HDF5Loader(str(data / "val.h5"), use_cache=False)
    try:
        with torch.no_grad():
            signals = next(calibrate.batches(task, dataset, list(range(8))))
            task.model(signals, torch.zeros_like(signals, dtype=torch.bool))
    finally:
        dataset.data.close()
        for hook in hooks:
            hook.remove()
    return counts


def run(args) -> dict:
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    config = capture.argument_quantization_config(args)
    if capture.safe_load_checkpoint(args.checkpoint).get("parent_sha256") != capture.sha256(args.fp32):
        raise ValueError("The PTQ checkpoint was not calibrated from this FP32 parent")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = capture.configure_runtime()
    names = tail_ranking(args.tail_sweep)
    reference = {"all": validation_auroc(args.checkpoint, lambda model: None, args.data, config),
                 "weights_only": validation_auroc(args.checkpoint, lambda model: quantize_only(model, "weights"),
                                                  args.data, config)}
    alone = {}
    for name in names:
        print(f"ALONE {name}", flush=True)
        others = [other for other in names if other != name]
        alone[name] = validation_auroc(args.checkpoint, lambda model: float_activations(model, others), args.data, config)
    ranking = sorted(names, key=lambda name: alone[name])  # largest drop from weights-only first
    capture.write_json(args.output / "sensitivity.json", {"reference": reference, "alone": alone, "ranking": ranking})
    fallback = {}
    for k in FALLBACK_K:
        print(f"FP16 top {k}", flush=True)
        fallback[k] = validation_auroc(args.checkpoint, lambda model: fp16_activations(model, ranking[:k]),
                                       args.data, config)
    within = [k for k in FALLBACK_K if fallback[k] >= reference["weights_only"] - TOLERANCE]
    chosen = within[0] if within else max(FALLBACK_K, key=fallback.get)
    print(f"TEST FP16 top {chosen}", flush=True)
    test = evaluate_variant(args.checkpoint, lambda model: fp16_activations(model, ranking[:chosen]), args.data, config)
    elements = activation_elements(args.checkpoint, args.data, config, names)
    result = {"time_utc": timestamp(), "fp32_sha256": capture.sha256(args.fp32), "quantization_config": config,
              "checkpoint_sha256": capture.sha256(args.checkpoint),
              "selection_rule": f"smallest k in {list(FALLBACK_K)} within {TOLERANCE} of the weights-only validation AUROC",
              "reference_validation_auroc": reference, "alone_validation_auroc": alone, "ranking": ranking,
              "fp16_fallback_validation_auroc": {str(k): value for k, value in fallback.items()},
              "chosen_k": chosen, "fp16_quantizers": ranking[:chosen], "fp16_variant": test,
              "fp16_element_share": sum(elements[name] for name in ranking[:chosen]) / sum(elements.values()),
              "elements_per_window": elements,
              "data": {split: capture.sha256(args.data / f"{split}.h5") for split in ("val", "test")},
              "runtime": runtime, "source": capture.source_record(args.source)}
    capture.write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "fp32", "checkpoint", "tail-sweep", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps({"chosen_k": result["chosen_k"], "test_auroc": round(result["fp16_variant"]["test"]["auroc"], 4),
                      "fp16_element_share": round(result["fp16_element_share"], 4)}))


if __name__ == "__main__":
    main()
