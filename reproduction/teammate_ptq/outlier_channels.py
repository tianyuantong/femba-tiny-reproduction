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
"""Whether the largest scan-output activations of a binary-task PTQ model sit in a few channels.

Reads the calibrated per-tensor clipping threshold of the four scan-output activation
quantizers of one PTQ checkpoint, then disables the activation quantizers (weights stay
quantized) and records their inputs on a fixed subset of validation windows. For each it
reports the share of the largest 0.1% of |x| held by the top 1 / 4 / 16 / 77 channels
(77 = 5% of 1,540) and by the top 1 / 4 time steps, how many channels never exceed the
threshold, and the largest per-channel maximum over the median one.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import calibrate
import evaluate as capture
from compare_calibration import quantize_only
from threshold_control import timestamp

WINDOWS = 1024
WINDOW_SEED = 0
TOP_FRACTION = 1e-3
CHANNEL_COUNTS = (1, 4, 16, 77)
STEP_COUNTS = (1, 4)
SCAN_OUTPUTS = tuple(f"mamba_blocks.{block}.mamba_{direction}.ssm.output_quant.act_quant"
                     for block in (0, 1) for direction in ("fwd", "rev"))


def concentration(x, threshold: float) -> dict:
    """x: (channels, windows, steps) inputs of one activation quantizer."""
    magnitude = x.abs()
    cut = magnitude.flatten().topk(math.ceil(TOP_FRACTION * magnitude.numel())).values.min()
    large = magnitude >= cut
    by_channel = large.sum(dim=(1, 2)).sort(descending=True).values.double()
    by_step = large.sum(dim=(0, 1)).sort(descending=True).values.double()
    channel_share, step_share = by_channel.cumsum(0) / by_channel.sum(), by_step.cumsum(0) / by_step.sum()
    channel_max = magnitude.amax(dim=(1, 2))
    return {"threshold": threshold, "fraction_above_threshold": (magnitude > threshold).double().mean().item(),
            "top_share_by_channels": {str(n): channel_share[n - 1].item() for n in CHANNEL_COUNTS},
            "channels_for_90_percent_of_top": int((channel_share < 0.9).sum()) + 1,
            "top_share_by_steps": {str(n): step_share[n - 1].item() for n in STEP_COUNTS},
            "channels_never_above_threshold": int((channel_max <= threshold).sum()),
            "max_over_median_channel_max": (channel_max.max() / channel_max.median()).item()}


def clipping_thresholds(model) -> dict:
    """Calibrated clipping threshold of each scan output: its scale times the integer scaling."""
    modules = dict(model.named_modules())
    thresholds = {}
    for name in SCAN_OUTPUTS:
        tensor_quant = modules[name].fused_activation_quant_proxy.tensor_quant
        integer_scaling = tensor_quant.int_scaling_impl(tensor_quant.msb_clamp_bit_width_impl())
        thresholds[name] = float(modules[name].scale()) * float(integer_scaling)
    return thresholds


def record_inputs(task, data: Path) -> dict:
    """Inputs of the scan-output quantizers as (channels, windows, steps) on fixed validation windows."""
    import torch
    from datasets.hdf5_dataset import HDF5Loader

    modules, recorded = dict(task.model.named_modules()), {name: [] for name in SCAN_OUTPUTS}

    def keep(name: str, x) -> None:
        channel_dim = 1 if x.shape[1] > x.shape[2] else 2  # 1,540 channels against 80 time steps
        recorded[name].append(x.detach().permute(channel_dim, 0, 3 - channel_dim))

    hooks = [modules[name].register_forward_pre_hook(lambda module, inputs, name=name: keep(name, inputs[0]))
             for name in SCAN_OUTPUTS]
    dataset = HDF5Loader(str(data / "val.h5"), use_cache=False)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(WINDOW_SEED))
    try:
        with torch.no_grad():
            for signals in calibrate.batches(task, dataset, sorted(order[:WINDOWS].tolist())):
                task.model(signals, torch.zeros_like(signals, dtype=torch.bool))
    finally:
        dataset.data.close()
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(parts, dim=1) for name, parts in recorded.items()}


def run(args) -> dict:
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    config = capture.argument_quantization_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = capture.configure_runtime()
    task, _ = capture.load_task(args.checkpoint, "ptq_original", expected_config=config)
    thresholds = clipping_thresholds(task.model)
    quantize_only(task.model, "weights")
    inputs = record_inputs(task, args.data)
    result = {"time_utc": timestamp(), "quantization_config": config, "checkpoint_sha256": capture.sha256(args.checkpoint),
              "data": {"val": capture.sha256(args.data / "val.h5")},
              "windows": {"split": "val", "count": WINDOWS, "seed": WINDOW_SEED}, "top_fraction": TOP_FRACTION,
              "scan_outputs": {name: concentration(inputs[name], thresholds[name]) for name in SCAN_OUTPUTS},
              "runtime": runtime, "source": capture.source_record(args.source)}
    capture.write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps({name.removesuffix(".ssm.output_quant.act_quant"): stats["channels_for_90_percent_of_top"]
                      for name, stats in result["scan_outputs"].items()}))


if __name__ == "__main__":
    main()
