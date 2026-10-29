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
"""Split a binary-task PTQ model's loss between weights and activations at one weight setting.

Uses an existing percentile checkpoint or calibrates one at the requested W/A setting, then
evaluates it with all quantizers, with weight quantizers only, with activation quantizers
only, and with the K activation quantizers that have the heaviest tails left in float.
Tails are ranked by the ratio of the 99.999 to the 90 percentile scale that the formal
sweep calibrated for the same parent, so no ranking looks at labels.
"""

import argparse
import json
from pathlib import Path
import sys

import evaluate as capture
from compare_calibration import calibrate_method, evaluate_variant, quantize_only
from threshold_control import timestamp


def tail_ranking(sweep: Path) -> list[str]:
    """Activation quantizers ordered from the heaviest tail: 99.999 over 90 percentile scale."""
    heavy = json.loads((sweep / "original" / "results.json").read_text())["calibration"]["scales"]
    light = json.loads((sweep / "p90" / "results.json").read_text())["calibration"]["scales"]
    ratios = {name: heavy[name] / light[name] for name in heavy if heavy[name] != light[name]}
    if len(ratios) != 35:
        raise ValueError(f"Expected 35 activation scales that depend on the percentile, found {len(ratios)}")
    return sorted(ratios, key=ratios.get, reverse=True)


def float_activations(model, names: list[str]) -> None:
    """Leave the named activation quantizers in float; every layer then passes plain tensors."""
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase

    modules = dict(model.named_modules())
    for name in names:
        if not isinstance(modules[name], ActQuantProxyFromInjectorBase):
            raise ValueError(f"Not an activation quantizer: {name}")
        modules[name].disable_quant = True
    for module in model.modules():
        if hasattr(module, "return_quant_tensor"):
            module.return_quant_tensor = False


def run(args) -> dict:
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    config = capture.argument_quantization_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = capture.configure_runtime()
    if args.checkpoint is None:
        print(f"CALIBRATE p{args.percentile:g}", flush=True)
        checkpoint, _ = calibrate_method(args, "percentile", args.percentile)
    else:
        checkpoint = args.checkpoint
    if capture.safe_load_checkpoint(checkpoint).get("parent_sha256") != capture.sha256(args.fp32):
        raise ValueError("The PTQ checkpoint was not calibrated from this FP32 parent")
    ranking = tail_ranking(args.tail_sweep)
    variants = [(role, lambda model, role=role: quantize_only(model, role), [])
                for role in ("all", "weights", "activations")]
    variants += [(f"float_top{k}", lambda model, k=k: float_activations(model, ranking[:k]), ranking[:k])
                 for k in args.float_top_k]
    rows = []
    for name, prepare, floated in variants:
        print(f"EVALUATE {name}", flush=True)
        rows.append({"variant": name, "float_activations": floated,
                     **evaluate_variant(checkpoint, prepare, args.data, config)})
    result = {"time_utc": timestamp(), "fp32_sha256": capture.sha256(args.fp32), "quantization_config": config,
              "checkpoint_sha256": capture.sha256(checkpoint), "tail_ranking": ranking,
              "data": {split: capture.sha256(args.data / f"{split}.h5") for split in ("val", "test")},
              "variants": rows, "runtime": runtime, "source": capture.source_record(args.source)}
    capture.write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "fp32", "tail-sweep", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Existing PTQ checkpoint; otherwise calibrate --percentile")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--float-top-k", type=int, nargs="*", default=[])
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps({row["variant"]: round(row["test"]["auroc"], 4) for row in result["variants"]}))


if __name__ == "__main__":
    main()
