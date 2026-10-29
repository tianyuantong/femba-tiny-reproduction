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
"""Calibrate native quantizers at independent W/A widths on 512 fixed windows."""

import argparse
import math
from numbers import Real
from pathlib import Path
import sys

from evaluate import (add_quantization_arguments, argument_quantization_config, configure_runtime,
                      load_task, quantization_contract, sha256, source_record, write_json)

METHODS = ("percentile", "mse", "kl")


def validate_percentile(percentile: float) -> float:
    if isinstance(percentile, bool) or not isinstance(percentile, Real):
        raise TypeError("Percentile must be a real number")
    percentile = float(percentile)
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("Percentile must be finite and in (0,100]")
    return percentile


def calibration_indices(windows: int, count: int = 512) -> list[int]:
    import torch

    if isinstance(windows, bool) or not isinstance(windows, int) or windows < count:
        raise ValueError(f"The training split must contain at least {count} windows")
    return torch.randperm(windows, generator=torch.Generator().manual_seed(42))[:count].tolist()


def native_observers(model) -> list:
    """The 35 fresh activation proxies whose scales still come from 99.999 percentile observers."""
    import brevitas
    from brevitas.core.scaling.standalone import ParameterFromRuntimeStatsScaling
    from brevitas.core.stats.stats_op import AbsPercentile
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase

    if brevitas.__version__ != "0.12.1":
        raise RuntimeError("Native calibration requires Brevitas 0.12.1")
    proxies = []
    for proxy in model.modules():
        if isinstance(proxy, ActQuantProxyFromInjectorBase) and proxy.fused_activation_quant_proxy is not None:
            scaling = proxy.fused_activation_quant_proxy.tensor_quant.scaling_impl
            if type(scaling) is not ParameterFromRuntimeStatsScaling or scaling.counter != 0:
                raise ValueError("Calibration requires fresh native activation scales")
            stats = scaling.stats.stats_impl
            if type(stats) is not AbsPercentile or stats.stats_reduce_dim is not None or stats.q != 99.999:
                raise ValueError("Expected native per-tensor 99.999 percentile observers")
            proxies.append(proxy)
    if len(proxies) != 35 or len({id(proxy) for proxy in proxies}) != 35:
        raise ValueError("Expected 35 distinct activation observers")
    return proxies


def set_percentile(model, percentile: float) -> None:
    percentile = validate_percentile(percentile)
    for proxy in native_observers(model):
        proxy.fused_activation_quant_proxy.tensor_quant.scaling_impl.stats.stats_impl.q = percentile


def set_clip_search(model, method: str) -> None:
    """Replace every percentile observer with a per-batch MSE or KL clipping search."""
    from clip_search import ClipSearch

    for proxy in native_observers(model):
        tensor_quant = proxy.fused_activation_quant_proxy.tensor_quant
        bit_width = tensor_quant.msb_clamp_bit_width_impl()  # proxies stay disabled until calibration
        tensor_quant.scaling_impl.stats.stats_impl = ClipSearch(
            method, int(tensor_quant.int_quant.min_int(bit_width)), int(tensor_quant.int_quant.max_int(bit_width)),
            float(tensor_quant.int_scaling_impl(bit_width)))


def floating_parameters(model) -> dict:
    from brevitas.core.scaling.standalone import ParameterFromRuntimeStatsScaling

    scales = {name + ".value" for name, module in model.named_modules()
              if isinstance(module, ParameterFromRuntimeStatsScaling)}
    assert len(scales) == 35
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if name not in scales}


def batches(task, dataset, indices):
    from torch.utils.data import DataLoader, Subset

    for signals, _ in DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False):
        yield task.normalize_fct(signals.to(next(task.parameters()).device))


def checkpoint_filename(config: dict) -> str:
    from models.femba_quantization import quantization_config

    config = quantization_config(**config)
    if config["weight_bit_width"] == config["activation_bit_width"] == 8:
        return "ptq_int8.ckpt"
    return f"ptq_w{config['weight_bit_width']}a{config['activation_bit_width']}.ckpt"


def save_reload(task, output: Path, parent_sha: str, recipe: dict, probe) -> dict:
    import pytorch_lightning as pl
    import torch
    from brevitas.proxy.quant_proxy import QuantProxyFromInjector
    from omegaconf import OmegaConf
    from models.femba_quantization import model_quantization_config

    cfg = OmegaConf.create(dict(task.hparams))
    config = model_quantization_config(task.model)
    if recipe.get("quantization_config", config) != config:
        raise ValueError("PTQ recipe configuration differs from the calibrated model")
    recipe = {**recipe, "quantization_config": config}
    cfg.model.quantization_enabled = True
    for key, value in config.items():
        cfg.model[key] = value
    checkpoint = output / checkpoint_filename(config)
    with checkpoint.open("xb") as stream:
        torch.save({"state_dict": task.state_dict(), "hyper_parameters": cfg,
                    "parent_sha256": parent_sha, "ptq_recipe": recipe,
                    "pytorch-lightning_version": pl.__version__}, stream)
    restored, contract = load_task(checkpoint, "ptq_original", expected_config=config, device=probe.device,
                                   num_classes=task.hparams.model.num_classes,
                                   classification_type=task.hparams.model.classification_type)
    quantization_contract(task.model, enabled=True)
    def scales(model):
        return {name: proxy.scale().detach().cpu().clone() for name, proxy in model.named_modules()
                if isinstance(proxy, QuantProxyFromInjector) and proxy.is_quant_enabled}
    expected_scales, actual_scales = scales(task.model), scales(restored.model)
    assert len(expected_scales) == 53 and expected_scales.keys() == actual_scales.keys()
    assert all(torch.equal(value, actual_scales[name]) for name, value in expected_scales.items())
    expected_float, restored_float = floating_parameters(task.model), floating_parameters(restored.model)
    assert expected_float.keys() == restored_float.keys()
    assert all(torch.equal(value, restored_float[name]) for name, value in expected_float.items())
    with torch.no_grad():
        mask = torch.zeros_like(probe, dtype=torch.bool)
        expected, actual = task.model(probe, mask)[0], restored.model(probe, mask)[0]
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {"checkpoint_sha256": sha256(checkpoint), "checkpoint_file": checkpoint.name,
            "strict_reload_exact": True, "floating_parameters_reload_exact": True,
            "all_53_scales_exact": True, "quantization": contract}


def run(args) -> dict:
    import torch

    percentile = validate_percentile(args.percentile)
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from datasets.hdf5_dataset import HDF5Loader
    from models.femba_ptq import calibrate_quantized

    config = argument_quantization_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime, provenance = configure_runtime(), source_record(source)
    parent_sha, train_sha = sha256(args.checkpoint), sha256(args.data / "train.h5")
    task, _ = load_task(args.checkpoint, "fp32", target_config=config,
                        num_classes=args.num_classes, classification_type=args.classification_type)
    initial = floating_parameters(task.model)
    dataset = HDF5Loader(str(args.data / "train.h5"), use_cache=False)
    try:
        indices = calibration_indices(len(dataset), args.calibration_windows)
        manifest = {"split": "train.h5", "seed": 42, "indices": indices, "train_sha256": train_sha}
        manifest_path = args.output / "calibration_indices.json"
        write_json(manifest_path, manifest)
        recipe = {"activation_clipping": args.method,
                  "activation_percentile": percentile if args.method == "percentile" else None,
                  "seed": 42, "batch_size": 32,
                  "windows": args.calibration_windows, "indices_sha256": sha256(manifest_path),
                  "quantization_config": config}
        if args.method == "percentile":
            set_percentile(task.model, percentile)
        else:
            set_clip_search(task.model, args.method)
        report = calibrate_quantized(task.model, batches(task, dataset, indices))
        assert report["num_batches"] == -(-args.calibration_windows // 32)
        current = floating_parameters(task.model)
        assert current.keys() == initial.keys()
        assert all(torch.equal(value, current[name]) for name, value in initial.items())
        probe = next(batches(task, dataset, indices[:32]))
        reload = save_reload(task, args.output, parent_sha, recipe, probe)
    finally:
        dataset.data.close()
    assert source_record(source) == provenance and sha256(args.checkpoint) == parent_sha
    assert sha256(args.data / "train.h5") == train_sha
    result = {"parent_sha256": parent_sha, "train_sha256": train_sha, "recipe": recipe,
              "runtime": runtime, "source": provenance, "calibration": report, **reload,
              "floating_parameters_unchanged": True, "runner_sha256": sha256(Path(__file__))}
    write_json(args.output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--method", choices=METHODS, default="percentile",
                        help="Activation clipping: one percentile for every quantizer, or a per-quantizer MSE / KL search")
    parser.add_argument("--percentile", type=float, default=99.999)
    parser.add_argument("--calibration-windows", type=int, default=512,
                        help="Fixed number of seed-42 training windows used for calibration (default 512)")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--classification-type", default="bc")
    add_quantization_arguments(parser)
    args = parser.parse_args()
    run(args)
    print("CALIBRATION_COMPLETE", args.output.name)


if __name__ == "__main__":
    main()
