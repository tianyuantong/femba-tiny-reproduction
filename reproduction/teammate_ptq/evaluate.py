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
"""Capture frozen validation/test logits using the pinned teammate evaluator."""

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from threshold_control import binary_metrics, ranking_metrics

BASE_COMMIT = "a2a4d441485efd8423c44b35d7448a30ccd44437"
MODEL_IDS = ("fp32", "ptq_original", "ptq_p99", "qap_b")
PILOT_SPLITS = {"val": (6426, [2927, 3499]), "test": (7211, [3304, 3907])}
SPLITS = dict(PILOT_SPLITS)


def use_data_splits(data: Path) -> dict:
    """Expected window and class counts: from <data>/data-manifest.json when present, else the pilot data."""
    manifest = data / "data-manifest.json"
    if manifest.is_file():
        splits = json.loads(manifest.read_text())["splits"]
        expected = {}
        for split in ("val", "test"):
            counts = json.loads((data / f"{split}-manifest.json").read_text())["label_counts"]
            expected[split] = (splits[split]["windows"], [counts.get("0", 0), counts.get("1", 0)])
    else:
        expected = PILOT_SPLITS
    SPLITS.clear()
    SPLITS.update(expected)
    return SPLITS
SOURCE_FILES = (
    "models/FEMBA_int8.py", "models/femba_ptq.py", "tasks/finetune_task.py",
    "models/femba_quantization.py", "models/femba_ternary.py",
    "datasets/hdf5_dataset.py", "data_module/finetune_data_module.py",
    "ARES/tests/test_networks/test_25_femba_tiny_int8.py",
    "ARES/tests/test_networks/brevitas_custom_layers.py", "util/train_utils.py",
)
QUANTIZATION_FIELDS = ("weight_bit_width", "activation_bit_width",
                       "weight_quant_scheme", "activation_quant_scheme")


def add_quantization_arguments(parser, *, infer: bool = False) -> None:
    """Evaluation infers omitted fields; new training/calibration defaults to W8A8."""
    for role in ("weight", "activation"):
        parser.add_argument(f"--{role}-bit-width", type=int, default=None if infer else 8)
        parser.add_argument(f"--{role}-quant-scheme", default=None if infer else "uniform")


def argument_quantization_config(args, *, infer: bool = False) -> dict | None:
    # Source-specific imports must follow insertion of the requested --source path.
    from models.femba_quantization import quantization_config

    supplied = {key: getattr(args, key, None) for key in QUANTIZATION_FIELDS
                if getattr(args, key, None) is not None}
    canonical = quantization_config(**supplied)
    return ({key: canonical[key] for key in supplied} or None) if infer else canonical


def checkpoint_quantization_config(checkpoint: Mapping, expected_config=None) -> dict:
    from models.femba_quantization import config_from_mapping, quantization_config

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Checkpoint must contain hyper_parameters")
    model_config = hyperparameters.get("model", {})
    actual = config_from_mapping(model_config)
    metadata = getattr(checkpoint.get("state_dict"), "_metadata", {}).get("model", {})
    if "femba_quantization" in metadata:
        saved = metadata["femba_quantization"]
        if (config_from_mapping(saved) != actual or not isinstance(saved.get("enabled"), bool)
                or saved["enabled"] != model_config.get("quantization_enabled", True)):
            raise ValueError("Checkpoint hyperparameters and state_dict quantization metadata differ")
    if expected_config is not None:
        if not isinstance(expected_config, Mapping):
            raise TypeError("Expected quantization configuration must be a mapping")
        unknown = set(expected_config) - set(QUANTIZATION_FIELDS)
        if unknown:
            raise ValueError(f"Unknown quantization configuration fields: {sorted(unknown)}")
        validated = quantization_config(**expected_config)
        for key in expected_config:
            if validated[key] != actual[key]:
                raise ValueError(f"Checkpoint quantization configuration mismatch: {key} "
                                 f"is {actual[key]!r}, requested {validated[key]!r}")
    return actual


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def decision_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    """Use z1 > z0; a tie selects class zero, matching two-class argmax."""
    labels, logits = np.asarray(labels), np.asarray(logits)
    if labels.ndim != 1 or logits.shape != (len(labels), 2):
        raise ValueError("Expected labels [N] and logits [N,2]")
    if not np.isfinite(logits).all():
        raise ValueError("Expected finite logits")
    margin = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    return {**binary_metrics(labels, margin, threshold=0.0), "windows": len(labels),
            "class_counts": np.bincount(labels, minlength=2).tolist(),
            "scores": {"logit_margin": ranking_metrics(labels, margin)}}


def source_record(source: Path) -> dict:
    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(source), *arguments], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != BASE_COMMIT:
        raise ValueError(f"Expected teammate source commit {BASE_COMMIT}, got {commit}")
    return {"repository": "https://github.com/berumottovv00/BioFoundation",
            "base_commit": commit, "dirty": bool(git("status", "--porcelain")),
            "files_sha256": {name: sha256(source / name) for name in SOURCE_FILES},
            "capture_script_sha256": sha256(Path(__file__))}


def configure_runtime() -> dict:
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("The frozen replay requires a CUDA GPU")
    if importlib.metadata.version("brevitas") != "0.12.1":
        raise RuntimeError("The frozen quantizer contract requires Brevitas 0.12.1")
    OmegaConf.register_new_resolver("env", os.getenv, replace=True)
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return {"seed": 42, "batch_size": 32, "workers": 4, "precision": "32-true",
            "matmul_precision": "high", "tf32_matmul": True, "tf32_cudnn": True,
            "cudnn_benchmark": True, "gpu": torch.cuda.get_device_name(0),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "pytorch-lightning", "torchmetrics", "brevitas", "numpy", "h5py")}}


def safe_load_checkpoint(path: Path) -> dict:
    import torch
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    with torch.serialization.safe_globals([DictConfig, ListConfig, Metadata,
            ContainerMetadata, AnyNode, dict, list, int, Any, defaultdict]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_task(path: Path, model_id: str, *, expected_config=None, device="cuda",
              target_config=None, num_classes=2, classification_type="bc"):
    """Strictly reload; only an FP32 checkpoint may adopt an explicit PTQ target."""
    from omegaconf import OmegaConf
    from models.femba_quantization import quantization_config
    from tasks.finetune_task import FinetuneTask

    checkpoint = safe_load_checkpoint(path)
    cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    config = checkpoint_quantization_config(checkpoint, expected_config)
    is_enabled = cfg.model.get("quantization_enabled", True)
    if not isinstance(is_enabled, bool) or is_enabled != (model_id != "fp32"):
        raise ValueError("Checkpoint quantization mode differs from requested model kind")
    if target_config is not None:
        if model_id != "fp32" or is_enabled:
            raise ValueError("Only FP32 weights may be converted to a target quantization configuration")
        config = quantization_config(**target_config)
    for key, value in config.items():
        cfg.model[key] = value
    if cfg.model._target_ != "models.FEMBA_int8.FEMBATinyInt8Classifier":
        raise ValueError("Expected the teammate ARES binary classifier")
    if cfg.model.num_classes != num_classes or cfg.model.classification_type != classification_type:
        raise ValueError(f"Expected {num_classes}-class {classification_type} classification")
    norm = cfg.input_normalization
    if not norm.normalize or (norm.quartile_normalization_lower_val,
                             norm.quartile_normalization_upper_val) != (-20, 20):
        raise ValueError("Expected the fixed [-20,20] normalization")
    task = FinetuneTask(cfg)
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    task.eval()
    contract = quantization_contract(task.model, enabled=model_id != "fp32")
    return task.to(device).eval(), contract


def quantization_contract(model, enabled: bool) -> dict:
    import torch
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
    from brevitas.proxy.quant_proxy import QuantProxyFromInjector
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
    from models.femba_quantization import model_quantization_config

    config = model_quantization_config(model)
    ranks = [list(parameter.shape) for name, parameter in model.named_parameters()
             if name.endswith(".dt_proj.weight")]
    assert ranks == [[1540, 96]] * 4
    proxies = {name: proxy for name, proxy in model.named_modules()
               if isinstance(proxy, QuantProxyFromInjector) and proxy.is_quant_enabled}
    if not enabled:
        assert not proxies and model.quantization_enabled is False
        return {"enabled": False, "dt_proj_shapes": ranks, "configuration": config}
    acts = {name: proxy for name, proxy in proxies.items()
            if isinstance(proxy, ActQuantProxyFromInjectorBase)}
    weights = {name: proxy for name, proxy in proxies.items()
               if isinstance(proxy, WeightQuantProxyFromInjectorBase)}
    assert len(proxies) == 53 and len(acts) == 35 and len(weights) == 18
    actual_widths = {}
    for role, selected in (("activation", acts), ("weight", weights)):
        actual_widths[role] = {name: int(proxy.bit_width().item()) for name, proxy in selected.items()}
        assert set(actual_widths[role].values()) == {config[f"{role}_bit_width"]}
        for proxy in selected.values():
            scale = proxy.scale()
            assert scale.numel() == 1
            assert torch.isfinite(scale).all() and (scale > 0).all()
    depthwise = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.Conv1d) and module.groups == module.in_channels
                 and module.in_channels > 1]
    assert len(depthwise) == 4
    assert all(not any(name.startswith(layer + ".") for name in weights) for layer in depthwise)
    for proxy in acts.values():
        scaling = proxy.fused_activation_quant_proxy.tensor_quant.scaling_impl
        assert not scaling.training and scaling.counter > scaling.collect_stats_steps
    return {"enabled": True, "activation_quantizers": 35, "weight_quantizers": 18,
            "dt_proj_shapes": ranks, "initialized_frozen_scales": True,
            "configuration": config, "actual_proxy_bit_widths": actual_widths,
            "fp32_depthwise_weights": depthwise}


def frozen_snapshot(model) -> tuple:
    from brevitas.core.scaling.standalone import ParameterFromRuntimeStatsScaling

    tensors = {kind + ":" + name: value.detach().cpu().clone()
               for kind, values in (("parameter", model.named_parameters()),
                                    ("buffer", model.named_buffers())) for name, value in values}
    counters = {name: (module.counter, module.collect_stats_steps)
                for name, module in model.named_modules()
                if isinstance(module, ParameterFromRuntimeStatsScaling)}
    return tensors, counters


def make_capture():
    import pytorch_lightning as pl
    import torch
    from biofoundation.core.batch import as_signal_batch

    class Capture(pl.Callback):
        def on_test_start(self, trainer, pl_module):
            self.parts = defaultdict(list)
            self.hook = pl_module.model.register_forward_hook(self.capture)

        def capture(self, module, inputs, output):
            logits = output[0].detach()
            assert logits.ndim == 2 and logits.shape[1] == 2
            positive = logits[:, 1]
            outside = ((positive < 0) | (positive > 1)).any()
            probabilities = logits.softmax(dim=1)
            for name, value in {"logits": logits, "native_softmax_p1": probabilities[:, 1],
                    "native_predictions": probabilities.argmax(dim=1),
                    "native_batch_scores": torch.where(outside, positive.sigmoid(), positive)}.items():
                self.parts[name].append(value)
            self.parts["batch_auto_sigmoid"].append(outside)

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            labels = as_signal_batch(batch)["label"].long().reshape(-1).detach()
            assert len(labels) == len(self.parts["logits"][-1])
            self.parts["labels"].append(labels)
            self.parts["batch_sizes"].append(len(labels))

        def on_test_end(self, trainer, pl_module):
            self.hook.remove()

        def arrays(self):
            return {name: (np.asarray(parts, dtype=np.int64) if name == "batch_sizes" else
                    (torch.stack(parts) if name == "batch_auto_sigmoid" else torch.cat(parts)).cpu().numpy())
                    for name, parts in self.parts.items()}

    return Capture()


def replay(task, dataset) -> tuple[dict, dict]:
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from data_module.finetune_data_module import FinetuneDataModule

    before = frozen_snapshot(task.model)
    capture = make_capture()
    data = FinetuneDataModule(train=None, val=None, test=dataset,
        cfg=OmegaConf.create({"batch_size": 32, "num_workers": 4}), name="eeg_tuar")
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="32-true", logger=False,
        enable_checkpointing=False, callbacks=[capture], benchmark=True)
    native = trainer.test(task, datamodule=data, verbose=False)[0]
    after = frozen_snapshot(task.model)
    assert before[1] == after[1] and before[0].keys() == after[0].keys()
    assert all(torch.equal(value, after[0][name]) for name, value in before[0].items())
    return native, capture.arrays()


def save_capture(output: Path, model_id: str, split: str, arrays: dict,
                 native: dict, inputs: dict, contract: dict) -> dict:
    import math

    row = decision_metrics(arrays["labels"], arrays["logits"])
    expected_count, expected_classes = SPLITS[split]
    assert row["windows"] == expected_count and row["class_counts"] == expected_classes
    predictions = arrays["logits"].argmax(axis=1)
    assert np.array_equal(predictions, arrays["native_predictions"])
    assert np.array_equal(predictions, arrays["native_softmax_p1"] > 0.5)
    assert all(math.isfinite(value) for value in native.values())
    assert math.isclose(native["test_BinaryAccuracy"], row["accuracy"], abs_tol=1e-7)
    assert math.isclose(native["test_MulticlassRecall"], row["balanced_accuracy"], abs_tol=1e-7)
    path = output / f"{model_id}_{split}_logits.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    row.update({"model": model_id, "split": split, "array_sha256": sha256(path),
                "labels_sha256": hashlib.sha256(arrays["labels"].tobytes()).hexdigest(),
                "checkpoint_sha256": inputs["checkpoints"][model_id],
                "data_sha256": inputs["data"][split], "native_metrics": native,
                "model_contract": contract, "logits_file": path.name,
                "strict_reload": True, "weights_and_scales_unchanged": True})
    write_json(output / f"{model_id}_{split}.json", row)
    return row


def run(args) -> None:
    from contextlib import ExitStack

    source, data, output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    use_data_splits(data)
    sys.path.insert(0, str(source))
    expected_config = argument_quantization_config(args, infer=True)
    provenance = source_record(source)
    checkpoints = {name: getattr(args, name).resolve() for name in MODEL_IDS}
    inputs = {"checkpoints": {name: sha256(path) for name, path in checkpoints.items()},
              "data": {split: sha256(data / f"{split}.h5") for split in SPLITS}}
    for model_id, path in checkpoints.items():
        checkpoint_quantization_config(safe_load_checkpoint(path),
                                       expected_config if model_id != "fp32" else None)
    output.mkdir(parents=True, exist_ok=False)
    runtime = configure_runtime()
    write_json(output / "provenance.json", {"source": provenance, "inputs": inputs, "runtime": runtime})
    from datasets.hdf5_dataset import HDF5Loader

    rows, labels = [], {}
    # Complete every validation capture before opening the test dataset.
    for split in SPLITS:
        with ExitStack() as resources:
            dataset = HDF5Loader(str(data / f"{split}.h5"), use_cache=False)
            resources.callback(dataset.data.close)
            for model_id in MODEL_IDS:
                print(f"REPLAY {split} {model_id}", flush=True)
                task, contract = load_task(checkpoints[model_id], model_id,
                                           expected_config=expected_config if model_id != "fp32" else None)
                native, arrays = replay(task, dataset)
                if split in labels:
                    assert np.array_equal(labels[split], arrays["labels"])
                else:
                    labels[split] = arrays["labels"].copy()
                rows.append(save_capture(output, model_id, split, arrays, native, inputs, contract))
                del task
    assert source_record(source) == provenance, "Source changed during frozen replay"
    assert inputs["data"] == {split: sha256(data / f"{split}.h5") for split in SPLITS}
    assert inputs["checkpoints"] == {name: sha256(path) for name, path in checkpoints.items()}
    write_json(output / "results.json", {"rows": rows, "threshold_fitted": False,
               "source_data_checkpoints_unchanged": True,
               "ranking_score": "float64(z1)-float64(z0)", "default_decision": "z1 > z0"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "data", "fp32", "ptq-original", "ptq-p99", "qap-b", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    add_quantization_arguments(parser, infer=True)
    run(parser.parse_args())
