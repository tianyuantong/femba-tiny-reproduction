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
"""Run one unchanged native training stage with read-only provenance hooks.

Portable publication of the experiment runner. Historical runs used SHA256
04a3898f063fcdf36b987e80e04573e6fcd530a5189c9eee9bb288f7c55739b9;
the public wrapper has a different file hash. Training/evaluation logic is retained.
The portable entrypoint additionally checks parent checkpoint metadata before training.
"""

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType

BASE_COMMIT = "a2a4d441485efd8423c44b35d7448a30ccd44437"
SPLITS = {
    "train": (55524, 12508987552, "9fb802613f4256e6fb15ba1bd5c0853a22b983567ae48b0c8b88ac64c123d8b5"),
    "val": (6426, 1447713360, "e293d3a96ee12be7673fd67485b5ea3cc179ca8b86a102f684aa93e72fe78711"),
    "test": (7211, 1624566488, "fb9950063c5d877f01d93d096af7a09208b12e4241a6cc6946ed1f905a03cd97"),
}
SOURCE_FILES = (
    "run_train.py", "models/FEMBA_int8.py", "models/femba_ptq.py", "models/femba_decoder.py",
    "tasks/pretrain_task.py", "tasks/finetune_task.py", "datasets/hdf5_dataset.py",
    "data_module/pretrain_data_module.py", "data_module/finetune_data_module.py",
    "ARES/tests/test_networks/test_25_femba_tiny_int8.py",
    "ARES/tests/test_networks/brevitas_custom_layers.py", "util/train_utils.py",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def rng_digest():
    import numpy as np
    import torch

    digest = hashlib.sha256(repr(random.getstate()).encode())
    numpy_state = np.random.get_state()
    digest.update(numpy_state[1].tobytes())
    digest.update(repr((numpy_state[0], *numpy_state[2:])).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def unchanged_rng():
    before = rng_digest()
    yield
    assert rng_digest() == before, "Audit instrumentation changed native RNG state"


def model_hashes(model):
    import torch

    groups = {"encoder": {}, "classifier": {}, "decoder": {}}
    for name, parameter in model.named_parameters():
        if "scaling_impl" in name:
            continue
        group = name.split(".")[0] if name.startswith(("classifier.", "decoder.")) else "encoder"
        raw = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        groups[group][name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                              "sha256": hashlib.sha256(raw).hexdigest()}
    return {name: {"sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
                   "parameters": values} for name, values in groups.items()}


def observer_state(model):
    from brevitas.core.scaling.standalone import ParameterFromRuntimeStatsScaling

    # Calling proxy.scale() here would collect statistics; inspect stored values only.
    return {name: {"counter": scaling.counter, "collect_stats_steps": scaling.collect_stats_steps,
                   "buffer": float(scaling.buffer.detach().cpu()),
                   "value": float(scaling.value.detach().cpu())}
            for name, scaling in model.named_modules()
            if isinstance(scaling, ParameterFromRuntimeStatsScaling)}


def model_contract(model, pretrain, enabled):
    from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
    from brevitas.proxy.quant_proxy import QuantProxyFromInjector
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase

    ranks = [list(parameter.shape) for name, parameter in model.named_parameters()
             if name.endswith(".dt_proj.weight")]
    assert ranks == [[1540, 96]] * 4, ranks
    proxies = [proxy for proxy in model.modules() if isinstance(proxy, QuantProxyFromInjector)
               and (getattr(proxy, "tensor_quant", None) is not None or
                    getattr(proxy, "fused_activation_quant_proxy", None) is not None)]
    acts = [proxy for proxy in proxies if isinstance(proxy, ActQuantProxyFromInjectorBase)]
    weights = [proxy for proxy in proxies if isinstance(proxy, WeightQuantProxyFromInjectorBase)]
    assert (len(acts), len(weights)) == ((34, 17) if pretrain else (35, 18))
    assert all(proxy.is_quant_enabled == enabled for proxy in proxies)
    fixed = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    assert fixed == [f"mamba_blocks.{block}.mamba_{branch}.conv1d.weight_scale"
                     for block in range(2) for branch in ("fwd", "rev")]
    count = sum(parameter.numel() for parameter in model.parameters())
    if not pretrain:
        assert count == 8670626
    return {"parameters": count, "dt_proj_shapes": ranks, "quantization_enabled": enabled,
            "activation_quantizers": len(acts), "weight_quantizers": len(weights),
            "native_fixed_parameters_at_construction": fixed}


def data_manifest(args):
    manifest = json.loads(args.data_manifest.read_text())
    result = {}
    for split, (windows, size, digest) in SPLITS.items():
        entry = manifest["data"][split]
        path = args.data_root / f"{split}.h5"
        assert entry["sha256"] == digest and entry["bytes"] == size
        assert path.is_file() and path.stat().st_size == size
        result[split] = {"path": str(path), "windows": windows, "bytes": size, "sha256": digest,
                         "mtime_ns": path.stat().st_mtime_ns}
    return {"data": result, "manifest_sha256": sha256(args.data_manifest),
            "verification": "retained SHA256 manifest plus current file sizes; no repeated full-data hashing"}


def compose_config(args):
    import hydra
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", os.getenv)
    if not OmegaConf.has_resolver("get_method"):
        OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

    pretrain = args.stage == "pretrain"
    enabled = args.group != "fp32"
    overrides = [f"+experiment={'FEMBA_qap_tuar' if pretrain else 'FEMBA_quantized'}",
                 f"tag=FEMBA_{args.group}_{args.stage}_seed{args.seed}", f"seed={args.seed}",
                 "resume=false", f"+model.quantization_enabled={str(enabled).lower()}",
                 f"io.base_output_path={json.dumps(str(args.output / 'tb_logs'))}",
                 f"io.checkpoint_dirpath={json.dumps(str(args.output / 'checkpoints'))}",
                 "final_test=false"]
    if pretrain:
        overrides.extend(["+data_module.datasets.tuar_train.use_cache=false",
                          f"data_module.datasets.tuar_train.hdf5_file={json.dumps(str(args.data_root / 'train.h5'))}"])
    else:
        overrides.extend(["data_module=finetune_data_module_tuar", "gpus=1", "trainer.strategy=auto",
                          "batch_size=32", "num_workers=4", "scheduler.warmup_epochs=1",
                          f"pretrained_checkpoint_path={json.dumps(str(args.pretrained))}"])
        for split in SPLITS:
            overrides.extend([f"+data_module.{split}.use_cache=false",
                              f"data_module.{split}.hdf5_file={json.dumps(str(args.data_root / (split + '.h5')))}"])
    with hydra.initialize_config_dir(config_dir=str(args.source / "config"), version_base="1.1"):
        cfg = hydra.compose(config_name="defaults", overrides=overrides)
    return cfg, overrides


def validate_config(cfg, args):
    pretrain = args.stage == "pretrain"
    assert cfg.seed == args.seed and cfg.resume is False and cfg.training is True
    assert cfg.batch_size == 32 and cfg.num_workers == 4 and cfg.gpus == 1
    assert cfg.trainer.accelerator == "gpu" and cfg.trainer.strategy == "auto"
    assert cfg.trainer.get("precision", "32-true") in (32, "32-true")
    assert cfg.trainer.get("accumulate_grad_batches", 1) == 1
    assert cfg.trainer.max_epochs == (10 if pretrain else 5)
    assert cfg.final_validate is True and cfg.final_test is False
    assert cfg.model.quantization_enabled == (args.group != "fp32")
    assert cfg.model.seq_length == 1280 and cfg.model.num_channels == 22
    assert cfg.model.embed_dim == 35 and cfg.model.num_blocks == 2
    assert cfg.optimizer.optim == "AdamW" and cfg.optimizer.lr == (1e-4 if pretrain else 5e-5)
    assert cfg.model_checkpoint.monitor == "val_loss" and cfg.model_checkpoint.mode == "min"
    norm = cfg.input_normalization
    assert norm.normalize and (norm.quartile_normalization_lower_val, norm.quartile_normalization_upper_val) == (-20, 20)
    if pretrain:
        assert cfg.data_module.train_val_split_ratio == 0.9
        assert cfg.masking.masking_ratio == 0.6 and list(cfg.masking.patch_size) == [2, 16]
        assert cfg.scheduler._target_ == "schedulers.multi_step_lr.multi_step_lr"
        assert cfg.scheduler.milestones == "200000+400000" and cfg.scheduler.gamma == 0.1
    else:
        assert cfg.optimizer.weight_decay == 0.5 and cfg.layerwise_lr_decay == 0.75
        assert cfg.finetuning.freeze_layers is True
        assert cfg.scheduler.warmup_epochs == 1 and cfg.scheduler.min_lr == 1e-6
        assert cfg.model.num_classes == 2 and cfg.model.classification_type == "bc"


def validate_pretrained_metadata(hyperparameters: Mapping, args) -> None:
    """Reject a different seed, classifier, or pretraining quantization group."""
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Pretrained checkpoint must contain hyper_parameters")
    if hyperparameters.get("seed") != args.seed:
        raise ValueError("Pretrained checkpoint seed must match the fine-tuning seed")
    model = hyperparameters.get("model")
    if not isinstance(model, Mapping) or model.get("_target_") != "models.FEMBA_int8.FEMBATinyInt8Pretrain":
        raise ValueError("Fine-tuning requires a pretraining checkpoint, not a classifier")
    # Original QAP configs omit this field; the native constructor defaults to True.
    enabled = model.get("quantization_enabled", True)
    if not isinstance(enabled, bool) or enabled != (args.group == "qap_b"):
        raise ValueError("Pretrained checkpoint quantization group differs from the requested handoff")


class StageAudit:
    def __init__(self, args):
        self.args = args
        self.task = None
        self.initial = None
        self.fit_summary = None
        self.epochs = []
        self.load_count = 0
        self.fit_started = False

    def record_data(self, module):
        if self.args.stage != "pretrain":
            assert [len(getattr(module, split)) for split in SPLITS] == [55524, 6426, 7211]
            return
        import numpy as np

        result = {}
        for split in ("train", "val"):
            subsets = getattr(module, split).datasets
            assert len(subsets) == 1 and len(subsets[0].dataset) == 55524
            indices = np.asarray(subsets[0].indices, dtype="<i8")
            path = self.args.output / f"pretrain_{split}_indices.npy"
            with path.open("xb") as stream:
                np.save(stream, indices, allow_pickle=False)
            result[split] = {"windows": len(indices), "ordered_int64_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                             "file": path.name, "file_sha256": sha256(path)}
        assert result["train"]["windows"] == 49971 and result["val"]["windows"] == 5553
        write_json(self.args.output / "pretrain_split.json", result)

    def attach_task(self, task):
        self.task = task
        self.initial = model_hashes(task.model)
        contract = model_contract(task.model, self.args.stage == "pretrain", self.args.group != "fp32")
        write_json(self.args.output / "initialization.json", {"seed": self.args.seed, "group": self.args.group,
                   "stage": self.args.stage, "hashes": self.initial, "contract": contract,
                   "observers": observer_state(task.model)})
        original_load = task.load_state_dict

        def audited_load(_task, state, *args, **kwargs):
            result = original_load(state, *args, **kwargs)
            if not self.fit_started:
                with unchanged_rng():
                    self.record_handoff(state, result)
            return result

        task.load_state_dict = MethodType(audited_load, task)

    def record_handoff(self, state, result):
        assert self.args.stage == "finetune" and self.load_count == 0
        self.load_count += 1
        metadata = getattr(state, "_metadata", {}).get("model", {})
        fresh = metadata.get("femba_unobserved_scales", [])
        observers = observer_state(self.task.model)
        current = model_hashes(self.task.model)
        assert current["classifier"] == self.initial["classifier"]
        assert set(result.missing_keys) == {"model.classifier.weight", "model.classifier.bias",
            "model.pre_classifier_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"}
        assert set(result.unexpected_keys) == {"model.decoder.dec_conv.weight", "model.decoder.conv_transpose.weight",
                                               "model.decoder.conv_transpose.bias"}
        fresh_count = sum(value["counter"] == 0 for value in observers.values())
        if self.args.group in ("fp32", "c"):
            assert len(fresh) == 34 and fresh_count == 35
        else:
            assert not fresh and fresh_count == 1
            assert all(value["counter"] > value["collect_stats_steps"] for name, value in observers.items()
                       if not name.startswith("pre_classifier_quant."))
        write_json(self.args.output / "pretrained_handoff.json", {"source_fresh_observers": fresh,
                   "target_fresh_observers": fresh_count, "observers": observers, "hashes_after_load": current,
                   "classifier_initialization_preserved": True, "missing_keys": result.missing_keys,
                   "unexpected_keys": result.unexpected_keys, "pretrained_sha256": self.args.pretrained_sha256})

    def callback(self):
        import pytorch_lightning as pl
        import torch

        audit = self

        class AuditCallback(pl.Callback):
            def on_fit_start(self, trainer, task):
                audit.fit_started = True
                assert audit.load_count == (0 if audit.args.stage == "pretrain" else 1)
                if audit.args.stage == "finetune":
                    assert all(parameter.requires_grad for parameter in task.model.parameters())

            def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
                if trainer.current_epoch != 0 or batch_idx != 0:
                    return
                with unchanged_rng():
                    observers = observer_state(task.model)
                    if audit.args.group == "c":
                        assert all(value["counter"] > 0 for value in observers.values())
                        first = observers["input_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl"]
                        assert first["counter"] <= first["collect_stats_steps"] and first["buffer"] != 1
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                    assert torch.isfinite(loss).all()
                    write_json(audit.args.output / "first_training_batch.json", {
                        "loss": float(loss.detach().cpu()), "observers": observers, "rng_unchanged": True})

            def on_train_epoch_end(self, trainer, task):
                with unchanged_rng():
                    row = {"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                           "metrics": {name: float(value.detach().cpu()) for name, value in trainer.callback_metrics.items()
                                       if isinstance(value, torch.Tensor) and value.numel() == 1}}
                    audit.epochs.append(row)
                    with (audit.args.output / "epochs.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")

            def on_fit_end(self, trainer, task):
                with unchanged_rng():
                    epochs = 10 if audit.args.stage == "pretrain" else 5
                    expected_steps = epochs * ((49971 if audit.args.stage == "pretrain" else 55524) // 32)
                    assert [row["epoch"] for row in audit.epochs] == list(range(epochs))
                    assert trainer.global_step == expected_steps
                    checkpoint = trainer.checkpoint_callback
                    best = Path(checkpoint.best_model_path)
                    assert best.is_file() and torch.isfinite(checkpoint.best_model_score)
                    audit.fit_summary = {"epochs": epochs, "global_step": trainer.global_step,
                        "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
                        "best_validation_loss": float(checkpoint.best_model_score.detach().cpu()),
                        "tb_log_dir": trainer.loggers[0].log_dir, "completed_at": timestamp()}
                    write_json(audit.args.output / "fit_summary.json", audit.fit_summary)

        return AuditCallback()


def run_native(native, cfg, audit):
    import hydra
    import pytorch_lightning as pl

    instantiate = hydra.utils.instantiate
    trainer_class = native.Trainer

    def audited_instantiate(*args, **kwargs):
        result = instantiate(*args, **kwargs)
        if isinstance(result, pl.LightningDataModule):
            with unchanged_rng():
                audit.record_data(result)
        elif isinstance(result, pl.LightningModule):
            with unchanged_rng():
                audit.attach_task(result)
        return result

    class AuditedTrainer(trainer_class):
        def __init__(self, *args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), audit.callback()]
            super().__init__(*args, **kwargs)

    hydra.utils.instantiate = audited_instantiate
    native.Trainer = AuditedTrainer
    try:
        native.train(cfg)
    finally:
        hydra.utils.instantiate = instantiate
        native.Trainer = trainer_class
    assert audit.fit_summary is not None


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Pinned teammate checkout with source.patch applied")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing the verified train.h5, val.h5 and test.h5")
    parser.add_argument("--data-manifest", type=Path, required=True,
                        help='Previously verified SHA256 manifest: {"data": {split: {"bytes": ..., "sha256": ...}}}')
    parser.add_argument("--output", type=Path, required=True,
                        help="New artifact directory outside the source checkout; never resumed")
    parser.add_argument("--group", choices=("fp32", "qap_b", "c"), required=True,
                        help="fp32: quantization off; qap_b: QAP then quantized fine-tuning; c: FP32 then quantized fine-tuning")
    parser.add_argument("--stage", choices=("pretrain", "finetune"), required=True,
                        help="Native pretraining (10 epochs) or fine-tuning (5 epochs), always batch 32")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--pretrained", type=Path,
                        help="Selected pretraining checkpoint for fine-tuning, including group C")
    parser.add_argument("--pretrained-sha256", help="SHA256 of --pretrained")
    parser.add_argument("--plan-only", action="store_true",
                        help="Resolve and validate the fixed Hydra recipe, without training or creating outputs")
    args = parser.parse_args(argv)
    for name in ("source", "data_root", "data_manifest", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output == args.source or args.source in args.output.parents:
        raise ValueError("Store stage artifacts outside the source checkout")
    if not (args.source / "config/defaults.yaml").is_file() or not (args.source / "run_train.py").is_file():
        raise ValueError("Source must contain the pinned teammate config/defaults.yaml and run_train.py")
    if args.output.exists() and not args.plan_only:
        raise ValueError("Output already exists; choose a new directory instead of resuming a stage")
    if args.group == "c" and args.stage != "finetune":
        raise ValueError("This protocol defines C as FP32-pretrained quantized fine-tuning")
    if args.stage == "finetune":
        if args.pretrained is None or not args.pretrained_sha256:
            raise ValueError("Fine-tuning requires an explicit pretrained checkpoint and SHA256")
        args.pretrained = args.pretrained.expanduser().resolve()
        digest = args.pretrained_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Pretrained SHA256 must contain 64 lowercase hexadecimal characters")
    elif args.pretrained is not None or args.pretrained_sha256 is not None:
        raise ValueError("Pretraining starts from the native random initialization")
    return args


def main():
    args = arguments()
    os.environ["DATA_PATH"] = str(args.data_root.parent)
    os.environ["CHECKPOINT_DIR"] = str(args.output)
    sys.path.insert(0, str(args.source))
    import torch
    from omegaconf import OmegaConf

    if not args.plan_only:
        import run_train as native

    cfg, overrides = compose_config(args)
    OmegaConf.resolve(cfg)
    validate_config(cfg, args)
    if args.plan_only:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    assert torch.cuda.is_available() and importlib.metadata.version("brevitas") == "0.12.1"
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == BASE_COMMIT
    if args.pretrained is not None:
        assert sha256(args.pretrained) == args.pretrained_sha256
        from evaluate import safe_load_checkpoint

        validate_pretrained_metadata(safe_load_checkpoint(args.pretrained).get("hyper_parameters"), args)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = data_manifest(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    write_json(args.output / "run_manifest.json", {"group": args.group, "stage": args.stage,
        "seed": args.seed, "started_at": timestamp(), "hydra_overrides": overrides, **inputs,
        "source_base_commit": commit, "source_dirty": bool(subprocess.check_output(
            ["git", "-C", str(args.source), "status", "--porcelain"], text=True).strip()),
        "executed_files_sha256": {name: sha256(args.source / name) for name in SOURCE_FILES},
        "wrapper_sha256": sha256(Path(__file__)), "config": OmegaConf.to_container(cfg, resolve=True),
        "runtime": {"gpu": torch.cuda.get_device_name(0), "matmul_precision": torch.get_float32_matmul_precision(),
                    "tf32_matmul": True, "tf32_cudnn": True, "cudnn_benchmark": True,
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "pytorch-lightning", "torchmetrics", "brevitas", "numpy", "h5py")}}})
    (args.output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    audit = StageAudit(args)
    try:
        run_native(native, cfg, audit)
    except Exception as error:
        write_json(args.output / "failure.json", {"at": timestamp(), "type": type(error).__name__, "message": str(error)})
        raise
    write_json(args.output / "complete.json", {"status": "complete", **audit.fit_summary})
    print("NATIVE_TRAIN_STAGE_COMPLETE", json.dumps(audit.fit_summary), flush=True)


if __name__ == "__main__":
    main()
