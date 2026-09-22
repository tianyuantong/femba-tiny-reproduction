"""Pinned upstream training with explicit single-GPU and file-IO adapters.

No modified local model/task/data-loader/scheduler implementation is used.
Single-GPU accumulation uses complete global batches of 1024, matching the
official update budget, but not exact DDP sample ordering or arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
REF = "d88596590f3bd3fce573be07646b7d3977ce7bcc"
ROOT = Path(os.environ.get("FEMBA_RUN_ROOT", "outputs/tuar-2025-seed42"))
DATA = ROOT / "data/TUAR_data"
WEIGHT = Path(os.environ.get("FEMBA_WEIGHT", "checkpoints/FEMBA/TUAR/FEMBA_tiny.safetensors"))
GLOBAL_BATCH_SIZE = 1024
source_hashes = {}


def source(path):
    data = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=REPO)
    source_hashes[path] = hashlib.sha256(data).hexdigest()
    return data.decode()


def upstream_module(name, path):
    parent, _, leaf = name.rpartition(".")
    package = importlib.import_module(parent)
    module = types.ModuleType(name)
    module.__file__ = f"{REF}/{path}"
    module.__package__ = parent
    sys.modules[name] = module
    setattr(package, leaf, module)
    exec(compile(source(path), module.__file__, "exec"), module.__dict__)
    return module


def _training_budget(num_samples: int, batch_size: int, epochs: int,
                     warmup_epochs: int) -> dict:
    values = (num_samples, batch_size, epochs, warmup_epochs)
    if any(type(value) is not int for value in values):
        raise TypeError("Training budget values must be integers")
    if batch_size <= 0 or GLOBAL_BATCH_SIZE % batch_size:
        raise ValueError("Batch size must be a positive divisor of 1024")
    if epochs <= 0 or not 0 <= warmup_epochs <= epochs:
        raise ValueError("Require positive epochs and 0 <= warmup_epochs <= epochs")
    steps = num_samples // GLOBAL_BATCH_SIZE
    if steps < 1:
        raise ValueError("Training split must contain a complete global batch")
    accumulation = GLOBAL_BATCH_SIZE // batch_size
    return {
        "train_samples": num_samples,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "micro_batch_size": batch_size,
        "accumulation_steps": accumulation,
        "microbatches_per_epoch": steps * accumulation,
        "optimizer_steps_per_epoch": steps,
        "total_optimizer_steps": steps * epochs,
        "warmup_optimizer_steps": steps * warmup_epochs,
        "samples_per_epoch": steps * GLOBAL_BATCH_SIZE,
        "dropped_samples_per_epoch": num_samples - steps * GLOBAL_BATCH_SIZE,
    }


def _positive_learning_rate(value: str | float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return rate


def _configure_learning_rate(cfg, requested: float | None) -> dict:
    upstream_rate = float(cfg.optimizer.lr)
    if requested is not None:
        cfg.optimizer.lr = _positive_learning_rate(requested)
    return {"requested": requested, "upstream": upstream_rate,
            "resolved": float(cfg.optimizer.lr)}


def _prepare_output(output_root: Path, stage: str, batch_size: int) -> None:
    final_path = output_root / "training/checkpoints/upstream2025-final.ckpt"
    protected = [final_path]
    if stage == "train":
        protected.extend([
            output_root / "results/upstream2025-run.json",
            output_root / "results/upstream2025-training-results.json",
            output_root / "logs/upstream2025-epochs.jsonl",
        ])
        protected.extend((output_root / "training/checkpoints").glob("*.ckpt"))
    else:
        protected.append(output_root / f"results/upstream2025-probe-b{batch_size}.json")
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"Existing run must be preserved; choose --output-root: {existing}")
    for directory in ("results", "logs", "training/checkpoints"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)


def _record_source(output_root: Path, script_source: bytes) -> dict:
    digest = hashlib.sha256(script_source).hexdigest()
    relative_path = Path("results") / f"upstream2025-train-source-{digest[:12]}.py"
    snapshot = output_root / relative_path
    if snapshot.exists():
        if snapshot.read_bytes() != script_source:
            raise ValueError("Existing source snapshot does not match executed script")
    else:
        with snapshot.open("xb") as handle:
            handle.write(script_source)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    return {"sha256": digest, "snapshot": str(relative_path),
            "working_tree_base_commit": head, "working_tree_dirty": bool(status.strip())}


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2)


def _evaluate_checkpoints(trainer, model, datamodule, best_checkpoint: str,
                          final_checkpoint: Path) -> dict:
    if not best_checkpoint:
        raise ValueError("Validation did not select a best checkpoint")
    best_validation = trainer.validate(model, datamodule=datamodule,
                                       ckpt_path=best_checkpoint, weights_only=False)
    best_test = trainer.test(model, datamodule=datamodule, ckpt_path=None)
    # Keep the historical result key for consumers, but identify it as diagnostic.
    final_test = trainer.test(model, datamodule=datamodule,
                             ckpt_path=str(final_checkpoint), weights_only=False)
    return {
        "primary_checkpoint": "best_checkpoint",
        "primary_test": "best_test",
        "best_validation": best_validation,
        "best_test": best_test,
        "upstream_final_test": final_test,
        "upstream_final_test_role": "final_checkpoint_diagnostic",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["probe", "train"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-root", type=Path, default=ROOT,
                        help="New output directory; data and pretrained weights remain unchanged")
    parser.add_argument("--learning-rate", type=_positive_learning_rate,
                        help="Override only optimizer.lr; default uses the pinned upstream recipe")
    args = parser.parse_args()
    if args.batch_size <= 0 or GLOBAL_BATCH_SIZE % args.batch_size:
        raise ValueError("Batch size must be a positive divisor of 1024")
    output_root = args.output_root.expanduser().resolve()
    _prepare_output(output_root, args.stage, args.batch_size)
    script_source = Path(__file__).read_bytes()
    data_manifest_path = ROOT / "results/upstream2025-data.json"
    data_manifest_bytes = data_manifest_path.read_bytes()
    manifest = json.loads(data_manifest_bytes)
    assert manifest.get("status") == "complete", "Data preparation is incomplete"
    assert manifest["upstream_commit"] == REF

    import hydra
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    import torch
    from safetensors.torch import load_file

    OmegaConf.register_new_resolver("env", lambda key: os.environ[key], replace=True)
    os.environ["DATA_PATH"] = str(ROOT / "data")
    os.environ["CHECKPOINT_DIR"] = str(output_root / "training")
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    upstream_module("models.FEMBA", "models/FEMBA.py")
    upstream_module("util.train_utils", "util/train_utils.py")
    upstream_module("datasets.hdf5_dataset", "datasets/hdf5_dataset.py")
    upstream_module("data_module.finetune_data_module", "data_module/finetune_data_module.py")
    upstream_module("schedulers.cosine", "schedulers/cosine.py")
    task_module = upstream_module("tasks.finetune_task", "tasks/finetune_task.py")
    parts = []
    for path in ["config/defaults.yaml", "config/data_module/finetune_data_module.yaml",
                 "config/task/finetune_task.yaml", "config/scheduler/cosine.yaml",
                 "config/model/FEMBA_finetune.yaml", "config/criterion/finetune_criterion.yaml",
                 "config/experiment/FEMBA_finetune.yaml"]:
        cfg_part = OmegaConf.create(source(path))
        cfg_part.pop("defaults", None)
        parts.append(cfg_part)
    cfg = OmegaConf.merge(*parts)
    learning_rate = _configure_learning_rate(cfg, args.learning_rate)
    # Dataset/model identity and hardware/IO overrides only.
    cfg.model.embed_dim = 35
    cfg.model.num_blocks = 2
    cfg.model.num_classes = 2
    cfg.model.classification_type = "bc"
    cfg.gpus = 1
    cfg.batch_size = args.batch_size
    cfg.num_workers = 2
    cfg.resume = False
    cfg.trainer.strategy = "auto"
    cfg.trainer.accumulate_grad_batches = GLOBAL_BATCH_SIZE // args.batch_size
    cfg.trainer.precision = "32-true"
    for split in ("train", "val", "test"):
        cfg.data_module[split].hdf5_file = str(DATA / f"upstream2025_{split}.h5")
    cfg.io.base_output_path = str(output_root / "training/logs")
    cfg.io.checkpoint_dirpath = str(output_root / "training/checkpoints")
    dm = hydra.utils.instantiate(cfg.data_module)
    dm.setup("fit")
    budget = _training_budget(len(dm.train_dataset), args.batch_size,
                              cfg.trainer.max_epochs, cfg.scheduler.warmup_epochs)
    cfg.trainer.limit_train_batches = budget["microbatches_per_epoch"]
    model = task_module.FinetuneTask(cfg)
    raw = load_file(str(WEIGHT))
    expected = {k: v for k, v in model.model.state_dict().items() if not k.startswith("classifier.")}
    assert set(expected).issubset(raw)
    assert all(raw[k].shape == v.shape for k, v in expected.items())
    # Format adapter: the released file has bare model keys while the original
    # task loader expects a Lightning 'model.' prefix. Do not change tensors.
    task_module.load_file = lambda path: {"model." + k: v for k, v in load_file(path).items()}
    model.load_safetensors_checkpoint(str(WEIGHT))
    assert all(torch.equal(model.model.state_dict()[k], raw[k]) for k in expected)
    assert all(p.requires_grad for p in model.model.parameters())
    evidence = {
        "upstream_commit": REF, "source_hashes": source_hashes,
        "executed_adapter": _record_source(output_root, script_source),
        "training_budget": budget,
        "learning_rate": learning_rate,
        "data_root": str(ROOT), "output_root": str(output_root),
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": hashlib.sha256(data_manifest_bytes).hexdigest(),
        "precision": {"trainer": cfg.trainer.precision,
                      "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "released_sha256": hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
        "encoder_tensors_loaded": len(expected),
        "encoder_values_loaded": sum(v.numel() for v in expected.values()),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "adaptations": ["Tiny/TUAR identity and existing file paths",
                        "single GPU; only complete global batches of 1024; DDP ordering not reproduced",
                        "workers=2; FP32 with matmul/convolution TF32 disabled; no tensorboard directories",
                        "bare checkpoint keys prefixed model.; tensor values unchanged",
                        "validation-best is primary; final retained as diagnostic; isolated output root"],
        "torch": torch.__version__, "pytorch_lightning": pl.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    if args.stage == "probe":
        model.cuda().train()
        x, y = next(iter(dm.train_dataloader()))
        x = model.normalize_fct(x.cuda())
        mask = model.generate_fake_mask(len(x), x.shape[1], x.shape[2])
        pred = model._step(x, mask)
        loss = model.criterion(pred["logits"], y.cuda())
        assert torch.isfinite(loss) and torch.isfinite(pred["logits"]).all()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        evidence["probe"] = {"batch_size": len(x), "loss": loss.item(),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        _write_json(output_root / f"results/upstream2025-probe-b{args.batch_size}.json", evidence)
        print("UPSTREAM_PROBE_OK", json.dumps(evidence["probe"]), flush=True)
        return

    final_path = output_root / "training/checkpoints/upstream2025-final.ckpt"
    _write_json(output_root / "results/upstream2025-run.json", evidence)
    metric_log = output_root / "logs/upstream2025-epochs.jsonl"

    class EpochEvidence(pl.Callback):
        def on_train_start(self, trainer, module):
            assert trainer.num_training_batches == budget["microbatches_per_epoch"]
            assert trainer.estimated_stepping_batches == budget["total_optimizer_steps"]
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert scheduler.num_opt_steps_per_epoch == budget["optimizer_steps_per_epoch"]
            assert scheduler.total_steps == budget["total_optimizer_steps"]
            assert scheduler.warmup_t == budget["warmup_optimizer_steps"]

        def on_train_epoch_end(self, trainer, module):
            assert trainer.global_step == (trainer.current_epoch + 1) * budget["optimizer_steps_per_epoch"]

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is not None and not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")

        def on_validation_epoch_end(self, trainer, module):
            row = {"phase": trainer.state.fn.value,
                   "epoch": trainer.current_epoch, "step": trainer.global_step}
            row["learning_rates"] = sorted({float(group["lr"]) for optimizer in trainer.optimizers
                                             for group in optimizer.param_groups})
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    assert torch.isfinite(value), key
                    row[key] = value.item()
            with metric_log.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print("UPSTREAM_EPOCH", json.dumps(row), flush=True)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(final_path.parent), filename="upstream2025-best-{epoch}-{step}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=False)
    trainer_args = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = pl.Trainer(**trainer_args, logger=False, callbacks=[checkpoint, EpochEvidence()],
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, dm)
    # Match the upstream validate(best) -> test(current model) evaluation order.
    trainer.save_checkpoint(str(final_path))
    result = {"final_checkpoint": str(final_path), "best_checkpoint": checkpoint.best_model_path,
              "best_val_loss": float(checkpoint.best_model_score),
              "epochs_completed": trainer.current_epoch, "global_step": trainer.global_step}
    result.update(_evaluate_checkpoints(trainer, model, dm, checkpoint.best_model_path, final_path))
    _write_json(output_root / "results/upstream2025-training-results.json", result)
    print("UPSTREAM_TRAINING_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
