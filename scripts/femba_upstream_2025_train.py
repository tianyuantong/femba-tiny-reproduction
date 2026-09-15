"""Pinned upstream training with explicit single-GPU and file-IO adapters.

No modified local model/task/data-loader/scheduler implementation is used.
The official 4 x 256 batch is represented by one GPU, batch 256, accumulation 4.
This preserves nominal global batch, not exact DDP ordering or arithmetic.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
REF = "d88596590f3bd3fce573be07646b7d3977ce7bcc"
ROOT = Path(os.environ.get("FEMBA_RUN_ROOT", "/root/work/femba-artifacts/tuar-2025-seed42"))
DATA = ROOT / "data/TUAR_data"
WEIGHT = Path(os.environ.get("FEMBA_WEIGHT", "/root/checkpoints/FEMBA/TUAR/FEMBA_tiny.safetensors"))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["probe", "train"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    assert args.batch_size > 0 and 1024 % args.batch_size == 0
    manifest = json.loads((ROOT / "results/upstream2025-data.json").read_text())
    assert manifest.get("status") == "complete", "Data preparation is incomplete"
    assert manifest["upstream_commit"] == REF

    import hydra
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    import torch
    from safetensors.torch import load_file

    OmegaConf.register_new_resolver("env", lambda key: os.environ[key], replace=True)
    os.environ["DATA_PATH"] = str(ROOT / "data")
    os.environ["CHECKPOINT_DIR"] = str(ROOT / "training")
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(2)
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
    cfg.trainer.accumulate_grad_batches = 1024 // args.batch_size
    cfg.trainer.precision = "32-true"
    for split in ("train", "val", "test"):
        cfg.data_module[split].hdf5_file = str(DATA / f"upstream2025_{split}.h5")
    cfg.io.base_output_path = str(ROOT / "training/logs")
    cfg.io.checkpoint_dirpath = str(ROOT / "training/checkpoints")
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
    dm = hydra.utils.instantiate(cfg.data_module)
    dm.setup("fit")
    evidence = {
        "upstream_commit": REF, "source_hashes": source_hashes,
        "released_sha256": hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
        "encoder_tensors_loaded": len(expected),
        "encoder_values_loaded": sum(v.numel() for v in expected.values()),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "adaptations": ["Tiny/TUAR identity and existing file paths",
                        "single GPU; accumulation preserves nominal upstream global batch 1024",
                        "workers=2; FP32 explicit; no tensorboard directories",
                        "bare checkpoint keys prefixed model.; tensor values unchanged",
                        "save final and validation-best models separately in existing directory"],
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
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
        (ROOT / f"results/upstream2025-probe-b{args.batch_size}.json").write_text(json.dumps(evidence, indent=2))
        print("UPSTREAM_PROBE_OK", json.dumps(evidence["probe"]), flush=True)
        return

    final_path = ROOT / "training/checkpoints/upstream2025-final.ckpt"
    assert not final_path.exists(), "Existing run must be inspected before restarting"
    (ROOT / "results/upstream2025-run.json").write_text(json.dumps(evidence, indent=2))
    metric_log = ROOT / "logs/upstream2025-epochs.jsonl"
    assert not metric_log.exists()

    class EpochEvidence(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is not None and not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")

        def on_validation_epoch_end(self, trainer, module):
            row = {"epoch": trainer.current_epoch, "step": trainer.global_step}
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
    # Original no-resume run tests the final in-memory model. Preserve that
    # behavior and also report selected-best explicitly as a separate row.
    trainer.save_checkpoint(str(final_path))
    result = {"final_checkpoint": str(final_path), "best_checkpoint": checkpoint.best_model_path,
              "best_val_loss": float(checkpoint.best_model_score),
              "epochs_completed": trainer.current_epoch, "global_step": trainer.global_step}
    result["upstream_final_test"] = trainer.test(model, datamodule=dm, ckpt_path=None)
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    result["best_validation"] = trainer.validate(model, datamodule=dm, ckpt_path=None)
    result["best_test"] = trainer.test(model, datamodule=dm, ckpt_path=None)
    (ROOT / "results/upstream2025-training-results.json").write_text(json.dumps(result, indent=2))
    print("UPSTREAM_TRAINING_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
