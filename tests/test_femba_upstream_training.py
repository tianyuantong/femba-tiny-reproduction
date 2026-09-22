"""CPU contracts for the single-GPU adapter; no model or data dependencies."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.femba_upstream_2025_train import (
    _configure_learning_rate,
    _evaluate_checkpoints,
    _prepare_output,
    _record_source,
    _training_budget,
    main,
)


class UpstreamTrainingTests(unittest.TestCase):
    def test_default_learning_rate_preserves_upstream_recipe(self):
        cfg = SimpleNamespace(optimizer=SimpleNamespace(lr=5e-4))
        evidence = _configure_learning_rate(cfg, None)
        self.assertEqual(cfg.optimizer.lr, 5e-4)
        self.assertEqual(evidence, {"requested": None, "upstream": 5e-4, "resolved": 5e-4})

    def test_learning_rate_override_changes_only_optimizer_rate(self):
        cfg = SimpleNamespace(optimizer=SimpleNamespace(
            lr=5e-4, optim="AdamW", weight_decay=0.5, betas=[0.9, 0.999]),
            scheduler=SimpleNamespace(warmup_epochs=10, min_lr=2.5e-6,
                                      warmup_lr_init=2.5e-7))
        before = {key: value for key, value in vars(cfg.optimizer).items() if key != "lr"}
        scheduler_before = vars(cfg.scheduler).copy()
        evidence = _configure_learning_rate(cfg, 1e-4)
        self.assertEqual(cfg.optimizer.lr, 1e-4)
        self.assertEqual(evidence, {"requested": 1e-4, "upstream": 5e-4, "resolved": 1e-4})
        self.assertEqual({key: value for key, value in vars(cfg.optimizer).items() if key != "lr"},
                         before)
        self.assertEqual(vars(cfg.scheduler), scheduler_before)

    def test_invalid_learning_rate_cli_fails_before_creating_output(self):
        for value in ("0", "-0.1", "nan", "inf", "-inf", "invalid"):
            with self.subTest(value=value), patch("sys.argv", ["train.py", "train",
                    f"--learning-rate={value}"]), patch("sys.stderr", new_callable=io.StringIO), \
                    patch("scripts.femba_upstream_2025_train._prepare_output") as prepare:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()

    def test_actual_pinned_scheduler_warmup_duration_and_group_rates(self):
        missing = [name for name in ("torch", "timm", "omegaconf")
                   if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest(f"Actual scheduler dependencies unavailable: {', '.join(missing)}")
        import torch
        from omegaconf import OmegaConf
        from scripts.femba_upstream_2025_train import upstream_module

        module = upstream_module("schedulers.cosine", "schedulers/cosine.py")
        for requested, expected in ((None, (0.000250125, 0.00014075)),
                                    (1e-4, (0.000050125, 0.00002825))):
            with self.subTest(requested=requested):
                cfg = OmegaConf.create({"optimizer": {"lr": 5e-4}})
                evidence = _configure_learning_rate(cfg, requested)
                reloaded = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                rate = reloaded.optimizer.lr
                self.assertEqual(rate, evidence["resolved"])
                optimizer = torch.optim.SGD([
                    {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate},
                    {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": rate * 0.75 ** 2},
                ], lr=rate)
                scheduler = module.CosineLRSchedulerWrapper(
                    optimizer, total_training_opt_steps=1620,
                    trainer=SimpleNamespace(max_epochs=30), warmup_epochs=10,
                    min_lr=2.5e-6, warmup_lr_init=2.5e-7,
                )
                self.assertEqual(scheduler.num_opt_steps_per_epoch, 54)
                self.assertEqual(scheduler.warmup_t, 540)
                self.assertEqual(scheduler.t_initial, 1620)
                self.assertEqual(scheduler.total_steps, 1620)
                scheduler.step_update(270)
                for group, expected_rate in zip(optimizer.param_groups, expected):
                    self.assertAlmostEqual(group["lr"], expected_rate, places=12)

    def test_recorded_corpus_uses_only_complete_global_batches(self):
        budget = _training_budget(56172, 256, 30, 10)
        self.assertEqual(budget["microbatches_per_epoch"], 216)
        self.assertEqual(budget["optimizer_steps_per_epoch"], 54)
        self.assertEqual(budget["total_optimizer_steps"], 1620)
        self.assertEqual(budget["warmup_optimizer_steps"], 540)
        self.assertEqual(budget["samples_per_epoch"], 55296)
        self.assertEqual(budget["dropped_samples_per_epoch"], 876)

    def test_microbatch_choice_preserves_optimizer_budget(self):
        for size, batches in ((64, 864), (128, 432), (256, 216), (512, 108), (1024, 54)):
            with self.subTest(batch_size=size):
                budget = _training_budget(56172, size, 30, 10)
                self.assertEqual(budget["microbatches_per_epoch"], batches)
                self.assertEqual(budget["total_optimizer_steps"], 1620)
                self.assertEqual(batches % budget["accumulation_steps"], 0)

    def test_invalid_budget_fails_before_training(self):
        invalid = ((1023, 256, 30, 10), (56172, 0, 30, 10),
                   (56172, 3, 30, 10), (56172, 256, 0, 0), (56172, 256, 30, 31))
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                _training_budget(*arguments)
        with self.assertRaises(TypeError):
            _training_budget(56172, 256.0, 30, 10)

    def test_existing_final_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "training/checkpoints/upstream2025-final.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"preserved checkpoint")
            for stage in ("probe", "train"):
                with self.subTest(stage=stage), self.assertRaises(FileExistsError):
                    _prepare_output(root, stage, 256)
            self.assertEqual(checkpoint.read_bytes(), b"preserved checkpoint")

    def test_interrupted_training_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _prepare_output(root, "train", 256)
            evidence = root / "results/upstream2025-run.json"
            evidence.write_text('{"status": "started"}')
            with self.assertRaises(FileExistsError):
                _prepare_output(root, "train", 256)
            self.assertEqual(evidence.read_text(), '{"status": "started"}')
            self.assertFalse((root / "data").exists())

    def test_snapshot_hash_identifies_retained_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _prepare_output(root, "probe", 256)
            source = b"print('source actually executed')\n"
            with patch("scripts.femba_upstream_2025_train.subprocess.check_output",
                       side_effect=["abc123\n", " M scripts/femba_upstream_2025_train.py\n"]):
                evidence = _record_source(root, source)
            snapshot = root / evidence["snapshot"]
            self.assertEqual(snapshot.read_bytes(), source)
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), evidence["sha256"])
            self.assertTrue(evidence["working_tree_dirty"])
            self.assertEqual(evidence["working_tree_base_commit"], "abc123")

    def test_primary_test_uses_validation_selected_weights(self):
        model = SimpleNamespace(weight=9)
        events = []
        checkpoints = {"best.ckpt": 2, "final.ckpt": 9}

        def validate(current_model, datamodule, ckpt_path, weights_only=None):
            self.assertIs(weights_only, False)
            current_model.weight = checkpoints[ckpt_path]
            events.append(("validate", ckpt_path, current_model.weight))
            return [{"weight": current_model.weight}]

        def test(current_model, datamodule, ckpt_path, weights_only=None):
            if ckpt_path is not None:
                self.assertIs(weights_only, False)
                current_model.weight = checkpoints[ckpt_path]
            events.append(("test", ckpt_path, current_model.weight))
            return [{"weight": current_model.weight}]

        trainer = SimpleNamespace(validate=validate, test=test)
        result = _evaluate_checkpoints(trainer, model, object(), "best.ckpt", Path("final.ckpt"))
        self.assertEqual(events, [("validate", "best.ckpt", 2), ("test", None, 2),
                                  ("test", "final.ckpt", 9)])
        self.assertEqual(result["primary_test"], "best_test")
        self.assertEqual(result["best_test"], [{"weight": 2}])
        self.assertEqual(result["upstream_final_test"], [{"weight": 9}])
        self.assertEqual(result["upstream_final_test_role"], "final_checkpoint_diagnostic")

    def test_missing_best_checkpoint_cannot_silently_test_final(self):
        trainer = Mock()
        with self.assertRaises(ValueError):
            _evaluate_checkpoints(trainer, object(), object(), "", Path("final.ckpt"))
        trainer.validate.assert_not_called()
        trainer.test.assert_not_called()


if __name__ == "__main__":
    unittest.main()
