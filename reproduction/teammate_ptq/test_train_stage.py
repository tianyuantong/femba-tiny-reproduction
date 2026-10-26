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
"""CPU contracts for the portable native training and classifier replay runners.

Native model fixtures additionally use FEMBA_SOURCE (pinned teammate checkout)
and FEMBA_DATA_ROOT (verified HDF5 directory); no training or GPU work is run.
Both published source patches are loaded into an isolated temporary source tree.
"""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_one
import train_stage
from source_fixture import activated_precision_module


class ArgumentContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "config").mkdir(parents=True)
        (self.source / "config/defaults.yaml").touch()
        (self.source / "run_train.py").touch()
        self.common = ["--source", str(self.source), "--data-root", str(self.root / "data"),
                       "--output", str(self.root / "output"), "--seed", "43"]
        self.training = [*self.common, "--data-manifest", str(self.root / "manifest.json"),
                         "--group", "fp32", "--stage", "pretrain"]

    def test_pretraining_rejects_checkpoint_options(self):
        for option in ("--pretrained", "--pretrained-sha256"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "random initialization"):
                train_stage.arguments([*self.training, option, "unexpected"])

    def test_finetuning_requires_explicit_checkpoint_digest(self):
        args = [*self.training[:-2], "--stage", "finetune"]
        with self.assertRaisesRegex(ValueError, "explicit pretrained"):
            train_stage.arguments(args)
        with self.assertRaisesRegex(ValueError, "64 lowercase"):
            train_stage.arguments([*args, "--pretrained", "parent.ckpt", "--pretrained-sha256", "invalid"])
        parsed = train_stage.arguments([*args, "--pretrained", "parent.ckpt", "--pretrained-sha256", "a" * 64])
        self.assertTrue(parsed.pretrained.is_absolute())

    def test_group_c_accepts_each_seed_but_only_finetuning(self):
        args = [*self.training[:-4], "--group", "c", "--stage", "finetune",
                "--pretrained", "parent.ckpt", "--pretrained-sha256", "a" * 64]
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                args[args.index("--seed") + 1] = str(seed)
                parsed = train_stage.arguments(args)
                self.assertEqual((parsed.group, parsed.seed), ("c", seed))
        args[args.index("--stage") + 1] = "pretrain"
        with self.assertRaisesRegex(ValueError, "FP32-pretrained quantized fine-tuning"):
            train_stage.arguments(args)

    def test_source_checkout_and_previous_outputs_are_protected(self):
        args = list(self.training)
        args[args.index("--output") + 1] = str(self.source / "outputs")
        with self.assertRaisesRegex(ValueError, "outside the source"):
            train_stage.arguments(args)
        (self.root / "output").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            train_stage.arguments(self.training)
        self.assertTrue(train_stage.arguments([*self.training, "--plan-only"]).plan_only)

    def test_evaluation_accepts_each_c_seed_and_rejects_overwriting_selection(self):
        args = [*self.common, "--checkpoint", "classifier.ckpt", "--checkpoint-sha256", "a" * 64,
                "--model-kind", "c"]
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                args[args.index("--seed") + 1] = str(seed)
                parsed = evaluate_one.arguments(args)
                self.assertEqual((parsed.model_kind, parsed.seed), ("c", seed))
        (self.root / "output").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            evaluate_one.arguments(args)


class PretrainedMetadataContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(activated_precision_module())

    def metadata(self, seed=43, enabled=False, target="FEMBATinyInt8Pretrain"):
        return {"seed": seed, "model": {"_target_": "models.FEMBA_int8." + target,
                                      "quantization_enabled": enabled}}

    def test_matching_seed_and_pretraining_group_are_accepted(self):
        for group, seed, enabled in (("fp32", 43, False), ("qap_b", 44, True),
                                     ("c", 42, False), ("c", 43, False), ("c", 44, False)):
            with self.subTest(group=group, seed=seed):
                train_stage.validate_pretrained_metadata(self.metadata(seed, enabled),
                                                        SimpleNamespace(group=group, seed=seed))
        native_qap = self.metadata(enabled=True)
        del native_qap["model"]["quantization_enabled"]
        train_stage.validate_pretrained_metadata(native_qap, SimpleNamespace(group="qap_b", seed=43))

    def test_other_seed_and_classifier_checkpoints_are_rejected(self):
        for group in ("fp32", "c"):
            with self.subTest(group=group):
                args = SimpleNamespace(group=group, seed=43)
                with self.assertRaisesRegex(ValueError, "seed must match"):
                    train_stage.validate_pretrained_metadata(self.metadata(seed=42), args)
                with self.assertRaisesRegex(ValueError, "not a classifier"):
                    train_stage.validate_pretrained_metadata(self.metadata(target="FEMBATinyInt8Classifier"), args)

    def test_cross_group_pretraining_is_rejected(self):
        for group, seed, enabled in (("fp32", 43, True), ("qap_b", 43, False),
                                     ("c", 42, True), ("c", 43, True), ("c", 44, True)):
            with self.subTest(group=group, seed=seed), self.assertRaisesRegex(ValueError, "quantization group"):
                train_stage.validate_pretrained_metadata(self.metadata(seed, enabled),
                                                        SimpleNamespace(group=group, seed=seed))

    def test_quantized_parent_requires_same_precision_but_fp32_can_convert(self):
        args = SimpleNamespace(group="qap_b", seed=43, weight_bit_width=4, activation_bit_width=8)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            train_stage.validate_pretrained_metadata(self.metadata(enabled=True), args)
        metadata = self.metadata(enabled=True)
        metadata["model"].update(weight_bit_width=4, activation_bit_width=8)
        train_stage.validate_pretrained_metadata(metadata, args)
        args.group = "c"
        train_stage.validate_pretrained_metadata(self.metadata(enabled=False), args)

    def test_missing_metadata_and_non_boolean_quantization_are_rejected(self):
        args = SimpleNamespace(group="fp32", seed=43)
        with self.assertRaisesRegex(ValueError, "hyper_parameters"):
            train_stage.validate_pretrained_metadata(None, args)
        with self.assertRaisesRegex(ValueError, "not a classifier"):
            train_stage.validate_pretrained_metadata({"seed": 43}, args)
        with self.assertRaisesRegex(ValueError, "quantization group"):
            train_stage.validate_pretrained_metadata(self.metadata(enabled="false"), args)


class ReplaySelectionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(activated_precision_module())

    def test_threshold_is_locked_before_test_is_opened(self):
        labels = np.array([0, 0, 1, 1])
        validation_margin = np.array([-2., 0., 1., 2.])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "classifier.ckpt"
            checkpoint.write_bytes(b"fixture")
            data = root / "data"
            data.mkdir()
            (data / "val.h5").write_bytes(b"validation fixture")
            args = SimpleNamespace(source=root / "source", data=data, checkpoint=checkpoint,
                                   checkpoint_sha256=evaluate_one.capture.sha256(checkpoint),
                                   output=root / "evaluation", model_kind="fp32", seed=43)

            def replay(arguments, split, inputs):
                if split == "test":
                    selection = args.output / "selection.json"
                    started = json.loads((args.output / "test_started.json").read_text())
                    self.assertEqual(started["selection_sha256"], evaluate_one.capture.sha256(selection))
                    return {"array_sha256": "test"}, labels, -validation_margin
                # Creating this only after validation also proves the runner hashes no test input earlier.
                (data / "test.h5").write_bytes(b"test fixture")
                return {"array_sha256": "validation"}, labels, validation_margin

            with patch.object(evaluate_one.capture, "source_record", return_value={"fixture": True}), \
                 patch.object(evaluate_one.capture, "configure_runtime", return_value={}), \
                 patch.object(evaluate_one.capture, "safe_load_checkpoint",
                              return_value={"hyper_parameters": {"seed": 43}}), \
                 patch.object(evaluate_one, "replay_split", side_effect=replay):
                before = list(sys.path)
                try:
                    result = evaluate_one.run(args)
                finally:
                    sys.path[:] = before
            self.assertEqual(result["auroc"], 0.0)
            self.assertTrue(result["strict_reload_each_split"])
            selected = json.loads((args.output / "selection.json").read_text())
            self.assertEqual(selected["threshold"], 0.0)


@unittest.skipUnless(os.environ.get("FEMBA_SOURCE") and os.environ.get("FEMBA_DATA_ROOT"),
                     "Set FEMBA_SOURCE and FEMBA_DATA_ROOT for native CPU model fixtures")
class TrainingAuditFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hydra
        import pytorch_lightning as pl
        from omegaconf import OmegaConf

        cls.hydra, cls.pl, cls.omega = hydra, pl, OmegaConf
        cls.source = Path(os.environ["FEMBA_SOURCE"]).expanduser().resolve()
        cls.data_root = Path(os.environ["FEMBA_DATA_ROOT"]).expanduser().resolve()
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        from source_fixture import activated_source, reconstruct_source

        cls.source = reconstruct_source(cls.source, Path(cls.temporary.name) / "source")
        cls.enterClassContext(activated_source(cls.source))
        cls.helper = importlib.import_module("models.femba_ptq")

    def arguments(self, root, group, stage, seed=43):
        return SimpleNamespace(source=self.source, data_root=self.data_root, output=Path(root),
                               group=group, stage=stage, seed=seed,
                               pretrained=Path(root) / "pretrain.ckpt", pretrained_sha256="a" * 64,
                               max_epochs=10 if stage == "pretrain" else 5, save_every_epoch=False,
                               num_classes=2, classification_type="bc",
                               split_table={s: {"windows": w, "bytes": b, "sha256": d}
                                            for s, (w, b, d) in train_stage.SPLITS.items()})

    def config(self, args):
        cfg, _ = train_stage.compose_config(args)
        self.omega.resolve(cfg)
        train_stage.validate_config(cfg, args)
        return cfg

    def instantiate(self, cfg, args):
        self.pl.seed_everything(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            data = self.hydra.utils.instantiate(cfg.data_module)
            datasets = ([data.train.datasets[0].dataset] if args.stage == "pretrain"
                        else [getattr(data, split) for split in train_stage.SPLITS])
            for dataset in datasets:
                self.addCleanup(dataset.data.close)
            before = train_stage.rng_digest()
            audit = train_stage.StageAudit(args)
            with train_stage.unchanged_rng():
                audit.record_data(data)
            self.assertEqual(before, train_stage.rng_digest())
            task = self.hydra.utils.instantiate(cfg.task, cfg)
            before = train_stage.rng_digest()
            with train_stage.unchanged_rng():
                audit.attach_task(task)
            self.assertEqual(before, train_stage.rng_digest())
        return data, task, audit

    def test_paired_pretrain_initialization_and_splits(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            left_args = self.arguments(first, "fp32", "pretrain")
            right_args = self.arguments(second, "qap_b", "pretrain")
            _, _, left = self.instantiate(self.config(left_args), left_args)
            _, _, right = self.instantiate(self.config(right_args), right_args)
            self.assertEqual(left.initial, right.initial)
            left_split = json.loads((Path(first) / "pretrain_split.json").read_text())
            right_split = json.loads((Path(second) / "pretrain_split.json").read_text())
            self.assertEqual(left_split, right_split)

    def test_all_native_finetune_handoffs(self):
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                self._check_native_finetune_handoffs(seed)

    def _check_native_finetune_handoffs(self, seed):
        with TemporaryDirectory() as pretrain_root:
            parent_args = self.arguments(pretrain_root, "fp32", "pretrain", seed=seed)
            _, parent, _ = self.instantiate(self.config(parent_args), parent_args)
            for group in ("fp32", "qap_b", "c"):
                if group == "qap_b":
                    self.helper.set_quantization_enabled(parent.model, True)
                    for module in parent.model.modules():
                        if hasattr(module, "collect_stats_steps") and hasattr(module, "init_scale"):
                            module.buffer.fill_(3.0)
                            module.init_scale()
                elif group == "c":
                    self.helper.set_quantization_enabled(parent.model, False)
                    for module in parent.model.modules():
                        if hasattr(module, "collect_stats_steps") and hasattr(module, "init_scale"):
                            module.counter = 0
                            module.buffer.fill_(1.0)
                            module.value.data.fill_(1.0)
                with self.subTest(group=group), TemporaryDirectory() as root:
                    args = self.arguments(root, group, "finetune", seed=seed)
                    _, task, audit = self.instantiate(self.config(args), args)
                    before = train_stage.rng_digest()
                    task.load_state_dict(parent.state_dict(), strict=False)
                    self.assertEqual(before, train_stage.rng_digest())
                    report = json.loads((Path(root) / "pretrained_handoff.json").read_text())
                    self.assertEqual(report["target_fresh_observers"], 1 if group == "qap_b" else 35)
                    self.assertTrue(report["classifier_initialization_preserved"])
                    self.assertEqual(audit.load_count, 1)


if __name__ == "__main__":
    unittest.main()
