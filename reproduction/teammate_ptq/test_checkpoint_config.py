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
"""CPU checkpoint contracts using an isolated copy of the published source patches."""

from contextlib import redirect_stdout
import importlib
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import calibrate
import evaluate
from source_fixture import activated_source, reconstruct_source


@unittest.skipUnless(os.environ.get("FEMBA_SOURCE"), "Set FEMBA_SOURCE for native CPU checkpoint tests")
class CheckpointConfigurationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from omegaconf import OmegaConf

        cls.torch, cls.omega = torch, OmegaConf
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        source = reconstruct_source(Path(os.environ["FEMBA_SOURCE"]), cls.root / "source")
        cls.enterClassContext(activated_source(source))
        cls.helper = importlib.import_module("models.femba_ptq")
        cls.precision = importlib.import_module("models.femba_quantization")
        cls.task_class = importlib.import_module("tasks.finetune_task").FinetuneTask
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def setUp(self):
        self.directory = TemporaryDirectory(dir=self.root)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.torch.manual_seed(43)
        self.probe = self.torch.randn(2, 22, 64)

    def config(self, weight_width=8, activation_width=8):
        return self.omega.create({"seed": 43, "model": {
            "_target_": "models.FEMBA_int8.FEMBATinyInt8Classifier", "seq_length": 64,
            "num_classes": 2, "classification_type": "bc", "quantization_enabled": False,
            **self.precision.quantization_config(weight_width, activation_width)},
            "input_normalization": {"normalize": True,
                "quartile_normalization_lower_val": -20, "quartile_normalization_upper_val": 20}})

    def make_task(self, cfg):
        with redirect_stdout(io.StringIO()):
            return self.task_class(cfg).eval()

    def reload(self, path, kind, **kwargs):
        with redirect_stdout(io.StringIO()):
            return evaluate.load_task(path, kind, device="cpu", **kwargs)

    def write_quantized(self, path, weight_width=8, activation_width=8, legacy=False):
        cfg = self.config(weight_width, activation_width)
        task = self.make_task(cfg)
        self.helper.calibrate_quantized(task.model, [self.probe])
        cfg.model.quantization_enabled = True
        state = task.state_dict()
        if legacy:
            for field in self.precision.CONFIG_FIELDS:
                del cfg.model[field]
            for metadata in state._metadata.values():
                metadata.pop("femba_quantization", None)
        self.torch.save({"hyper_parameters": cfg, "state_dict": state}, path)
        return task

    def test_infers_config_and_checks_each_actual_proxy_role(self):
        path = self.output / "w8a4.ckpt"
        self.write_quantized(path, activation_width=4)
        restored, contract = self.reload(path, "qap_b")
        self.assertEqual(restored.hparams.model.activation_bit_width, 4)
        self.assertEqual(contract["configuration"], self.precision.quantization_config(8, 4))
        self.assertEqual(set(contract["actual_proxy_bit_widths"]["activation"].values()), {4})
        self.assertEqual(set(contract["actual_proxy_bit_widths"]["weight"].values()), {8})
        self.assertEqual(len(contract["fp32_depthwise_weights"]), 4)

    def test_requested_config_and_inconsistent_hparams_are_rejected(self):
        path = self.output / "w4a8.ckpt"
        self.write_quantized(path, weight_width=4)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            self.reload(path, "qap_b", expected_config={"weight_bit_width": 8})
        with self.assertRaisesRegex(ValueError, "Only FP32"):
            self.reload(path, "qap_b", target_config=self.precision.quantization_config())
        record = evaluate.safe_load_checkpoint(path)
        record["hyper_parameters"].model.weight_bit_width = 8
        self.torch.save(record, path)
        with self.assertRaisesRegex(ValueError, "quantization metadata differ"):
            self.reload(path, "qap_b")
        record["hyper_parameters"].model.weight_bit_width = 4
        record["hyper_parameters"].model.quantization_enabled = False
        self.torch.save(record, path)
        with self.assertRaisesRegex(ValueError, "quantization metadata differ"):
            self.reload(path, "fp32")

    def test_legacy_checkpoint_missing_config_remains_exact_w8a8(self):
        path = self.output / "legacy.ckpt"
        original = self.write_quantized(path, legacy=True)
        restored, contract = self.reload(path, "qap_b")
        self.assertEqual(contract["configuration"], self.precision.quantization_config())
        with self.torch.no_grad():
            expected = original.model(self.probe)[0]
            actual = restored.model(self.probe)[0]
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            self.reload(path, "qap_b", expected_config={"activation_bit_width": 4})

    def test_fp32_conversion_and_ptq_reload_preserve_weights_scales_and_output(self):
        cfg = self.config()
        parent = self.make_task(cfg)
        parent_path = self.output / "fp32.ckpt"
        self.torch.save({"hyper_parameters": cfg, "state_dict": parent.state_dict()}, parent_path)
        config = self.precision.quantization_config(4, 8)
        task, _ = self.reload(parent_path, "fp32", target_config=config)
        expected_float = calibrate.floating_parameters(parent.model)
        actual_float = calibrate.floating_parameters(task.model)
        self.assertEqual(expected_float.keys(), actual_float.keys())
        for name, tensor in expected_float.items():
            self.assertTrue(self.torch.equal(tensor, actual_float[name]), name)
        report = self.helper.calibrate_quantized(task.model, [self.probe])
        self.assertEqual(report["num_batches"], 1)
        calibrated_float = calibrate.floating_parameters(task.model)
        for name, tensor in expected_float.items():
            self.assertTrue(self.torch.equal(tensor, calibrated_float[name]), name)
        with redirect_stdout(io.StringIO()):
            result = calibrate.save_reload(task, self.output, evaluate.sha256(parent_path),
                                          {"quantization_config": config}, self.probe)
        self.assertEqual(result["checkpoint_file"], "ptq_w4a8.ckpt")
        self.assertTrue(result["all_53_scales_exact"])
        self.assertTrue(result["floating_parameters_reload_exact"])
        self.assertTrue(result["strict_reload_exact"])
        saved = evaluate.safe_load_checkpoint(self.output / result["checkpoint_file"])
        self.assertEqual(saved["ptq_recipe"]["quantization_config"], config)
        self.assertEqual(saved["parent_sha256"], evaluate.sha256(parent_path))
        self.assertEqual(calibrate.checkpoint_filename(self.precision.quantization_config()), "ptq_int8.ckpt")


if __name__ == "__main__":
    unittest.main()
