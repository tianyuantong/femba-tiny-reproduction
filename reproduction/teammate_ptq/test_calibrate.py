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
"""Contracts for the fixed native PTQ calibration recipe."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import calibrate


class CalibrationContracts(unittest.TestCase):
    def test_invalid_percentiles(self):
        for value in (True, "99.0", None):
            with self.assertRaises(TypeError):
                calibrate.validate_percentile(value)
        for value in (0, -1, 100.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calibrate.validate_percentile(value)
        self.assertEqual(calibrate.validate_percentile(99), 99.0)

    def test_exact_immutable_training_manifest(self):
        indices = calibrate.calibration_indices(55524)
        self.assertEqual(len(indices), 512)
        self.assertEqual(len(set(indices)), 512)
        self.assertEqual(indices, calibrate.calibration_indices(55524))
        manifest = {"split": "train.h5", "seed": 42, "indices": indices,
                    "train_sha256": "9fb802613f4256e6fb15ba1bd5c0853a22b983567ae48b0c8b88ac64c123d8b5"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration_indices.json"
            calibrate.write_json(path, manifest)
            self.assertEqual(calibrate.sha256(path),
                             "58886a53d3250e3d803f044aa3bd412bdb5a6a4b49aec38c196f071ff9c804bf")
            with self.assertRaises(FileExistsError):
                calibrate.write_json(path, manifest)
        with self.assertRaises(ValueError):
            calibrate.calibration_indices(511)

    def test_batch_iterator_keeps_all_512_indices_in_manifest_order(self):
        import torch
        from types import SimpleNamespace

        indices = calibrate.calibration_indices(55524)
        dataset = torch.utils.data.TensorDataset(torch.arange(55524, dtype=torch.float32).unsqueeze(1),
                                                torch.zeros(55524))
        parameter = torch.nn.Parameter(torch.zeros(1))
        task = SimpleNamespace(parameters=lambda: iter([parameter]), normalize_fct=lambda signals: signals)
        batches = list(calibrate.batches(task, dataset, indices))
        self.assertEqual(len(batches), 16)
        self.assertTrue(all(batch.shape == (32, 1) for batch in batches))
        self.assertEqual(torch.cat(batches).flatten().to(torch.int64).tolist(), indices)

    @unittest.skipUnless(os.environ.get("FEMBA_SOURCE"), "Set FEMBA_SOURCE for the native graph fixture")
    def test_native_observers_change_only_percentile(self):
        import torch

        sys.path.insert(0, os.environ["FEMBA_SOURCE"])
        from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
        from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
        from models.FEMBA_int8 import FEMBATinyInt8Classifier

        model = FEMBATinyInt8Classifier(quantization_enabled=False).eval()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        topology = {name: id(module) for name, module in model.named_modules()}
        calibrate.set_percentile(model, 99.0)
        activations = [module for module in model.modules()
                       if isinstance(module, ActQuantProxyFromInjectorBase)
                       and module.fused_activation_quant_proxy is not None]
        weights = [module for module in model.modules()
                   if isinstance(module, WeightQuantProxyFromInjectorBase) and module.tensor_quant is not None]
        self.assertEqual((len(activations), len(weights)), (35, 18))
        self.assertTrue(all(not proxy.is_quant_enabled for proxy in activations + weights))
        self.assertTrue(all(proxy.fused_activation_quant_proxy.tensor_quant.scaling_impl.stats.stats_impl.q == 99.0
                            for proxy in activations))
        self.assertEqual(topology, {name: id(module) for name, module in model.named_modules()})
        current = model.state_dict()
        self.assertEqual(before.keys(), current.keys())
        self.assertTrue(all(torch.equal(value, current[name]) for name, value in before.items()))
        with self.assertRaises(ValueError):
            calibrate.set_percentile(model, 99.9)


if __name__ == "__main__":
    unittest.main()
