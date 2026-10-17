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
"""Native CPU contracts for precision configuration; fixtures do not train models."""

import copy
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import calibrate
import evaluate
import train_stage
from source_fixture import activated_source, reconstruct_source


@unittest.skipUnless(os.environ.get("FEMBA_SOURCE"), "Set FEMBA_SOURCE to the pinned Git checkout")
class QuantizationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        repository = Path(os.environ["FEMBA_SOURCE"])
        cls.reference = reconstruct_source(repository, cls.root / "reference", configurable=False)
        cls.source = reconstruct_source(repository, cls.root / "configurable")
        cls.enterClassContext(activated_source(cls.source))
        cls.models = importlib.import_module("models.FEMBA_int8")
        cls.helper = importlib.import_module("models.femba_ptq")
        cls.precision = importlib.import_module("models.femba_quantization")
        cls.ternary = importlib.import_module("models.femba_ternary")

    def make_model(self, **kwargs):
        self.torch.manual_seed(43)
        return self.models.FEMBATinyInt8Classifier(**kwargs)

    def probe(self, length=1280, batch=1):
        generator = self.torch.Generator().manual_seed(91)
        return self.torch.rand((batch, 22, length), generator=generator)

    def output(self, model, probe):
        with self.torch.no_grad():
            return model(probe, self.torch.zeros_like(probe, dtype=self.torch.bool))[0]

    def assert_same_tensors(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            self.assertTrue(self.torch.equal(first[name], second[name]), name)

    def test_default_w8_matches_historical_source_and_explicit_w8(self):
        with activated_source(self.reference):
            reference_class = importlib.import_module("models.FEMBA_int8").FEMBATinyInt8Classifier
            self.torch.manual_seed(43)
            reference = reference_class()
            state = copy.deepcopy(reference.state_dict())
            rng = self.torch.get_rng_state().clone()
            expected = self.output(reference, self.probe())
        for config in ({}, {"weight_bit_width": 8, "activation_bit_width": 8}):
            with self.subTest(config=config):
                model = self.make_model(**config)
                self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
                self.assert_same_tensors(state, model.state_dict())
                self.assertTrue(self.torch.equal(expected, self.output(model, self.probe())))

    def test_pretrain_default_and_disabled_match_historical_state_output_rng(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                with activated_source(self.reference):
                    constructor = importlib.import_module("models.FEMBA_int8").FEMBATinyInt8Pretrain
                    self.torch.manual_seed(43)
                    reference = constructor(quantization_enabled=enabled)
                    state = copy.deepcopy(reference.state_dict())
                    rng = self.torch.get_rng_state().clone()
                    expected = self.output(reference, self.probe())
                self.torch.manual_seed(43)
                model = self.models.FEMBATinyInt8Pretrain(quantization_enabled=enabled,
                    weight_bit_width=8, activation_bit_width=8)
                self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
                self.assert_same_tensors(state, model.state_dict())
                self.assertTrue(self.torch.equal(expected, self.output(model, self.probe())))

    def test_mixed_widths_reach_every_proxy_and_keep_fp32_depthwise(self):
        from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
        from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase

        for weight, activation in ((4, 8), (8, 4), (4, 4)):
            with self.subTest(weight=weight, activation=activation):
                model = self.make_model(weight_bit_width=weight, activation_bit_width=activation)
                proxies = [p for p in model.modules() if getattr(p, "is_quant_enabled", False)]
                weights = [p for p in proxies if isinstance(p, WeightQuantProxyFromInjectorBase)]
                acts = [p for p in proxies if isinstance(p, ActQuantProxyFromInjectorBase)]
                self.assertEqual((len(weights), len(acts)), (18, 35))
                self.assertEqual({int(p.bit_width()) for p in weights}, {weight})
                self.assertEqual({int(p.bit_width()) for p in acts}, {activation})
                depthwise = [m for m in model.modules() if type(m) is self.torch.nn.Conv1d]
                self.assertEqual(len(depthwise), 4)
                self.assertTrue(all(m.groups == 1540 and m.weight.dtype == self.torch.float32
                                    for m in depthwise))
                self.assertEqual(train_stage.model_contract(model, False, True)["parameters"], 8670626)

    def test_disabled_quantization_keeps_parameters_rng_and_output(self):
        left = self.make_model(quantization_enabled=False)
        rng = self.torch.get_rng_state().clone()
        right = self.make_model(quantization_enabled=False, weight_bit_width=4, activation_bit_width=4)
        self.assertTrue(self.torch.equal(rng, self.torch.get_rng_state()))
        self.assert_same_tensors(left.state_dict(), right.state_dict())
        self.assertTrue(self.torch.equal(self.output(left, self.probe()), self.output(right, self.probe())))

    def test_legacy_strict_reload_defaults_to_w8_and_rejects_other_widths(self):
        original = self.make_model()
        self.output(original, self.probe())
        state = copy.deepcopy(original.state_dict())
        # The old serialization has no precision metadata, but retains tensor values.
        state._metadata[""].pop("femba_quantization", None)
        self.make_model().load_state_dict(state, strict=True)
        for enabled in (True, False):
            with self.subTest(enabled=enabled), self.assertRaises((ValueError, RuntimeError)):
                self.make_model(weight_bit_width=4, quantization_enabled=enabled).load_state_dict(state)
        del state["classifier.weight"]
        with self.assertRaises(RuntimeError):
            self.make_model().load_state_dict(state, strict=True)

    def test_quantized_state_rejects_configuration_mismatch_even_non_strict(self):
        original = self.make_model(weight_bit_width=4, activation_bit_width=8)
        self.output(original, self.probe())
        state = copy.deepcopy(original.state_dict())
        restored = self.make_model(weight_bit_width=4, activation_bit_width=8)
        restored.load_state_dict(state, strict=True)
        original.eval()
        restored.eval()
        self.assertTrue(self.torch.equal(self.output(original, self.probe()), self.output(restored, self.probe())))
        for weight, activation in ((8, 8), (4, 4)):
            with self.subTest(weight=weight, activation=activation), self.assertRaises((ValueError, RuntimeError)):
                self.make_model(weight_bit_width=weight, activation_bit_width=activation).load_state_dict(
                    state, strict=False)
        self.helper.set_quantization_enabled(original, False)
        with self.assertRaises((ValueError, RuntimeError)):
            self.make_model(weight_bit_width=8).load_state_dict(original.state_dict(), strict=False)

    def test_fp32_pretrain_handoff_preserves_34_fresh_then_updates_all_35(self):
        self.torch.manual_seed(43)
        parent = self.models.FEMBATinyInt8Pretrain(quantization_enabled=False)
        self.assertEqual(len(train_stage.observer_state(parent)), 34)
        model = self.make_model(weight_bit_width=4, activation_bit_width=4)
        classifier = model.classifier.weight.detach().clone()
        incompatible = model.load_state_dict(parent.state_dict(), strict=False)
        self.assertTrue(all(name.startswith(("decoder.",)) for name in incompatible.unexpected_keys))
        self.assertTrue(self.torch.equal(classifier, model.classifier.weight))
        states = train_stage.observer_state(model)
        self.assertEqual(len(states), 35)
        self.assertTrue(all(state["counter"] == 0 for state in states.values()))
        self.output(model.train(), self.probe())
        self.assertTrue(all(state["counter"] > 0 for state in train_stage.observer_state(model).values()))

    def test_ptq_512_synthetic_windows_preserves_weights_and_strict_reload(self):
        # Short sequences keep this CPU fixture small; real TUAR dimensions are checked above.
        model = self.make_model(seq_length=32, quantization_enabled=False,
                                weight_bit_width=4, activation_bit_width=8)
        before = calibrate.floating_parameters(model)
        calibrate.set_percentile(model, 99.0)
        probe = self.probe(length=32, batch=32)
        report = self.helper.calibrate_quantized(model, (probe for _ in range(16)))
        self.assertEqual(report["num_batches"], 16)
        self.assertEqual((report["activation_quantizers"], report["weight_quantizers"]), (35, 18))
        self.assert_same_tensors(before, calibrate.floating_parameters(model))
        state = copy.deepcopy(model.state_dict())
        restored = self.make_model(seq_length=32, weight_bit_width=4).eval()
        restored.load_state_dict(state, strict=True)
        self.assertTrue(self.torch.equal(self.output(model, probe), self.output(restored, probe)))
        self.assertEqual(evaluate.quantization_contract(model, True),
                         evaluate.quantization_contract(restored, True))

    def test_configuration_validation_and_legacy_defaults(self):
        default = self.precision.quantization_config()
        self.assertEqual(self.precision.config_from_mapping({}), default)
        for invalid in (True, 4.0, "4", None):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                self.precision.quantization_config(weight_bit_width=invalid)
        for invalid in (1, 17, -4):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.precision.quantization_config(activation_bit_width=invalid)
        with self.assertRaises(ValueError):
            self.precision.quantization_config(weight_bit_width=2, weight_quant_scheme="ternary")
        self.assertEqual(self.precision.quantization_config(weight_bit_width=2)["weight_quant_scheme"], "uniform")

    def test_ternary_levels_boundaries_ste_and_serialization(self):
        mapping = self.ternary.TernaryMapping(scale=2.0, threshold=0.5)
        values = self.torch.tensor([-3., -1.001, -1., 0., 1., 1.001, 3.], requires_grad=True)
        before = values.detach().clone()
        codes = self.ternary.ternary_codes(values, mapping)
        self.assertEqual(codes.tolist(), [-1, -1, 0, 0, 0, 1, 1])
        self.assertEqual(codes.dtype, self.torch.int8)
        result = self.ternary.ternary_fake_quantize(values, mapping)
        self.assertTrue(self.torch.equal(result.detach(), codes * 2.0))
        result.sum().backward()
        self.assertTrue(self.torch.equal(values.grad, self.torch.ones_like(values)))
        self.assertTrue(self.torch.equal(values.detach(), before))
        restored = self.ternary.TernaryMapping.from_dict(json.loads(json.dumps(mapping.to_dict())))
        self.assertEqual(restored, mapping)
        self.assertTrue(self.torch.equal(self.ternary.ternary_codes(values, restored), codes))
        for kwargs in ({"scale": 0., "threshold": .5}, {"scale": 1., "threshold": -1.},
                       {"scale": float("inf"), "threshold": .5}):
            with self.assertRaises((TypeError, ValueError)):
                self.ternary.TernaryMapping(**kwargs)
        invalid = mapping.to_dict()
        invalid["mapping"] = "uniform_2bit"
        with self.assertRaises((ValueError, TypeError)):
            self.ternary.TernaryMapping.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
