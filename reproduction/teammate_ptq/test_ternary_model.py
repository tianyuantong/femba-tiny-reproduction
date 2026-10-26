"""Ternary weight scheme on the native FEMBA graph (ternary-model.patch)."""
import os
import sys
import unittest


@unittest.skipUnless(os.environ.get("FEMBA_SOURCE"), "Set FEMBA_SOURCE to a checkout with all three patches")
class TernaryModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["FEMBA_SOURCE"])
        import torch
        from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
        from models.FEMBA_int8 import FEMBATinyInt8Classifier, FEMBATinyInt8Pretrain
        from models.femba_quantization import quantization_config

        cls.torch, cls.proxy_type = torch, WeightQuantProxyFromInjectorBase
        cls.model_type, cls.quantization_config = FEMBATinyInt8Classifier, staticmethod(quantization_config)
        cls.pretrain_type = FEMBATinyInt8Pretrain

    def test_config_requires_two_bit_ternary_weights(self):
        config = self.quantization_config(2, 8, "ternary", "uniform")
        self.assertEqual(config["weight_quant_scheme"], "ternary")
        with self.assertRaises(ValueError):
            self.quantization_config(4, 8, "ternary", "uniform")
        with self.assertRaises(ValueError):
            self.quantization_config(2, 8, "uniform", "ternary")

    def test_all_weight_quantizers_are_three_level_with_mean_abs_scale(self):
        self.torch.manual_seed(0)
        model = self.model_type(weight_bit_width=2, weight_quant_scheme="ternary")
        proxies = {name: module for name, module in model.named_modules()
                   if isinstance(module, self.proxy_type) and module.is_quant_enabled}
        self.assertEqual(len(proxies), 18)
        for name, proxy in proxies.items():
            layer = model.get_submodule(name.rsplit(".", 1)[0])
            quantized = layer.quant_weight()
            self.assertEqual(int(proxy.bit_width()), 2, name)
            self.assertEqual(len(self.torch.unique(quantized.value)), 3, name)
            self.torch.testing.assert_close(quantized.scale.reshape(()), layer.weight.detach().abs().mean())
        x = self.torch.randn(2, 22, 1280) * 50
        logits, _ = model(x, None)
        logits.sum().backward()
        self.assertTrue(self.torch.isfinite(logits).all())
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_training_start_check_reads_ternary_widths(self):
        import train_stage

        for model, pretrain in ((self.pretrain_type(weight_bit_width=2, weight_quant_scheme="ternary"), True),
                                (self.model_type(weight_bit_width=2, weight_quant_scheme="ternary"), False)):
            widths = train_stage.model_contract(model, pretrain=pretrain, enabled=True)["actual_proxy_bit_widths"]
            self.assertEqual((set(widths["weight"]), set(widths["activation"])), ({2}, {8}))


if __name__ == "__main__":
    unittest.main()
