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
"""Apply the published patch and exercise native Brevitas observer reloads."""

import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BASE = "a2a4d441485efd8423c44b35d7448a30ccd44437"


@unittest.skipUnless(os.environ.get("FEMBA_SOURCE"), "Set FEMBA_SOURCE to the pinned teammate checkout")
class SourcePatchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from brevitas.nn import QuantIdentity
        from brevitas.quant import Int8ActPerTensorFloat

        cls.torch = torch
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        (root / "models").mkdir()
        original = subprocess.run(
            ["git", "-C", os.environ["FEMBA_SOURCE"], "show", f"{BASE}:models/FEMBA_int8.py"],
            check=True, capture_output=True).stdout
        (root / "models/FEMBA_int8.py").write_bytes(original)
        subprocess.run(["git", "apply", str(Path(__file__).with_name("source.patch"))],
                       cwd=root, check=True, capture_output=True)
        spec = importlib.util.spec_from_file_location("published_femba_ptq", root / "models/femba_ptq.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

        class Model(torch.nn.Module):
            def __init__(self, enabled):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
                self.quant = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
                helper.set_quantization_enabled(self, enabled)

        cls.make_model = staticmethod(Model)

    def scaling(self, model):
        return model.quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl

    def saved_state(self, model):
        stream = io.BytesIO()
        self.torch.save(model.state_dict(), stream)
        stream.seek(0)
        return self.torch.load(stream, weights_only=True)

    def test_fp32_to_quantized_load_collects_fresh_statistics(self):
        original = self.make_model(False)
        state = self.saved_state(original)
        restored = self.make_model(True)
        restored.load_state_dict(state, strict=True)
        scaling = self.scaling(restored)
        self.assertEqual(scaling.counter, 0)
        self.assertEqual(float(scaling.buffer), 1.0)
        restored.train()
        output = restored.quant(self.torch.tensor([-853.0, 853.0])).value
        self.assertEqual(scaling.counter, 1)
        self.assertEqual(float(scaling.buffer), 853.0)
        self.assertGreater(float(output.abs().max()), 800.0)

    def test_fp32_to_disabled_load_remains_fresh(self):
        restored = self.make_model(False)
        restored.load_state_dict(self.saved_state(self.make_model(False)), strict=True)
        self.assertEqual(self.scaling(restored).counter, 0)
        self.assertFalse(restored.quant.act_quant.is_quant_enabled)
        self.torch.testing.assert_close(restored.weight, self.torch.tensor(2.0), rtol=0, atol=0)

    def test_observed_quantized_load_preserves_scale_and_output(self):
        original = self.make_model(True).train()
        original.quant(self.torch.tensor([-853.0, 853.0]))
        self.scaling(original).init_scale()
        original.eval()
        restored = self.make_model(True).eval()
        restored.load_state_dict(self.saved_state(original), strict=True)
        self.assertGreater(self.scaling(restored).counter, self.scaling(restored).collect_stats_steps)
        self.torch.testing.assert_close(restored.quant.act_quant.scale(), original.quant.act_quant.scale(),
                                        rtol=0, atol=0)
        values = self.torch.tensor([-1000.0, -3.0, 42.0, 1000.0])
        self.torch.testing.assert_close(restored.quant(values).value, original.quant(values).value,
                                        rtol=0, atol=0)

    def test_strict_load_still_rejects_missing_model_weight(self):
        state = self.saved_state(self.make_model(False))
        del state["weight"]
        with self.assertRaisesRegex(RuntimeError, "weight"):
            self.make_model(True).load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
