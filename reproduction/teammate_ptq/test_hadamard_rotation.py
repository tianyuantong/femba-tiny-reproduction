"""CPU checks of hadamard_rotation.py: exact structure, orthogonality, float equivalence, negative controls."""

import unittest

import torch
from torch import nn

import hadamard_rotation as rotation


class FakeBlock(nn.Module):
    """ssm.output_quant then out_proj, the two modules the hooks and the fold touch."""

    def __init__(self):
        super().__init__()
        self.ssm = nn.Module()
        self.ssm.output_quant = nn.Identity()
        self.out_proj = nn.Linear(rotation.DIMENSION, 385, bias=False)

    def forward(self, y):
        return self.out_proj(self.ssm.output_quant(y))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba_blocks = nn.ModuleList([nn.Module(), nn.Module()])
        self.mamba_blocks[1].mamba_fwd = FakeBlock()
        self.mamba_blocks[1].mamba_rev = FakeBlock()

    def forward(self, y):
        return self.mamba_blocks[1].mamba_fwd(y) + self.mamba_blocks[1].mamba_rev(y.flip(1))


class RotationTests(unittest.TestCase):
    def test_fixed_basis_and_metadata(self):
        plan = rotation.prepare_rotation(FakeModel(), "h128")
        report = rotation.verify_fixed_basis(plan)
        rotation.require_fixed_basis(report)
        self.assertTrue(report["allclose"])
        for name in rotation.TARGETS:
            layer = report["layers"][name]
            self.assertEqual(layer["blocks"], [128] * 12 + [4])
            self.assertEqual(layer["seed"], rotation.TARGET_SEEDS[name])
            self.assertEqual(layer["activation"]["elements"], rotation.DIMENSION ** 2)

    def test_dense_matrix_is_orthogonal(self):
        plan = rotation.prepare_rotation(FakeModel(), "h128")
        for name in rotation.TARGETS:
            matrix = plan.matrix(name)
            defect = (matrix @ matrix.T - torch.eye(rotation.DIMENSION)).abs().max().item()
            self.assertLess(defect, 1e-6)
            expected, _ = rotation.rotation_matrix(rotation.DIMENSION, rotation.TARGET_SEEDS[name])
            self.assertTrue(torch.equal(matrix, expected))

    def test_rotated_model_matches_float_model(self):
        torch.manual_seed(0)
        model = FakeModel()
        y = torch.randn(3, 7, rotation.DIMENSION)
        y[:, :, :20] *= 300.0  # a few outlier channels, as in the second block
        expected = model(y)
        plan = rotation.prepare_rotation(model, "h128")
        plan.fold_into_model_(model)
        handles = plan.install(model)
        actual = model(y)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
        self.assertEqual(len(handles), 2)
        with self.assertRaises(ValueError):
            plan.fold_into_model_(model)

    def test_rotation_flattens_outlier_channels(self):
        torch.manual_seed(0)
        plan = rotation.prepare_rotation(FakeModel(), "h128")
        y = torch.randn(4, 80, rotation.DIMENSION)
        y[:, :, :20] *= 300.0
        rotated = plan.rotate_activation(rotation.TARGETS[0], y)
        self.assertLess(rotated.abs().max().item(), y.abs().max().item() / 4)
        torch.testing.assert_close(rotated.double().norm(), y.double().norm(), rtol=1e-5, atol=0)

    def test_identity_plan_keeps_values(self):
        model = FakeModel()
        y = torch.randn(2, 5, rotation.DIMENSION)
        expected = model(y)
        plan = rotation.prepare_rotation(model, "identity")
        plan.fold_into_model_(model)
        plan.install(model)
        self.assertTrue(torch.equal(model(y), expected))

    def test_negative_controls_are_rejected(self):
        plan = rotation.prepare_rotation(FakeModel(), "h128")
        report = rotation.negative_controls(plan)
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["controls"]), {"h4_identity:" + name for name in rotation.TARGETS}
                         | {"activation_transpose", "swapped_pd_both"})
        for control in report["controls"].values():
            self.assertTrue(control["rejected"])
            self.assertGreater(control["failure_count"], 0)

    def test_parity_reports(self):
        reference = {"logits": torch.randn(4, 2), **{name: torch.randn(4, 80, 385) for name in rotation.TARGETS}}
        same = rotation.compare_parity(reference, reference)
        rotation.require_parity(same)
        self.assertTrue(same["allclose"] and same["legacy_allclose"])
        shifted = {name: value + 1e-3 for name, value in reference.items()}
        report = rotation.compare_parity(reference, shifted)
        self.assertFalse(report["allclose"])
        with self.assertRaises(RuntimeError):
            rotation.require_parity(report)

    def test_rejects_wrong_dtype_and_shape(self):
        plan = rotation.prepare_rotation(FakeModel(), "h128")
        with self.assertRaises(TypeError):
            plan.rotate_activation(rotation.TARGETS[0], torch.randn(1, 2, rotation.DIMENSION, dtype=torch.float64))
        with self.assertRaises(ValueError):
            plan.rotate_activation(rotation.TARGETS[0], torch.randn(1, 2, 1536))


if __name__ == "__main__":
    unittest.main()
