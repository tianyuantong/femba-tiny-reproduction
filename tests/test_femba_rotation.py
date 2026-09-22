"""CPU contracts for paired rotation and complete parity gates; no Mamba/GPU."""

from copy import deepcopy
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from biofoundation.femba_rotation import (
    ELEMENTWISE_PROTOCOL,
    EXPECTED_DIMENSIONS,
    TARGETS,
    VECTOR_RMS_PROTOCOL,
    audit_rotation_negative_controls,
    compare_parity,
    prepare_rotations,
    require_fixed_basis,
    require_parity,
    tensor_parity,
    vector_rms_parity,
    verify_fixed_basis,
)


def toy_model() -> nn.Module:
    model = nn.Module()
    model.mamba_blocks = nn.ModuleList([nn.Module(), nn.Module()])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        for block in model.mamba_blocks:
            for direction in ("mamba_fwd", "mamba_rev"):
                branch = nn.Module()
                branch.out_proj = nn.Linear(1540, 3)
                setattr(block, direction, branch)
        model.classifier = nn.Module()
        model.classifier.mamba_1 = nn.Module()
        model.classifier.mamba_1.out_proj = nn.Linear(512, 3)
    return model


def qdq(value: torch.Tensor, bits: int) -> torch.Tensor:
    maximum = 2 ** (bits - 1) - 1
    scale = value.abs().max().clamp_min(1e-8) / maximum
    return (value / scale).round().clamp(-maximum, maximum) * scale


class RotationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = toy_model()
        cls.rotation = prepare_rotations(cls.original)
        cls.identity = prepare_rotations(cls.original, mode="identity")

    def test_block_dimensions_and_layer_streams_are_reproducible(self):
        again = prepare_rotations(self.original)
        self.assertEqual(self.rotation.to_dict(), again.to_dict())
        records = self.rotation.to_dict()["layers"]
        self.assertEqual(records[TARGETS[0]]["blocks"], [128] * 12 + [4])
        self.assertEqual(records[TARGETS[-1]]["blocks"], [128] * 4)
        self.assertNotEqual(records[TARGETS[0]]["matrix_sha256"],
                            records[TARGETS[1]]["matrix_sha256"])

    def test_paired_transform_preserves_all_five_linear_outputs(self):
        model = deepcopy(self.original)
        before = dict(self.original.named_modules())
        after = dict(model.named_modules())
        self.rotation.transform_out_proj_weights_(model)
        generator = torch.Generator().manual_seed(23)
        for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
            value = torch.randn((2, dimension, 3), generator=generator)
            rotated = self.rotation.rotate_activation(name, value)
            self.assertEqual(rotated.dtype, torch.float32)
            expected = before[f"{name}.out_proj"](value.transpose(1, 2))
            actual = after[f"{name}.out_proj"](rotated.transpose(1, 2))
            self.assertTrue(tensor_parity(expected, actual)["allclose"], name)
            self.assertTrue(torch.allclose(value.square().sum(dim=1),
                                           rotated.square().sum(dim=1), rtol=1e-5))

    def test_compact_rotation_matches_explicit_p_d_h_order(self):
        name = TARGETS[-1]
        metadata = self.rotation.metadata[name]
        permutation = torch.tensor(metadata["permutation"])
        p = torch.eye(512, dtype=torch.float32)[permutation]
        d = torch.diag(torch.tensor(metadata["signs"], dtype=torch.float32))
        h = torch.ones((1, 1), dtype=torch.float32)
        for _ in range(7):
            h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
        h = torch.block_diag(*([h / 128 ** 0.5] * 4))
        value = torch.randn((2, 512), generator=torch.Generator().manual_seed(31))
        expected = ((value @ p) @ d) @ h
        actual = self.rotation.rotate_activation(name, value[:, :, None]).squeeze(-1)
        self.assertTrue(tensor_parity(expected, actual)["allclose"])

    def test_encoder_tail_uses_h4_after_the_permutation_and_signs(self):
        name, dimension = TARGETS[0], EXPECTED_DIMENSIONS[0]
        metadata = self.rotation.metadata[name]
        permutation = torch.tensor(metadata["permutation"])
        signs = torch.tensor(metadata["signs"], dtype=torch.float32)
        # Choose source coordinates which P moves into the four tail positions.
        source_indices = permutation.argsort()[-4:]
        value = torch.eye(dimension, dtype=torch.float32)[source_indices]
        h4 = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                           [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float32) / 2
        expected = torch.zeros((4, dimension), dtype=torch.float32)
        expected[:, -4:] = signs[-4:, None] * h4
        actual = self.rotation.rotate_activation(name, value[:, :, None]).squeeze(-1)
        self.assertTrue(torch.equal(actual, expected))

    def test_real_encoder_projection_width_preserves_the_linear_map(self):
        name, dimension = TARGETS[0], EXPECTED_DIMENSIONS[0]
        generator = torch.Generator().manual_seed(103)
        weight = torch.randn((385, dimension), generator=generator) / dimension ** 0.5
        bias = torch.randn((385,), generator=generator)
        value = torch.randn((2, dimension, 8), generator=generator)
        rotated_weight = self.rotation.transforms[name].apply(weight)
        rotated_value = self.rotation.rotate_activation(name, value)
        reference = F.linear(value.transpose(1, 2), weight, bias)
        actual = F.linear(rotated_value.transpose(1, 2), rotated_weight, bias)
        self.assertEqual(actual.shape, (2, 8, 385))
        self.assertTrue(tensor_parity(reference, actual)["allclose"])

    def test_identity_preserves_weight_and_activation_qdq_paths(self):
        model = deepcopy(self.original)
        self.identity.transform_out_proj_weights_(model)
        before, after = dict(self.original.named_modules()), dict(model.named_modules())
        generator = torch.Generator().manual_seed(29)
        for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
            source = before[f"{name}.out_proj"]
            target = after[f"{name}.out_proj"]
            self.assertTrue(torch.equal(source.weight, target.weight))
            value = torch.randn((1, dimension, 2), generator=generator)
            for bits in (4, 8):
                for activation_bits in (None, 8):
                    original_input = qdq(value, 8) if activation_bits else value
                    rotated_input = self.identity.rotate_activation(name + ".scan_gate_output", value)
                    if activation_bits:
                        rotated_input = qdq(rotated_input, 8)
                    expected = F.linear(original_input.transpose(1, 2), qdq(source.weight, bits), source.bias)
                    actual = F.linear(rotated_input.transpose(1, 2), qdq(target.weight, bits), target.bias)
                    self.assertTrue(torch.equal(expected, actual), (name, bits, activation_bits))

    def test_rotation_off_changes_neither_weights_nor_activations(self):
        model = deepcopy(self.original)
        plan = prepare_rotations(model, mode="none")
        plan.transform_out_proj_weights_(model)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))
        value = torch.ones((1, 1540, 2), dtype=torch.float32)
        self.assertIs(plan.rotate_activation(TARGETS[0], value), value)

    def test_identity_preserves_noncontiguous_activation_storage_and_strides(self):
        for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
            values = (torch.zeros((2, 3, dimension)).transpose(1, 2),
                      torch.zeros((2, dimension, 8))[:, :, 1::2])
            for value in values:
                with self.subTest(name=name, stride=value.stride()):
                    self.assertFalse(value.is_contiguous())
                    actual = self.identity.rotate_activation(name, value)
                    self.assertIs(actual, value)
                    self.assertEqual(actual.untyped_storage().data_ptr(),
                                     value.untyped_storage().data_ptr())
                    self.assertEqual(actual.storage_offset(), value.storage_offset())
                    self.assertEqual(actual.stride(), value.stride())

    def test_transforms_are_fp32_even_inside_autocast(self):
        model = deepcopy(self.original)
        value = torch.ones((1, 1540, 2), dtype=torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.rotation.transform_out_proj_weights_(model)
            actual = self.rotation.rotate_activation(TARGETS[0], value)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, self.rotation.rotate_activation(TARGETS[0], value)))

    def test_repeated_weight_transform_is_rejected(self):
        model = deepcopy(self.original)
        self.rotation.transform_out_proj_weights_(model)
        with self.assertRaisesRegex(ValueError, "fresh FP32"):
            self.rotation.transform_out_proj_weights_(model)

    def test_invalid_shape_or_precision_fails_before_weight_mutation(self):
        model = deepcopy(self.original)
        model.classifier.mamba_1.out_proj = nn.Linear(511, 3)
        original_weight = model.mamba_blocks[0].mamba_fwd.out_proj.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.rotation.transform_out_proj_weights_(model)
        self.assertTrue(torch.equal(original_weight, model.mamba_blocks[0].mamba_fwd.out_proj.weight))
        with self.assertRaises(TypeError):
            self.rotation.rotate_activation(TARGETS[0], torch.ones(1, 1540, 2).half())
        with self.assertRaises(ValueError):
            self.rotation.rotate_activation(TARGETS[0], torch.ones(1, 2, 1540))


class ParityGateContracts(unittest.TestCase):
    def setUp(self):
        self.reference = {name: torch.zeros((2, 3), dtype=torch.float32)
                          for name in ("logits", *TARGETS)}

    def test_layer_failure_blocks_even_when_final_logits_match(self):
        candidate = {name: tensor.clone() for name, tensor in self.reference.items()}
        candidate[TARGETS[1]][0, 1] = 2e-5
        report = compare_parity(self.reference, candidate)
        self.assertTrue(report["outputs"]["logits"]["allclose"])
        self.assertFalse(report["allclose"])
        detail = report["outputs"][TARGETS[1]]
        self.assertEqual(detail["failure_count"], 1)
        self.assertEqual(detail["worst_index"], [0, 1])
        self.assertAlmostEqual(detail["max_error_over_tolerance"], 2.0)
        with self.assertRaisesRegex(RuntimeError, TARGETS[1]):
            require_parity(report)

    def test_complete_true_report_passes_without_relaxed_tolerance(self):
        report = compare_parity(self.reference, self.reference)
        require_parity(report)
        report["atol"] = 1e-4
        with self.assertRaisesRegex(ValueError, "tolerance"):
            require_parity(report)

    def test_missing_layer_cannot_pass(self):
        incomplete = dict(self.reference)
        del incomplete[TARGETS[-1]]
        with self.assertRaisesRegex(ValueError, "five"):
            compare_parity(self.reference, incomplete)

    def test_nonfinite_failure_produces_strict_json_evidence(self):
        candidate = {name: tensor.clone() for name, tensor in self.reference.items()}
        candidate["logits"][0, 0] = float("nan")
        report = compare_parity(self.reference, candidate)
        self.assertEqual(report["outputs"]["logits"]["nonfinite_count"], 1)
        json.dumps(report, allow_nan=False)
        with self.assertRaises(RuntimeError):
            require_parity(report)


class VectorRmsContracts(unittest.TestCase):
    def setUp(self):
        self.reference = {name: torch.zeros((2, 3, 2), dtype=torch.float32) for name in TARGETS}
        self.reference["logits"] = torch.zeros((2, 2), dtype=torch.float32)

    def test_explicit_norm_protocol_retains_legacy_failure(self):
        self.reference[TARGETS[0]][0, 0, 0] = 100
        candidate = {name: value.clone() for name, value in self.reference.items()}
        candidate[TARGETS[0]][0, 0, 1] = 5e-5
        legacy = compare_parity(self.reference, candidate)
        self.assertEqual(legacy, compare_parity(self.reference, candidate, protocol=ELEMENTWISE_PROTOCOL))
        self.assertFalse(legacy["allclose"])
        report = compare_parity(self.reference, candidate, protocol=VECTOR_RMS_PROTOCOL)
        self.assertTrue(report["allclose"])
        self.assertFalse(report["legacy_allclose"])
        self.assertEqual(report["outputs"], legacy["outputs"])
        require_parity(report)

    def test_one_bad_vector_cannot_hide_in_a_whole_layer_average(self):
        reference = self.reference[TARGETS[0]]
        candidate = reference.clone()
        candidate[0, 0, 0] = 2.1e-5
        self.assertLess(float(candidate.square().mean().sqrt()), 1e-5)
        report = vector_rms_parity(reference, candidate)
        self.assertEqual(report["vectors"], 6)
        self.assertEqual(report["failure_count"], 1)
        self.assertFalse(report["allclose"])

    def test_logits_still_use_the_original_elementwise_gate(self):
        candidate = {name: value.clone() for name, value in self.reference.items()}
        candidate["logits"][0, 0] = 2e-5
        report = compare_parity(self.reference, candidate, protocol=VECTOR_RMS_PROTOCOL)
        self.assertTrue(all(row["allclose"] for row in report["vector_rms"].values()))
        with self.assertRaisesRegex(RuntimeError, "logits"):
            require_parity(report)

    def test_nonfinite_vector_is_rejected_without_nonfinite_json(self):
        reference = self.reference[TARGETS[0]]
        candidate = reference.clone()
        candidate[0, 1, 0] = float("inf")
        report = vector_rms_parity(reference, candidate)
        self.assertEqual(report["nonfinite_count"], 1)
        self.assertEqual(report["failure_count"], 1)
        json.dumps(report, allow_nan=False)

    def test_unknown_protocol_and_incomplete_norm_reports_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "protocol"):
            compare_parity(self.reference, self.reference, protocol="automatic")
        report = compare_parity(self.reference, self.reference, protocol=VECTOR_RMS_PROTOCOL)
        del report["vector_rms"][TARGETS[0]]
        with self.assertRaisesRegex(ValueError, "Incomplete vector"):
            require_parity(report)

    def test_rms_reduction_does_not_overflow_on_finite_fp32_values(self):
        reference = torch.full((1, 1, 2), 1e30, dtype=torch.float32)
        self.assertTrue(vector_rms_parity(reference, reference)["allclose"])
        report = vector_rms_parity(reference, reference * 2)
        self.assertEqual(report["nonfinite_count"], 0)
        self.assertEqual(report["failure_count"], 1)


class FixedBasisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = toy_model()
        cls.plan = prepare_rotations(cls.model)
        cls.positive = verify_fixed_basis(cls.plan)
        cls.negative = audit_rotation_negative_controls(cls.plan)

    def test_complete_basis_covers_real_weight_and_activation_paths(self):
        require_fixed_basis(self.positive)
        for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
            row = self.positive["layers"][name]
            self.assertTrue(row["metadata_matches"])
            for path in ("activation", "weight"):
                self.assertEqual(row[path]["elements"], dimension ** 2)
                self.assertEqual(row[path]["failure_count"], 0)

    def test_h4_only_controls_fail_exactly_the_changed_tail(self):
        self.assertTrue(self.negative["all_rejected"])
        self.assertEqual(len(self.negative["controls"]), 6)
        for target in TARGETS[:-1]:
            control = self.negative["controls"]["h4_identity:" + target]
            self.assertEqual(control["targets"], [target])
            self.assertTrue(control["rejected"])
            self.assertEqual(control["failure_count"], 32)
            with self.assertRaises(RuntimeError):
                require_fixed_basis(control["gate_report"])
            for name, row in control["gate_report"]["layers"].items():
                for path in ("activation", "weight"):
                    self.assertEqual(row[path]["failure_count"], 16 if name == target else 0)

    def test_transpose_and_both_sides_wrong_order_are_actually_rejected(self):
        transpose = self.negative["controls"]["activation_transpose"]
        swapped = self.negative["controls"]["swapped_pd_both"]
        for name in TARGETS:
            self.assertGreater(transpose["gate_report"]["layers"][name]["activation"]["failure_count"], 0)
            self.assertEqual(transpose["gate_report"]["layers"][name]["weight"]["failure_count"], 0)
            for path in ("activation", "weight"):
                self.assertGreater(swapped["gate_report"]["layers"][name][path]["failure_count"], 0)

    def test_basis_verifier_detects_a_bypassed_weight_mutation_loop(self):
        plan = prepare_rotations(self.model)
        plan.transform_out_proj_weights_ = lambda model: None
        report = verify_fixed_basis(plan)
        self.assertFalse(report["allclose"])
        for row in report["layers"].values():
            self.assertEqual(row["activation"]["failure_count"], 0)
            self.assertGreater(row["weight"]["failure_count"], 0)

    def test_controls_leave_the_real_model_and_plan_unchanged(self):
        again = prepare_rotations(self.model)
        self.assertEqual(self.plan.to_dict(), again.to_dict())
        for name in TARGETS[:-1]:
            self.assertTrue(torch.equal(self.plan.transforms[name].tail_block,
                                        again.transforms[name].tail_block))
        state = torch.get_rng_state().clone()
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        require_fixed_basis(verify_fixed_basis(self.plan))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))


if __name__ == "__main__":
    unittest.main()
