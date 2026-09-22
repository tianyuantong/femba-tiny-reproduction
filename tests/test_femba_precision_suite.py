"""Pure protocol contracts plus actual CPU tensor contracts (run in the WSL environment)."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import femba_precision_suite as suite

try:
    import torch
except ImportError:
    torch = None


class PrecisionSuiteTests(unittest.TestCase):
    def test_calibration_is_balanced_unique_and_reproducible(self):
        labels = [0] * 1500 + [1] * 1700
        indices = suite.balanced_indices(labels)
        self.assertEqual(indices, suite.balanced_indices(labels))
        self.assertEqual(len(indices), 2048)
        self.assertEqual(len(set(indices)), 2048)
        self.assertEqual(sum(labels[index] for index in indices), 1024)
        self.assertNotEqual(indices, suite.balanced_indices(labels, seed=43))

    def test_invalid_calibration_cannot_silently_shrink_or_relabel(self):
        for labels in ([0] * 1023 + [1] * 1024, [0, 1, 2]):
            with self.assertRaises(ValueError):
                suite.balanced_indices(labels)

    def test_scope_has_unique_declared_native_boundaries(self):
        self.assertEqual(len(suite.ACTIVATION_SITES), 57)
        self.assertEqual(len(set(suite.ACTIVATION_SITES)), 57)
        self.assertEqual(sum(name.endswith('.scan_gate_output') for name in suite.ACTIVATION_SITES), 5)

    def test_rotations_have_matching_weight_and_activation_controls(self):
        modes = {variant.name: variant for variant in suite.VARIANTS}
        for bits in (4, 8):
            for activations in ('a32', 'a8-float'):
                plain, rotated = modes[f'w{bits}{activations}'], modes[f'rot-w{bits}{activations}']
                self.assertEqual(plain.weight_bits, rotated.weight_bits)
                self.assertEqual(plain.activation_scale, rotated.activation_scale)
                self.assertFalse(plain.rotation)
                self.assertTrue(rotated.rotation)
        self.assertTrue(all(variant.diagnostic for variant in suite.FP16_DIAGNOSTICS))

    def test_parity_requires_explicit_pass(self):
        self.assertTrue(suite._report_passed({'allclose': True}))
        self.assertFalse(suite._report_passed({'passed': False, 'max_abs_error': 0.0}))
        with self.assertRaises(ValueError):
            suite._report_passed({'max_abs_error': 0.0})

    def test_explicit_path_needs_fp32_and_own_width_gate(self):
        variant = suite.Variant('w8a8', 8, 'float')
        parity = {'weight_gate_passed': {'None': True, '8': True, '4': False}, 'rotation_passed': True}
        self.assertIsNone(suite.gate_failure_reason(variant, parity))
        parity['weight_gate_passed']['None'] = False
        self.assertEqual(suite.gate_failure_reason(variant, parity), 'explicit_qdq_off_gate_failed')
        self.assertIsNone(suite.gate_failure_reason(suite.Variant('w8a32', 8), parity))
        self.assertIsNone(suite.gate_failure_reason(suite.Variant('fp16', precision='fp16-direct'), parity))

    def test_calibration_observes_once_per_rotation_group_and_shares_statistics(self):
        statistics = {'scale_float': {'node': 0.03}, 'scale_pot': {'node': 0.03125},
                      'max_abs': {'node': 3.81}, 'elements': {'node': 2048}}
        context = SimpleNamespace()
        modes = [suite.Variant('w8a8', 8, 'float'), suite.Variant('w4a8', 4, 'float'),
                 suite.Variant('w8a8-pot', 8, 'pot'), suite.Variant('rot-w8a8', 8, 'float', True),
                 suite.Variant('rot-w4a8', 4, 'float', True)]
        with TemporaryDirectory() as temporary, patch.object(suite, '_collect_calibration', return_value=statistics) as collect:
            results = []
            for mode in modes:
                directory = Path(temporary) / mode.name
                directory.mkdir()
                results.append(suite._calibrate(context, mode, [0, 1], directory))
            self.assertEqual(collect.call_count, 2)
            self.assertEqual([call.args[1] for call in collect.call_args_list], [False, True])
            self.assertEqual(len({item['shared_statistics_sha256'] for item in results[:3]}), 1)
            self.assertEqual(results[0]['scales'], results[1]['scales'])
            self.assertEqual(results[2]['scales'], statistics['scale_pot'])
            results[0]['max_abs']['node'] = -1
            self.assertEqual(context.calibration_cache['plain_fp32']['max_abs']['node'], 3.81)

    def test_evidence_rejects_nonfinite_json_numbers(self):
        with self.assertRaises(ValueError):
            suite.canonical({'scale': float('nan')})

    def test_performance_threshold_accepts_boundary_and_preserves_negative(self):
        baseline = {key: 0.9 for key in ('accuracy', 'balanced_accuracy', 'auroc', 'average_precision')}
        reduced = {**baseline, 'auroc': 0.5}
        report = suite.performance_report(reduced, baseline)
        self.assertFalse(report['passed'])
        self.assertAlmostEqual(report['delta_vs_fp32']['auroc'], -0.4)
        self.assertTrue(suite.performance_report({key: 0.895 for key in baseline}, baseline)['passed'])

    def test_saved_scales_and_rotation_are_required_and_integrity_checked(self):
        metadata = {'variant': asdict(suite.Variant('fixture', 8, 'float', True)),
                    'scales': {'node': 0.01}, 'rotation': {'hash': 'original'},
                    'config': {'normalize': True}, 'checkpoint_sha256': 'parent'}
        restored = suite.validate_saved_metadata(deepcopy(metadata), metadata)
        self.assertEqual(restored, suite.Variant(**metadata['variant']))
        for field in metadata:
            damaged = deepcopy(metadata)
            del damaged[field]
            with self.assertRaises(ValueError):
                suite.validate_saved_metadata(damaged, metadata)
        for field, value in (('scales', {'node': 0.02}), ('rotation', {'hash': 'changed'})):
            damaged = deepcopy(metadata)
            damaged[field] = value
            with self.assertRaises(ValueError):
                suite.validate_saved_metadata(damaged, metadata)


@unittest.skipIf(torch is None, 'PyTorch CPU contracts must run in the WSL torch environment')
class TensorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite.torch = torch

    def test_weight_quantization_is_per_channel_and_symmetric(self):
        values = torch.tensor([[0.0, 0.5, 1.5, -7.0], [0.0, 0.05, 0.15, -0.7]])
        output, scales = suite.weight_qdq_tensor(values, 4)
        torch.testing.assert_close(scales, torch.tensor([[1.0], [0.1]]))
        torch.testing.assert_close(output[0], torch.tensor([0.0, 0.0, 2.0, -7.0]))
        for bits in (4, 8):
            output, scales = suite.weight_qdq_tensor(values, bits)
            limit = 2 ** (bits - 1) - 1
            self.assertTrue(bool(((output / scales).abs() <= limit + 1e-5).all()))
            torch.testing.assert_close(output[:, -1], values[:, -1])

    def test_observe_disables_weight_qdq_but_applies_fp32_rotation(self):
        task, plan = Mock(), Mock()
        with patch.object(suite, '_new_task', return_value=task), \
             patch.object(suite, '_weight_qdq') as weight, \
             patch.object(suite, '_manager', return_value=Mock()) as manager, \
             patch.object(suite, 'install_explicit_mamba_forward', return_value=[]), \
             patch.object(suite, 'install_outer_boundaries', return_value=[]), \
             patch.object(suite, '_cast_model', return_value={}) as cast, \
             patch('biofoundation.femba_rotation.prepare_rotations', return_value=plan):
            built = suite._build_variant(None, suite.Variant('rot-w8a8', 8, 'float', True), observe=True)
            weight.assert_not_called()
            plan.transform_out_proj_weights_.assert_called_once_with(task.model)
            self.assertEqual(manager.call_args.args[1], 'observe')
            cast.assert_called_once_with(task, 'fp32')
            self.assertIsNone(built.weight)

    def test_zero_weights_have_finite_positive_scales(self):
        output, scales = suite.weight_qdq_tensor(torch.zeros(2, 3), 4)
        self.assertTrue(bool((scales > 0).all()))
        self.assertTrue(bool(torch.isfinite(scales).all()))
        self.assertEqual(int(torch.count_nonzero(output)), 0)

    def test_activation_qdq_clamps_and_frozen_scales_cannot_change(self):
        scales = {name: 1.0 for name in suite.ACTIVATION_SITES}
        manager = suite._manager(None, 'qdq', scales)
        scales['model_input'] = 2.0
        output = manager.qdq('model_input', torch.tensor([-128.0, -127.0, -0.5, 1.5, 127.0, 128.0]))
        torch.testing.assert_close(output, torch.tensor([-127.0, -127.0, 0.0, 2.0, 127.0, 127.0]))
        with self.assertRaises(TypeError):
            manager.scales['model_input'] = 3.0
        with self.assertRaises(ValueError):
            suite._manager(None, 'qdq', {'model_input': 1.0})

    def test_observer_requires_every_node_and_zero_maxima_use_unit_scale(self):
        manager = suite._manager(None, 'observe')
        manager.qdq('model_input', torch.zeros(3))
        with self.assertRaises(ValueError):
            manager.freeze()
        for name in suite.ACTIVATION_SITES:
            manager.qdq(name, torch.zeros(3))
        floating, powers = manager.freeze()
        self.assertEqual(floating, {name: 1.0 for name in suite.ACTIVATION_SITES})
        self.assertEqual(floating, powers)

    def test_native_labels_preserve_softmax_rounding_at_a_near_tie(self):
        logits = torch.tensor([[0.0, 1e-8]], dtype=torch.float32)
        self.assertEqual(int(logits.argmax(1)), 1)
        self.assertEqual(int(suite.native_labels(logits)), 0)


if __name__ == '__main__':
    unittest.main()
