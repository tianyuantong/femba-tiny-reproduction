"""Pure protocol contracts plus actual CPU tensor contracts (run in the WSL environment)."""
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import femba_precision_suite as suite

try:
    import torch
except ImportError:
    torch = None


def _stub_mamba_task(scan):
    """Use the production explicit forward with tiny CPU modules and a recorded scan."""
    class Mamba(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dt_rank, self.d_state, self.activation = 1, 1, 'silu'
            self.in_proj = torch.nn.Linear(2, 4, bias=False)
            self.conv1d = torch.nn.Conv1d(2, 2, 1, groups=2, bias=False)
            self.act = torch.nn.Identity()
            self.x_proj = torch.nn.Linear(2, 3, bias=False)
            self.dt_proj = torch.nn.Linear(1, 2)
            self.out_proj = torch.nn.Linear(2, 2, bias=False)
            self.A_log, self.D = torch.nn.Parameter(torch.zeros(2, 1)), torch.nn.Parameter(torch.ones(2))
            with torch.no_grad():
                self.in_proj.weight.copy_(torch.cat((torch.eye(2), 0.23 * torch.eye(2))))
                self.conv1d.weight.fill_(0.3)
                self.x_proj.weight.copy_(torch.tensor([[0.7, 0.2], [0.3, 0.4], [0.9, 0.6]]))
                self.dt_proj.weight.copy_(torch.tensor([[0.7], [0.3]]))
                self.dt_proj.bias.zero_()
                self.out_proj.weight.copy_(torch.tensor([[0.31, 0.17], [0.23, 0.41]]))

    model = torch.nn.Module()
    model.mamba_blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for block in model.mamba_blocks:
        block.mamba_fwd, block.mamba_rev = Mamba(), Mamba()
    model.classifier = torch.nn.Module()
    model.classifier.mamba_1 = Mamba()
    model.classifier.fc1, model.classifier.fc3 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    names = ('mamba_ssm', 'mamba_ssm.modules', 'mamba_ssm.modules.mamba_simple',
             'mamba_ssm.ops', 'mamba_ssm.ops.selective_scan_interface', 'causal_conv1d')
    imports = {name: ModuleType(name) for name in names}
    imports['mamba_ssm.modules.mamba_simple'].Mamba = Mamba
    imports['mamba_ssm.ops.selective_scan_interface'].selective_scan_fn = scan
    imports['causal_conv1d'].causal_conv1d_fn = None
    return SimpleNamespace(model=model), imports


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

    def test_linear_scope_has_only_the_22_linear_operation_inputs(self):
        self.assertEqual(len(suite.LINEAR_INPUT_SITES), 22)
        self.assertEqual(len(set(suite.LINEAR_INPUT_SITES)), 22)
        self.assertFalse(set(suite.LINEAR_INPUT_SITES) & set(suite.ACTIVATION_SITES))
        for name in suite.MAMBA_NAMES:
            self.assertEqual(sum(site.startswith(name + '.') for site in suite.LINEAR_INPUT_SITES), 4)
        with self.assertRaises(ValueError):
            suite.Variant('invalid', activation_scope='unknown')

    def test_subset_requires_unique_known_variants_and_fp32_first(self):
        chosen = suite.selected_variants(['fp32', 'w8a8-linear', 'w4a8-linear'])
        self.assertEqual([variant.activation_scope for variant in chosen], ['all57', 'linear_inputs', 'linear_inputs'])
        self.assertEqual(suite.selected_variants(None), suite.VARIANTS)
        for names in ([], ['w8a8-linear'], ['fp32', 'fp32'], ['fp32', 'unknown']):
            with self.assertRaises(ValueError):
                suite.selected_variants(names)

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

    def test_linear_scope_cannot_reuse_the_legacy_gate(self):
        variant = suite.Variant('linear', 8, 'float', activation_scope='linear_inputs')
        gates = {'None': True, '8': True, '4': False}
        parity = {'weight_gate_passed': gates, 'rotation_passed': True}
        self.assertIsNotNone(suite.gate_failure_reason(variant, parity))
        parity['scope_weight_gate_passed'] = {'linear_inputs': gates.copy()}
        self.assertIsNone(suite.gate_failure_reason(variant, parity))
        parity['scope_weight_gate_passed']['linear_inputs']['None'] = False
        self.assertIsNotNone(suite.gate_failure_reason(variant, parity))

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

    def test_calibration_cache_isolates_scope_and_rotation(self):
        context = SimpleNamespace()
        def statistics(context, rotation, cohort, scope):
            return {'activation_scope': scope, 'statistics_group': suite.calibration_group(rotation, scope),
                    'scale_float': {name: 0.03 for name in suite.ACTIVATION_SCOPES[scope]},
                    'scale_pot': {name: 0.03125 for name in suite.ACTIVATION_SCOPES[scope]}}
        with TemporaryDirectory() as temporary, patch.object(suite, '_collect_calibration', side_effect=statistics) as collect:
            hashes = set()
            for rotation in (False, True):
                for scope in suite.ACTIVATION_SCOPES:
                    reports = []
                    for bits in (8, 4):
                        variant = suite.Variant(f'{rotation}-{scope}-{bits}', bits, 'float', rotation,
                                                activation_scope=scope)
                        directory = Path(temporary) / variant.name
                        directory.mkdir()
                        reports.append(suite._calibrate(context, variant, [0, 1], directory))
                    self.assertEqual(reports[0]['shared_statistics_sha256'], reports[1]['shared_statistics_sha256'])
                    hashes.add(reports[0]['shared_statistics_sha256'])
            self.assertEqual(collect.call_count, 4)
            self.assertEqual(len(hashes), 4)

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
                    'config': {'normalize': True}, 'checkpoint_sha256': 'parent',
                    'activation_scope': 'all57', 'scope_sha256': 'all57-scope'}
        restored = suite.validate_saved_metadata(deepcopy(metadata), metadata)
        self.assertEqual(restored, suite.Variant(**metadata['variant']))
        for field in metadata:
            damaged = deepcopy(metadata)
            del damaged[field]
            with self.assertRaises(ValueError):
                suite.validate_saved_metadata(damaged, metadata)
        for field, value in (('scales', {'node': 0.02}), ('rotation', {'hash': 'changed'}),
                             ('activation_scope', 'linear_inputs'), ('scope_sha256', 'wrong-scope')):
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

    def test_out_of_scope_calls_are_identity_and_never_observed(self):
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        for scope, excluded in (('all57', suite.LINEAR_INPUT_SITES[0]),
                                 ('linear_inputs', suite.ACTIVATION_SITES[0])):
            for mode in ('observe', 'qdq'):
                scales = {site: 0.1 for site in suite.ACTIVATION_SCOPES[scope]} if mode == 'qdq' else None
                manager = suite._manager(None, mode, scales, activation_scope=scope)
                self.assertIs(manager.qdq(excluded, value), value)
                self.assertEqual(manager.device_stats, {})
                self.assertEqual(manager.device_maxima, {})

    def test_each_scope_requires_its_complete_observation_and_restores_its_own_scales(self):
        for scope, sites in suite.ACTIVATION_SCOPES.items():
            observer = suite._manager(None, 'observe', activation_scope=scope)
            for name in sites[:-1]:
                observer.qdq(name, torch.tensor([0.21, -0.42]))
            with self.assertRaises(ValueError):
                observer.freeze()
            observer.qdq(sites[-1], torch.zeros(2))
            scales, _ = observer.freeze()
            variant = suite.Variant('fixture', 8, 'float', activation_scope=scope)
            metadata = {'variant': asdict(variant), 'scales': scales, 'rotation': None, 'config': {},
                        'checkpoint_sha256': 'parent', 'activation_scope': scope, 'scope_sha256': scope}
            buffer = io.BytesIO()
            torch.save(metadata, buffer)
            buffer.seek(0)
            restored = torch.load(buffer, weights_only=False)
            decoded = suite.validate_saved_metadata(restored, metadata)
            manager = suite._manager(None, 'qdq', restored['scales'], activation_scope=decoded.activation_scope)
            self.assertEqual(tuple(sorted(manager.scales)), sites)
            with self.assertRaises(TypeError):
                manager.scales[sites[0]] = 1.0
            wrong_scope = 'linear_inputs' if scope == 'all57' else 'all57'
            with self.assertRaises(ValueError):
                suite._manager(None, 'qdq', restored['scales'], activation_scope=wrong_scope)

    def test_linear_scope_quantizes_projection_inputs_without_direct_scan_quantization(self):
        scan_inputs, projection_inputs, rotations = [], {}, []
        def scan(x, dt, a, b, c, d, **kwargs):
            scan_inputs.append({'x': x.clone(), 'dt': dt.clone(), 'B': b.clone(), 'C': c.clone(),
                                'z': kwargs['z'].clone()})
            return x + 0.06
        def rotate(name, value):
            rotations.append(name)
            return value + 0.07
        task, imports = _stub_mamba_task(scan)
        scales = {name: 0.1 for name in suite.LINEAR_INPUT_SITES}
        manager = suite._manager(None, 'qdq', scales, SimpleNamespace(rotate_activation=rotate), 'linear_inputs')
        for name in suite.MAMBA_NAMES:
            for operation in ('x_proj', 'dt_proj', 'out_proj'):
                task.model.get_submodule(name + '.' + operation).register_forward_pre_hook(
                    lambda module, args, key=name + '.' + operation: projection_inputs.__setitem__(key, args[0].clone()))
        with patch.dict(sys.modules, imports):
            suite.install_explicit_mamba_forward(task, manager)
        with torch.inference_mode():
            handles = suite.install_outer_boundaries(task, manager)
            signals = torch.tensor([[[0.26, 0.73], [0.44, -0.04]]])
            for name in suite.MAMBA_NAMES:
                task.model.get_submodule(name)(signals)
            task.model.classifier.fc1(signals)
            task.model.classifier.fc3(signals)
            for handle in handles:
                handle.remove()
        self.assertEqual(len(rotations), 5)
        self.assertEqual(set(manager.device_stats), set(suite.LINEAR_INPUT_SITES))
        self.assertEqual(len(handles), 2)
        for name, received in zip(suite.MAMBA_NAMES, scan_inputs):
            torch.testing.assert_close(projection_inputs[name + '.x_proj'], torch.tensor([[0.1, 0.2], [0.1, 0.0]]))
            # The scan's convolution input is NOT the quantized x_proj input.
            torch.testing.assert_close(received['x'], torch.tensor([[[0.09, 0.12], [0.21, 0.0]]]))
            torch.testing.assert_close(received['B'], torch.tensor([[[0.11, 0.03]]]))
            torch.testing.assert_close(received['C'], torch.tensor([[[0.21, 0.09]]]))
            torch.testing.assert_close(received['dt'], torch.tensor([[[0.07, 0.07], [0.03, 0.03]]]))
            torch.testing.assert_close(received['z'], torch.tensor([[[0.069, 0.092], [0.161, 0.0]]]))
            expected_out_input = ((received['x'] + 0.06 + 0.07).transpose(1, 2) / 0.1).round() * 0.1
            torch.testing.assert_close(projection_inputs[name + '.out_proj'], expected_out_input)

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
