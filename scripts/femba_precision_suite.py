"""Evaluate a frozen paper-label FP32 parent and native-graph precision controls.

The native forward and boundary helpers are adapted from the audited WSL
implementation identified by NATIVE_SOURCE_SHA256 below.
All integer variants use FP32 fake QDQ and the installed selective-scan kernel.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
from types import MappingProxyType, MethodType, SimpleNamespace
from typing import Any


REPO = Path(__file__).resolve().parents[1]
REF = "d88596590f3bd3fce573be07646b7d3977ce7bcc"
NATIVE_SOURCE_SHA256 = "acc2d9020e9f2488505d551500435c1b851e0e69b490f5a9fe07fd27769af6ff"
MAMBA_NAMES = ("mamba_blocks.0.mamba_fwd", "mamba_blocks.0.mamba_rev",
               "mamba_blocks.1.mamba_fwd", "mamba_blocks.1.mamba_rev", "classifier.mamba_1")
MAMBA_SITES = ("in_proj_x", "in_proj_z", "conv_silu", "x_proj_dt_low", "x_proj_B",
               "x_proj_C", "dt_proj_pre_softplus", "scan_gate_output", "out_proj")
OUTER_SITES = ("model_input", "patch_embed_conv_out", "pos_embed_added",
               "encoder_block_0_bidirectional_sum", "encoder_block_1_bidirectional_sum",
               "encoder_residual_0", "encoder_residual_1", "encoder_norm_0", "encoder_norm_1",
               "classifier_fc1_out", "classifier_gelu_out", "classifier_pool_out")
ACTIVATION_SITES = sorted([f"{name}.{site}" for name in MAMBA_NAMES for site in MAMBA_SITES] + list(OUTER_SITES))
LINEAR_INPUT_SITES = sorted([f"{name}.{site}" for name in MAMBA_NAMES
                             for site in ("in_proj_input", "x_proj_input", "dt_proj_input", "out_proj_input")]
                            + ["classifier_fc1_input", "classifier_fc3_input"])
ACTIVATION_SCOPES = {"all57": tuple(ACTIVATION_SITES), "linear_inputs": tuple(LINEAR_INPUT_SITES)}
UPSTREAM_MODULES = (("models.FEMBA", "models/FEMBA.py"), ("util.train_utils", "util/train_utils.py"),
                    ("datasets.hdf5_dataset", "datasets/hdf5_dataset.py"),
                    ("data_module.finetune_data_module", "data_module/finetune_data_module.py"),
                    ("tasks.finetune_task", "tasks/finetune_task.py"))


@dataclass(frozen=True)
class Variant:
    name: str
    weight_bits: int | None = None
    activation_scale: str | None = None
    rotation: bool = False
    precision: str = "fp32"
    diagnostic: bool = False
    activation_scope: str = "all57"

    def __post_init__(self):
        if self.activation_scope not in ACTIVATION_SCOPES:
            raise ValueError(f"Unknown activation scope: {self.activation_scope}")


VARIANTS = (
    Variant("fp32"), Variant("fp16-direct", precision="fp16-direct"),
    Variant("w8a32", 8), Variant("w4a32", 4),
    Variant("rot-w8a32", 8, rotation=True), Variant("rot-w4a32", 4, rotation=True),
    Variant("w8a8-float", 8, "float"), Variant("w4a8-float", 4, "float"),
    Variant("w8a8-pot", 8, "pot"),
    Variant("w8a8-linear", 8, "float", activation_scope="linear_inputs"),
    Variant("w4a8-linear", 4, "float", activation_scope="linear_inputs"),
    Variant("rot-w8a8-float", 8, "float", True), Variant("rot-w4a8-float", 4, "float", True),
)
FP16_DIAGNOSTICS = (Variant("fp16-state-safe", precision="fp16-state-safe", diagnostic=True),
                    Variant("amp-fp16", precision="amp-fp16", diagnostic=True))


def selected_variants(names: list[str] | None) -> tuple[Variant, ...]:
    if names is None:
        return VARIANTS
    choices = {variant.name: variant for variant in VARIANTS}
    if not names or names[0] != "fp32" or len(names) != len(set(names)) or any(name not in choices for name in names):
        raise ValueError("Select unique known variants with fp32 first")
    return tuple(choices[name] for name in names)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def balanced_indices(labels: list[int], each: int = 1024, seed: int = 42) -> list[int]:
    if type(each) is not int or each <= 0 or any(label not in (0, 1) for label in labels):
        raise ValueError("Require binary labels and a positive count per class")
    generator = random.Random(seed)
    selected = []
    for label in (0, 1):
        candidates = [index for index, value in enumerate(labels) if value == label]
        if len(candidates) < each:
            raise ValueError(f"Not enough class {label} calibration samples")
        generator.shuffle(candidates)
        selected.extend(candidates[:each])
    generator.shuffle(selected)
    return selected


def install_explicit_mamba_forward(task, manager):
    from einops import rearrange
    from mamba_ssm.modules.mamba_simple import Mamba
    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    patched = []
    for module_name, sub in task.model.named_modules():
        if not isinstance(sub, Mamba):
            continue
        prefix = module_name
        def explicit_forward(self, hidden_states, inference_params=None, _prefix=prefix, _manager=manager):
            if inference_params is not None:
                raise NotImplementedError('W8A8 audit only supports inference_params=None')
            batch, seqlen, _ = hidden_states.shape
            # Match the pinned mamba_ssm non-fast path exactly: project in
            # channel-first layout, including the module bias cast.
            hidden_states = _manager.qdq(_prefix + '.in_proj_input', hidden_states)
            xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'), 'd (b l) -> b d l', l=seqlen)
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), 'd -> d 1')
            x, z = xz.chunk(2, dim=1)
            x = _manager.qdq(_prefix + '.in_proj_x', x)
            z = _manager.qdq(_prefix + '.in_proj_z', z)
            if causal_conv1d_fn is not None:
                x = causal_conv1d_fn(x=x, weight=rearrange(self.conv1d.weight, 'd 1 w -> d w'), bias=self.conv1d.bias, activation=self.activation)
            else:
                x = self.conv1d(x)[..., :seqlen]
                x = self.act(x)
            x = _manager.qdq(_prefix + '.conv_silu', x)
            # QDQ the projection input independently; scan still consumes x.
            x_input = _manager.qdq(_prefix + '.x_proj_input', rearrange(x, 'b d l -> (b l) d'))
            x_db = self.x_proj(x_input)
            dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = _manager.qdq(_prefix + '.x_proj_dt_low', dt)
            B = _manager.qdq(_prefix + '.x_proj_B', B)
            C = _manager.qdq(_prefix + '.x_proj_C', C)
            # Exact upstream projection order/layout before selective scan.
            dt = _manager.qdq(_prefix + '.dt_proj_input', dt)
            dt = self.dt_proj.weight @ dt.t()
            dt = _manager.qdq(_prefix + '.dt_proj_pre_softplus', dt)
            dt = rearrange(dt, 'd (b l) -> b d l', b=batch, l=seqlen).contiguous()
            B = rearrange(B, '(b l) n -> b n l', b=batch, l=seqlen).contiguous()
            C = rearrange(C, '(b l) n -> b n l', b=batch, l=seqlen).contiguous()
            A = -torch.exp(self.A_log.float())
            y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False)
            y = _manager.qdq(_prefix + '.scan_gate_output', y)
            y = rearrange(y, 'b d l -> b l d')
            y = _manager.qdq(_prefix + '.out_proj_input', y)
            out = self.out_proj(y)
            out = _manager.qdq(_prefix + '.out_proj', out)
            return out
        sub.forward = MethodType(explicit_forward, sub)
        patched.append(module_name)
    if len(patched) != 5:
        raise RuntimeError(f'expected five Mamba modules, found {len(patched)}: {patched}')
    return patched

def install_outer_boundaries(task, manager):
    modules = dict(task.model.named_modules())
    if manager.activation_scope == "linear_inputs":
        return [modules[name].register_forward_pre_hook(
            lambda module, args, site=site: (manager.qdq(site, args[0]), *args[1:]))
                for name, site in (("classifier.fc1", "classifier_fc1_input"),
                                   ("classifier.fc3", "classifier_fc3_input"))]
    required = ['patch_embed.proj', 'patch_embed', 'mamba_blocks.0', 'classifier.fc1', 'classifier.activation1', 'classifier.fc3']
    for name in required:
        if name not in modules:
            raise RuntimeError(f'missing required module {name}; available sample={list(modules)[:30]}')
    handles = []
    handles.append(modules['patch_embed.proj'].register_forward_hook(lambda m, a, o: manager.qdq('patch_embed_conv_out', o)))
    handles.append(modules['mamba_blocks.0'].register_forward_pre_hook(lambda m, a: (manager.qdq('pos_embed_added', a[0]), *a[1:])))
    for i in range(2):
        block_name = f'mamba_blocks.{i}'
        norm_name = f'norm_layers.{i}'
        if block_name not in modules or norm_name not in modules:
            raise RuntimeError(f'missing encoder modules {block_name}/{norm_name}')
        handles.append(modules[block_name].register_forward_hook(lambda m, a, o, n=f'encoder_block_{i}_bidirectional_sum': manager.qdq(n, o)))
        handles.append(modules[norm_name].register_forward_pre_hook(lambda m, a, n=f'encoder_residual_{i}': (manager.qdq(n, a[0]), *a[1:])))
        handles.append(modules[norm_name].register_forward_hook(lambda m, a, o, n=f'encoder_norm_{i}': manager.qdq(n, o)))
    handles.append(modules['classifier.fc1'].register_forward_hook(lambda m, a, o: manager.qdq('classifier_fc1_out', o)))
    handles.append(modules['classifier.activation1'].register_forward_hook(lambda m, a, o: manager.qdq('classifier_gelu_out', o)))
    handles.append(modules['classifier.fc3'].register_forward_pre_hook(lambda m, a: (manager.qdq('classifier_pool_out', a[0]), *a[1:])))
    return handles


def _snapshot(path: Path, directory: Path) -> dict:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    target = directory / f"{path.stem}-{digest[:16]}{path.suffix}"
    with target.open("xb") as handle:
        handle.write(content)
    return {"source_path": str(path), "sha256": digest, "snapshot": str(target)}


def _initialize_runtime(args: argparse.Namespace) -> SimpleNamespace:
    global torch, np, OmegaConf
    import numpy as np
    from omegaconf import OmegaConf
    import torch
    from scripts import femba_upstream_2025_train as adapter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = json.loads((args.parent_root / "results/upstream2025-run.json").read_text())
    final = json.loads((args.parent_root / "results/upstream2025-training-results.json").read_text())
    audit = json.loads((args.parent_root / "results/upstream2025-audit.json").read_text())
    checkpoint = Path(final["best_checkpoint"])
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != audit["checkpoints"]["best"]["checkpoint_sha256"]:
        raise ValueError("Parent best checkpoint differs from its independent audit")
    if args.checkpoint_sha256 and args.checkpoint_sha256 != checkpoint_hash:
        raise ValueError("Parent checkpoint differs from the preselected fingerprint")
    if run["upstream_commit"] != REF:
        raise ValueError("Wrong upstream source version")
    for name, path in UPSTREAM_MODULES:
        module = adapter.upstream_module(name, path)
        if adapter.source_hashes[path] != run["source_hashes"][path]:
            raise ValueError(f"Pinned source does not match parent: {path}")
    cfg = OmegaConf.create(run["config"])
    if int(cfg.batch_size) != 256:
        raise ValueError("This protocol requires upstream metric and forward batch size 256")
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return SimpleNamespace(args=args, run=run, final=final, audit=audit, cfg=cfg,
                           checkpoint=checkpoint, checkpoint_hash=checkpoint_hash, state=state,
                           task_module=module,
                           upstream_hashes=dict(adapter.source_hashes))


def _load_data(context: SimpleNamespace) -> None:
    import h5py
    from datasets.hdf5_dataset import HDF5Loader

    path = Path(context.run["data_manifest_path"])
    if sha256(path) != context.run["data_manifest_sha256"]:
        raise ValueError("Parent data manifest changed")
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete" or manifest.get("label_rule", {}).get("name") != "paper13_any":
        raise ValueError("This suite requires the complete paper13_any data protocol")
    context.data, context.datasets, context.labels, context.file_stats = manifest, {}, {}, {}
    for split in ("train", "val", "test"):
        item = manifest["outputs"][split]
        h5_path = Path(context.cfg.data_module[split].hdf5_file)
        if h5_path.resolve() != Path(item["path"]).resolve() or sha256(h5_path) != item["sha256"]:
            raise ValueError(f"HDF5 differs from frozen data manifest: {split}")
        original = Path(item["source_h5"]["path"])
        if sha256(original) != item["source_h5"]["sha256"]:
            raise ValueError(f"ExternalLink parent changed: {split}")
        for tracked in (h5_path, original):
            stat = tracked.stat()
            context.file_stats[str(tracked)] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        with h5py.File(h5_path, "r") as handle:
            for key in handle.keys():
                link = handle[key].get("X", getlink=True)
                if not isinstance(link, h5py.ExternalLink):
                    raise ValueError(f"Expected ExternalLink X: {split}/{key}")
                resolved = (h5_path.parent / link.filename).resolve()
                if resolved != original.resolve() or link.path != f"/{key}/X":
                    raise ValueError(f"ExternalLink target mismatch: {split}/{key}")
            labels = np.concatenate([handle[key]["y"][:] for key in handle.keys()]).astype(np.int64)
        if len(labels) != item["windows"] or int(labels.sum()) != item["positive"]:
            raise ValueError(f"Split label counts changed: {split}")
        if hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest() != item["new_labels_sha256"]:
            raise ValueError(f"Split label order changed: {split}")
        context.datasets[split] = HDF5Loader(str(h5_path), use_cache=False)
        context.labels[split] = labels


def _loader(context: SimpleNamespace, split: str, indices: list[int] | None = None):
    from torch.utils.data import DataLoader, Subset

    dataset = context.datasets[split]
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=int(context.cfg.batch_size), shuffle=False,
                      num_workers=0, drop_last=False, pin_memory=True)


def _new_task(context: SimpleNamespace):
    task = context.task_module.FinetuneTask(context.cfg)
    task.load_state_dict(context.state, strict=True)
    return task.cuda().eval()


def weight_qdq_tensor(weight: Any, bits: int) -> tuple:
    if bits not in (4, 8) or weight.ndim < 2 or weight.dtype != torch.float32:
        raise ValueError("Require FP32 weights with output-channel dimension and 4 or 8 bits")
    limit = 2 ** (bits - 1) - 1
    scale = weight.abs().amax(dim=tuple(range(1, weight.ndim)), keepdim=True).clamp_min(1e-8) / limit
    return (weight / scale).round().clamp(-limit, limit) * scale, scale


def _weight_qdq(task: Any, bits: int) -> dict:
    limit = 2 ** (bits - 1) - 1
    layers = []
    with torch.no_grad():
        for name, module in task.model.named_modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                continue
            weight = module.weight.detach()
            dequantized, scale = weight_qdq_tensor(weight, bits)
            error = (weight - dequantized).abs()
            layers.append({"name": name, "shape": list(weight.shape), "bits": bits,
                           "scale_min": float(scale.min()), "scale_max": float(scale.max()),
                           "max_abs_error": float(error.max())})
            module.weight.copy_(dequantized)
    if len(layers) != 28:
        raise ValueError(f"Expected 28 quantized weight tensors, found {len(layers)}")
    return {"qmin": -limit, "qmax": limit, "zero_point": 0, "rounding": "torch.round",
            "granularity": "per_output_channel", "layers": layers}


def _manager(context: SimpleNamespace, mode: str, scales: dict | None = None, rotation: Any = None,
             activation_scope: str = "all57"):
    if activation_scope not in ACTIVATION_SCOPES:
        raise ValueError(f"Unknown activation scope: {activation_scope}")
    sites = ACTIVATION_SCOPES[activation_scope]
    site_names = frozenset(sites)
    class Manager:
        def __init__(self):
            if mode not in ("off", "observe", "qdq"):
                raise ValueError("Unknown activation mode")
            if mode == "qdq" and (scales is None or tuple(sorted(scales)) != sites):
                raise ValueError(f"QDQ requires exactly {len(sites)} frozen {activation_scope} scales")
            self.activation_scope = activation_scope
            self.mode, self.scales = mode, MappingProxyType(dict(scales or {}))
            self.max_abs, self.calibration_counts = {}, {}
            self.device_maxima, self.device_stats, self.device_scales = {}, {}, {}

        def qdq(self, name, value):
            if rotation is not None and name.endswith(".scan_gate_output"):
                value = rotation.rotate_activation(name, value)
            if self.mode == "off" or name not in site_names:
                return value
            if self.mode == "observe":
                maximum = value.detach().abs().amax()
                self.device_maxima[name] = torch.maximum(self.device_maxima.get(name, maximum), maximum)
                self.calibration_counts[name] = self.calibration_counts.get(name, 0) + value.numel()
                return value
            if name not in self.device_scales:
                scale = self.scales[name]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Invalid activation scale at {name}")
                self.device_scales[name] = torch.tensor(scale, device=value.device, dtype=value.dtype)
            scale = self.device_scales[name]
            rounded = (value / scale).round()
            integers = rounded.clamp(-127, 127)
            output, error = integers * scale, value - integers * scale
            counts = torch.stack([((rounded < -127) | (rounded > 127)).sum(),
                                  ((value != 0) & (integers == 0)).sum(),
                                  (~torch.isfinite(value)).sum()])
            sums = torch.stack([value.float().square().sum(), error.float().square().sum(), error.float().abs().sum()])
            maximum = value.detach().abs().amax()
            if name not in self.device_stats:
                self.device_stats[name] = [0, 0, torch.zeros_like(counts), torch.zeros_like(sums), maximum]
            stats = self.device_stats[name]
            stats[0] += 1
            stats[1] += value.numel()
            stats[2] += counts
            stats[3] += sums
            stats[4] = torch.maximum(stats[4], maximum)
            return output

        def freeze(self):
            names = sorted(self.device_maxima)
            if tuple(names) != sites:
                raise ValueError(f"Calibration requires exactly {len(sites)} {activation_scope} boundaries")
            values = torch.stack([self.device_maxima[name] for name in names]).cpu().tolist()
            self.max_abs = dict(zip(names, values))
            if any(not math.isfinite(value) for value in values):
                raise FloatingPointError("Nonfinite activation calibration maximum")
            floating = {name: maximum / 127.0 if maximum else 1.0 for name, maximum in self.max_abs.items()}
            powers = {name: 2.0 ** math.ceil(math.log2(scale)) for name, scale in floating.items()}
            return floating, powers

        def finalized_stats(self):
            results = {}
            for name, (calls, elements, counts, sums, maximum) in sorted(self.device_stats.items()):
                clipped, zeroized, nonfinite = counts.cpu().tolist()
                signal, squared_error, abs_error = sums.cpu().tolist()
                if nonfinite:
                    raise FloatingPointError(f"Nonfinite activation values at {name}: {nonfinite}")
                sqnr = 10 * math.log10(signal / squared_error) if signal > 0 and squared_error > 0 else None
                results[name] = {"calls": calls, "elements": elements, "clip_rate": clipped / elements,
                                 "zeroized_rate": zeroized / elements, "nonfinite_count": nonfinite,
                                 "qdq_mae": abs_error / elements,
                                 "sqnr_db": sqnr if sqnr is None or math.isfinite(sqnr) else None,
                                 "max_abs_input": float(maximum.cpu())}
            return results
    return Manager()


def _cast_model(task: Any, precision: str) -> dict:
    if precision == "fp16-direct":
        task.model.half()
    elif precision == "fp16-state-safe":
        for name, parameter in task.model.named_parameters():
            keep = name.endswith((".A_log", ".D", ".dt_proj.bias"))
            parameter.data = parameter.data.to(torch.float32 if keep else torch.float16)
        for module in task.model.modules():
            for name, value in module._buffers.items():
                if value is not None and value.is_floating_point():
                    module._buffers[name] = value.half()
    elif precision not in ("fp32", "amp-fp16"):
        raise ValueError(f"Unknown precision {precision}")
    return {name: str(value.dtype) for name, value in task.model.named_parameters()}


def _build_variant(context: SimpleNamespace, variant: Variant, scales: dict | None = None,
                   observe: bool = False, rotation_mode: str | None = None) -> SimpleNamespace:
    task, plan = _new_task(context), None
    if variant.rotation or rotation_mode:
        from biofoundation.femba_rotation import prepare_rotations
        plan = prepare_rotations(task.model, seed=42, block=128, mode=rotation_mode or "outproj_h128")
        plan.transform_out_proj_weights_(task.model)
    weight = _weight_qdq(task, variant.weight_bits) if variant.weight_bits and not observe else None
    mode = "observe" if observe else ("qdq" if variant.activation_scale else "off")
    manager = _manager(context, mode, scales, plan, variant.activation_scope)
    handles, patched = [], []
    if observe or variant.activation_scale or plan is not None:
        patched = install_explicit_mamba_forward(task, manager)
        handles = install_outer_boundaries(task, manager)
    dtypes = _cast_model(task, variant.precision)
    return SimpleNamespace(task=task, manager=manager, handles=handles, plan=plan,
                           patched=patched, weight=weight, dtypes=dtypes, variant=variant)


def _step(built: SimpleNamespace, signals: Any):
    task, variant = built.task, built.variant
    signals = signals.cuda(non_blocking=True).float()
    normalized = task.normalize_fct(signals) if getattr(task, "normalize", False) else signals
    value = built.manager.qdq("model_input", normalized)
    if variant.precision in ("fp16-direct", "fp16-state-safe"):
        value = value.half()
    mask = task.generate_fake_mask(len(value), value.shape[1], value.shape[2])
    scope = torch.autocast("cuda", dtype=torch.float16) if variant.precision == "amp-fp16" else nullcontext()
    with scope:
        return task._step(value, mask)["logits"]


def _close(built: SimpleNamespace) -> None:
    for handle in built.handles:
        handle.remove()
    built.handles.clear()
    built.task.cpu()


def _capture(built: SimpleNamespace, signals: Any) -> dict:
    captures, handles = {}, []
    for name in MAMBA_NAMES:
        module = built.task.model.get_submodule(name).out_proj
        handles.append(module.register_forward_hook(
            lambda module, args, output, name=name: captures.__setitem__(name, output.detach().cpu())))
    try:
        with torch.inference_mode():
            captures["logits"] = _step(built, signals).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return captures


def _gate_pair(context: SimpleNamespace, signals: Any, bits: int | None,
               activation_scope: str = "all57") -> dict:
    from biofoundation.femba_rotation import compare_parity, tensor_parity

    native = _build_variant(context, Variant("gate-native", bits))
    try:
        with torch.inference_mode():
            fast = _step(native, signals).detach().cpu()
        for name in MAMBA_NAMES:
            native.task.model.get_submodule(name).use_fast_path = False
        reference = _capture(native, signals)
    finally:
        _close(native)
    explicit = _build_variant(context, Variant("gate-explicit", bits, activation_scope=activation_scope),
                              rotation_mode="none")
    try:
        candidate = _capture(explicit, signals)
    finally:
        _close(explicit)
    return {"native_fast_vs_slow": tensor_parity(fast, reference["logits"]),
            "native_slow_vs_explicit_off": compare_parity(reference, candidate)}


def _gate_rotations(context: SimpleNamespace, signals: Any) -> dict:
    from biofoundation.femba_rotation import compare_parity

    reference_model = _build_variant(context, Variant("gate-off"), rotation_mode="none")
    try:
        reference = _capture(reference_model, signals)
    finally:
        _close(reference_model)
    reports = {}
    for mode in ("identity", "outproj_h128"):
        candidate = _build_variant(context, Variant("gate-rotation"), rotation_mode=mode)
        try:
            reports[mode] = compare_parity(reference, _capture(candidate, signals))
            reports[mode]["rotation_spec"] = candidate.plan.to_dict()
        finally:
            _close(candidate)
    return reports


def _report_passed(report: dict) -> bool:
    """Parity reports must explicitly report a pass; never infer from error size."""
    for key in ("passed", "allclose"):
        if key in report:
            return report[key] is True
    raise ValueError("Parity report lacks its explicit pass field")


def _run_gates(context: SimpleNamespace, cohort: list[int], directory: Path) -> dict:
    gate_indices = {"train": cohort[:32], "val": balanced_indices(context.labels["val"].tolist(), 16, 43)}
    scopes = sorted({variant.activation_scope for variant in context.variants})
    evidence = {"sample_indices": gate_indices, "splits": {}, "base_passed": True,
                "scope_weight_gate_passed": {scope: {"None": True, "8": True, "4": True} for scope in scopes},
                "rotation_passed": True}
    for split, indices in gate_indices.items():
        signals, _ = next(iter(_loader(context, split, indices)))
        scoped_rows = {}
        for scope in scopes:
            rows = {}
            for bits in (None, 8, 4):
                try:
                    rows[str(bits)] = _gate_pair(context, signals, bits, scope)
                    passed = all(_report_passed(part) for part in rows[str(bits)].values())
                except (RuntimeError, ValueError, FloatingPointError, TypeError, KeyError) as error:
                    rows[str(bits)] = {"passed": False, "failure": str(error)}
                    passed = False
                evidence["scope_weight_gate_passed"][scope][str(bits)] &= passed
            scoped_rows[scope] = rows
        try:
            rotations = _gate_rotations(context, signals)
            rotation_passed = all(_report_passed(row) for row in rotations.values())
        except (RuntimeError, ValueError, FloatingPointError, TypeError, KeyError) as error:
            rotations, rotation_passed = {"passed": False, "failure": str(error)}, False
        evidence["splits"][split] = {"explicit_qdq_off": scoped_rows["all57"],
                                     "explicit_qdq_off_by_scope": scoped_rows, "rotation_qdq_off": rotations}
        evidence["rotation_passed"] &= rotation_passed
    evidence["weight_gate_passed"] = evidence["scope_weight_gate_passed"]["all57"]
    evidence["base_passed"] = all(evidence["weight_gate_passed"].values())
    write_json(directory / "parity.json", evidence)
    return evidence


def calibration_group(rotation: bool, activation_scope: str) -> str:
    group = "rotated_fp32" if rotation else "plain_fp32"
    return group if activation_scope == "all57" else f"{group}:{activation_scope}"


def _collect_calibration(context: SimpleNamespace, rotation: bool, cohort: list[int],
                         activation_scope: str = "all57") -> dict:
    built = _build_variant(context, Variant("calibration", rotation=rotation, activation_scope=activation_scope), observe=True)
    try:
        with torch.inference_mode():
            for signals, _ in _loader(context, "train", cohort):
                logits = _step(built, signals)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite calibration logits")
        floating, powers = built.manager.freeze()
        if tuple(sorted(floating)) != ACTIVATION_SCOPES[activation_scope]:
            raise ValueError("Calibration did not execute exactly the declared scope boundaries")
        return {"statistics_group": calibration_group(rotation, activation_scope),
                       "activation_scope": activation_scope,
                       "calibration_model": "rot-fp32" if rotation else "fp32",
                       "weight_quantization_enabled": False, "activation_quantization_enabled": False,
                       "cohort_sha256": context.cohort_hash,
                       "scope_sha256": context.scope_hashes[activation_scope], "sample_count": len(cohort),
                       "max_abs": built.manager.max_abs, "elements": built.manager.calibration_counts,
                       "scale_float": floating, "scale_pot": powers,
                       "rotation": built.plan.to_dict() if built.plan else None}
    finally:
        _close(built)


def _calibrate(context: SimpleNamespace, variant: Variant, cohort: list[int], out: Path) -> dict:
    if not hasattr(context, "calibration_cache"):
        context.calibration_cache = {}
    group = calibration_group(variant.rotation, variant.activation_scope)
    if group not in context.calibration_cache:
        context.calibration_cache[group] = _collect_calibration(context, variant.rotation, cohort, variant.activation_scope)
    statistics = deepcopy(context.calibration_cache[group])
    calibration = {**statistics, "variant": variant.name,
                   "shared_statistics_sha256": hashlib.sha256(canonical(statistics)).hexdigest(),
                   "scales": statistics["scale_pot" if variant.activation_scale == "pot" else "scale_float"]}
    write_json(out / "calibration.json", calibration)
    return calibration


def gate_failure_reason(variant: Variant, parity: dict) -> str | None:
    if variant.activation_scale or variant.rotation:
        gates = parity.get("scope_weight_gate_passed", {}).get(variant.activation_scope, {})
        if variant.activation_scope == "all57" and not gates:
            gates = parity.get("weight_gate_passed", {})
        if gates.get("None") is not True or gates.get(str(variant.weight_bits)) is not True:
            return "explicit_qdq_off_gate_failed"
    if variant.rotation and parity.get("rotation_passed") is not True:
        return "rotation_qdq_off_gate_failed"
    return None


def _identity_quant_gate(context: SimpleNamespace, variant: Variant, scales: dict,
                         cohort: list[int], out: Path) -> None:
    from biofoundation.femba_rotation import compare_parity

    reports = {}
    plain = Variant("identity-reference", variant.weight_bits, variant.activation_scale,
                    activation_scope=variant.activation_scope)
    for split, indices in (("train", cohort[:32]),
                           ("val", balanced_indices(context.labels["val"].tolist(), 16, 43))):
        signals, _ = next(iter(_loader(context, split, indices)))
        reference = _build_variant(context, plain, scales, rotation_mode="none")
        candidate = _build_variant(context, plain, scales, rotation_mode="identity")
        try:
            reports[split] = compare_parity(_capture(reference, signals), _capture(candidate, signals))
        finally:
            _close(reference)
            _close(candidate)
    write_json(out / "identity-qdq-parity.json", reports)
    if not all(_report_passed(report) for report in reports.values()):
        raise ValueError("Identity rotation does not preserve the quantized path")


def performance_report(metrics: dict, baseline: dict) -> dict:
    keys = ("accuracy", "balanced_accuracy", "auroc", "average_precision")
    differences = {key: float(metrics[key] - baseline[key]) for key in keys}
    passed = all(value >= -0.005 - 1e-12 for value in differences.values())
    return {"maximum_allowed_drop": 0.005, "passed": passed, "delta_vs_fp32": differences}


def native_performance_metrics(metrics: dict) -> dict:
    names = {"accuracy": "test_BinaryAccuracy", "balanced_accuracy": "test_MulticlassRecall",
             "auroc": "test_BinaryAUROC", "average_precision": "test_BinaryAveragePrecision"}
    return {name: metrics[key] for name, key in names.items()}


def native_labels(logits: Any):
    return logits.float().softmax(1).argmax(1)


def validate_saved_metadata(payload: dict, expected: dict) -> Variant:
    """Reject incomplete or changed inference metadata before loading tensors."""
    for key in ("variant", "scales", "rotation", "config", "checkpoint_sha256", "activation_scope", "scope_sha256"):
        if key not in payload or canonical(payload[key]) != canonical(expected[key]):
            raise ValueError(f"Saved inference metadata mismatch: {key}")
    variant = Variant(**payload["variant"])
    if variant.activation_scope != payload["activation_scope"]:
        raise ValueError("Saved scope differs from its variant")
    return variant


def _softmax_metrics(logits: Any, labels: Any) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score)

    z, y = logits.float(), labels.long().numpy()
    probabilities = z.softmax(1)
    pred, score = probabilities.argmax(1).numpy(), probabilities[:, 1].numpy()
    return {"samples": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, score)), "average_precision": float(average_precision_score(y, score)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "cross_entropy": float(torch.nn.functional.cross_entropy(z, labels.long())),
            "predicted_positive_rate": float(pred.mean()), "label_positive_rate": float(y.mean()),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def _first_nonfinite_hooks(task: Any) -> tuple[list, list]:
    first, handles = [], []
    for name in ("patch_embed", "mamba_blocks.0", "norm_layers.0", "mamba_blocks.1", "norm_layers.1", "classifier"):
        def hook(module, args, output, name=name):
            if not first and torch.is_tensor(output) and not torch.isfinite(output).all():
                first.append({"module": name, "dtype": str(output.dtype), "shape": list(output.shape),
                              "nan_count": int(torch.isnan(output).sum()), "inf_count": int(torch.isinf(output).sum())})
        handles.append(task.model.get_submodule(name).register_forward_hook(hook))
    return first, handles


def _evaluate(context: SimpleNamespace, built: SimpleNamespace, split: str, out: Path) -> dict:
    task = built.task
    label_metrics, logit_metrics = getattr(task, split + "_label_metrics"), getattr(task, split + "_logit_metrics")
    label_metrics.reset()
    logit_metrics.reset()
    logits, labels, batches, seen = [], [], [], 0
    first, hooks = _first_nonfinite_hooks(task) if built.variant.precision != "fp32" else ([], [])
    started, failure = time.perf_counter(), None
    try:
        with torch.inference_mode():
            for batch_index, (signals, y) in enumerate(_loader(context, split)):
                try:
                    z = _step(built, signals)
                    if not torch.isfinite(z).all():
                        raise FloatingPointError(f"NaN={int(torch.isnan(z).sum())} Inf={int(torch.isinf(z).sum())}")
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    failure = {"type": type(error).__name__, "message": str(error), "batch_index": batch_index,
                               "samples_before_failure": seen, "batch_samples": len(y), "first_nonfinite_module": first}
                    break
                z, y_cuda = z.float(), y.cuda()
                label_metrics(native_labels(z), y_cuda)
                logit_metrics(task._handle_binary(z), y_cuda)
                logits.append(z.cpu())
                labels.append(y.long().cpu())
                batches.append(len(y))
                seen += len(y)
    finally:
        for handle in hooks:
            handle.remove()
    status = "ok" if failure is None else ("numeric_failure" if failure["type"] == "FloatingPointError" else "error")
    report = {"split": split, "status": status, "samples": seen,
              "expected_samples": len(context.labels[split]), "failure": failure,
              "seconds": time.perf_counter() - started}
    if failure:
        return report
    joined_logits, joined_labels = torch.cat(logits), torch.cat(labels)
    if not np.array_equal(joined_labels.numpy(), context.labels[split]):
        raise ValueError("Evaluation labels/order differ from frozen split")
    native = {key: float(value.cpu()) for collection in (label_metrics, logit_metrics)
              for key, value in collection.compute().items()}
    predictions = out / f"{split}-predictions.pt"
    torch.save({"logits": joined_logits, "labels": joined_labels, "batch_sizes": batches,
                "sample_index": torch.arange(seen)}, predictions)
    report.update({"native_metrics": native, "softmax_metrics": _softmax_metrics(joined_logits, joined_labels),
                   "predictions_path": str(predictions), "predictions_sha256": sha256(predictions),
                   "labels_sha256": hashlib.sha256(joined_labels.numpy().astype("<i8").tobytes()).hexdigest(),
                   "nonfinite_logits": 0})
    return report


def _save_reload(context: SimpleNamespace, built: SimpleNamespace, calibration: dict | None,
                 cohort: list[int], out: Path) -> dict:
    artifact = out / "inference-state.pt"
    scales = calibration["scales"] if calibration else None
    metadata = {"variant": asdict(built.variant), "scales": scales,
                "rotation": built.plan.to_dict() if built.plan else None,
                "activation_scope": built.variant.activation_scope,
                "scope_sha256": context.scope_hashes[built.variant.activation_scope],
                "config": context.run["config"], "checkpoint_sha256": context.checkpoint_hash}
    torch.save({"model_state": {key: value.detach().cpu() for key, value in built.task.model.state_dict().items()},
                **metadata}, artifact)
    signals, _ = next(iter(_loader(context, "train", cohort[:32])))
    with torch.inference_mode():
        reference = _step(built, signals).float().cpu()
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    restored_variant = validate_saved_metadata(payload, metadata)
    reloaded = _build_variant(context, restored_variant, payload["scales"])
    try:
        restored_rotation = reloaded.plan.to_dict() if reloaded.plan else None
        if canonical(restored_rotation) != canonical(payload["rotation"]):
            raise ValueError("Saved rotation metadata does not reconstruct the same transform")
        reloaded.task.model.load_state_dict(payload["model_state"], strict=True)
        with torch.inference_mode():
            candidate = _step(reloaded, signals).float().cpu()
        equal = bool(torch.equal(reference, candidate))
        report = {"artifact_path": str(artifact), "artifact_sha256": sha256(artifact),
                  "probe_split": "train", "probe_indices": cohort[:32], "strict_load": True,
                  "metadata_verified": True,
                  "logits_exact_equal": equal, "max_abs_error": float((reference - candidate).abs().max())}
        if not equal:
            raise RuntimeError(f"Saved-state roundtrip changed logits: {report}")
        return report
    finally:
        _close(reloaded)


def _row_header(context: SimpleNamespace, variant: Variant) -> dict:
    return {"variant": asdict(variant), "status": "started", "checkpoint_sha256": context.checkpoint_hash,
            "activation_scope": variant.activation_scope,
            "data_manifest_sha256": context.run["data_manifest_sha256"], "scope_sha256": context.scope_hashes[variant.activation_scope],
            "source_set_sha256": context.source_hash, "cohort_sha256": context.cohort_hash}


def _run_variant(context: SimpleNamespace, variant: Variant, cohort: list[int], output: Path,
                 baseline: dict | None = None) -> dict:
    out = output / variant.name
    out.mkdir()
    result = _row_header(context, variant)
    built = None
    try:
        calibration = _calibrate(context, variant, cohort, out) if variant.activation_scale else None
        scales = calibration["scales"] if calibration else None
        if variant.rotation:
            _identity_quant_gate(context, variant, scales, cohort, out)
        before = hashlib.sha256(canonical(scales)).hexdigest()
        built = _build_variant(context, variant, scales)
        result.update({"weight_quantization": built.weight, "parameter_dtypes": built.dtypes,
                       "rotation": built.plan.to_dict() if built.plan else None,
                       "calibration_sha256": sha256(out / "calibration.json") if calibration else None})
        result["validation"] = _evaluate(context, built, "val", out)
        if result["validation"]["status"] != "ok":
            result.update({"status": result["validation"]["status"],
                           "test": {"status": "skipped", "reason": "validation_failed"}})
        else:
            result["test"] = _evaluate(context, built, "test", out)
            result["status"] = result["test"]["status"]
        if hashlib.sha256(canonical(dict(built.manager.scales) or None)).hexdigest() != before:
            raise ValueError("Activation scales changed during evaluation")
        if variant.activation_scale:
            if set(built.manager.device_stats) != set(ACTIVATION_SCOPES[variant.activation_scope]):
                raise ValueError("Not all calibrated activation sites executed")
            stats = built.manager.finalized_stats()
            for item in stats.values():
                if item["sqnr_db"] is not None and not math.isfinite(item["sqnr_db"]):
                    item["sqnr_db"] = None
            write_json(out / "activation-stats.json", {"scope": "validation+test before reload probe",
                        "activation_scope": variant.activation_scope, "sites": stats})
        if result["status"] == "ok":
            result["roundtrip"] = _save_reload(context, built, calibration, cohort, out)
            if baseline is not None:
                result["performance"] = {
                    "native": performance_report(native_performance_metrics(result["test"]["native_metrics"]),
                                                 native_performance_metrics(baseline["native_metrics"])),
                    "softmax_diagnostic": performance_report(result["test"]["softmax_metrics"], baseline["softmax_metrics"])}
                if not all(item["passed"] for item in result["performance"].values()):
                    result["status"] = "valid_negative"
    except (FloatingPointError, RuntimeError, ValueError, KeyError, TypeError) as error:
        status = "numeric_failure" if isinstance(error, FloatingPointError) else "error"
        if "Identity rotation" in str(error):
            status = "gate_failed"
        result.update({"status": status, "failure": {"type": type(error).__name__, "message": str(error)}})
    finally:
        if built is not None:
            _close(built)
    write_json(out / "result.json", result)
    print("PRECISION_ROW", variant.name, result["status"], flush=True)
    return result


def _verify_fp32(context: SimpleNamespace, result: dict) -> dict:
    if result["status"] != "ok":
        raise RuntimeError("FP32 reference failed")
    actual, expected = result["test"]["native_metrics"], context.audit["checkpoints"]["best"]["native_metrics"]
    errors = {key: abs(actual[key] - value) for key, value in expected.items()}
    if not errors or max(errors.values()) >= 1e-5:
        raise ValueError(f"FP32 native metrics disagree with parent audit: {errors}")
    return {"native_metric_errors": errors, "max_abs_error": max(errors.values())}


def scope_definition(name: str) -> dict:
    if name not in ACTIVATION_SCOPES:
        raise ValueError(f"Unknown activation scope: {name}")
    return {"activation_scope": name, "activation_sites": list(ACTIVATION_SCOPES[name]),
             "activation_bits": 8, "qmin": -127, "qmax": 127,
             "activation_granularity": "per_tensor", "weight_granularity": "per_output_channel",
             "fake_quantization": True, "integer_kernels": False,
             "calibration_weight_quantization": False, "calibration_activation_quantization": False,
             "calibration_statistics": "one FP32 observation per plain/rotated group and activation scope; shared across W4/W8 and float/pot scales",
             "floating_scope": ["bias", "A_log", "D", "pos_embed", "LayerNorm", "selective_scan", "nonlinearities", "arithmetic"]}


def _manifest(context: SimpleNamespace, output: Path, cohort: list[int]) -> dict:
    source_dir = output / "sources"
    source_dir.mkdir()
    sources = {}
    from biofoundation import femba_rotation
    from mamba_ssm.modules import mamba_simple
    from scripts import femba_upstream_2025_train
    for name, path in {"suite": Path(__file__),
                       "rotation": Path(femba_rotation.__file__), "upstream_adapter": Path(femba_upstream_2025_train.__file__),
                       "mamba_runtime": Path(inspect.getsourcefile(mamba_simple))}.items():
        sources[name] = _snapshot(path, source_dir)
    context.source_hash = hashlib.sha256(canonical({name: item["sha256"] for name, item in sources.items()})).hexdigest()
    scopes = {name: scope_definition(name) for name in ACTIVATION_SCOPES}
    context.scope_hashes = {name: hashlib.sha256(canonical(value)).hexdigest() for name, value in scopes.items()}
    context.scope_hash = context.scope_hashes["all57"]
    cohort_info = {"seed": 42, "split": "train", "indices": cohort,
                   "labels": [int(context.labels["train"][index]) for index in cohort],
                   "data_manifest_sha256": context.run["data_manifest_sha256"],
                   "train_identity_sha256": context.data["outputs"]["train"]["sample_identity_sha256"]}
    context.cohort_hash = hashlib.sha256(canonical(cohort_info)).hexdigest()
    write_json(output / "cohort.json", {**cohort_info, "identity_sha256": context.cohort_hash})
    manifest = {"checkpoint": str(context.checkpoint), "checkpoint_sha256": context.checkpoint_hash,
                "parent_root": str(context.args.parent_root), "parent_run_sha256": sha256(context.args.parent_root / "results/upstream2025-run.json"),
                "data_manifest": context.run["data_manifest_path"], "data_manifest_sha256": context.run["data_manifest_sha256"],
                "config": context.run["config"], "upstream_commit": REF, "upstream_hashes": context.upstream_hashes,
                "sources": sources, "native_helpers_original_sha256": NATIVE_SOURCE_SHA256, "source_set_sha256": context.source_hash,
                "scope": scopes["all57"], "scope_sha256": context.scope_hash,
                "scopes": scopes, "scope_sha256_by_name": context.scope_hashes,
                "requested_variants": [asdict(variant) for variant in context.variants],
                "cohort_sha256": context.cohort_hash, "batch_size": int(context.cfg.batch_size),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "evaluation": "full validation then full test; native TorchMetrics at batch_size=256 (raw positive logit) plus FP32 softmax diagnostic; scores cast FP32 for all modes; no test-based selection"}
    write_json(output / "manifest.json", manifest)
    return manifest


def _finish(context: SimpleNamespace, output: Path, results: list[dict], parity: dict) -> None:
    for filename, expected in context.file_stats.items():
        stat = Path(filename).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != expected:
            raise ValueError("Frozen data file changed while the suite was running")
    write_json(output / "results.json", {"status": "complete", "parity": parity,
                                          "checkpoint_sha256": context.checkpoint_hash, "results": results})
    lines = ["# Common-parent precision comparison", "", "Native metrics retain the upstream batch-256 TorchMetrics scoring. Softmax columns are diagnostics; all rows share the same paper13_any test labels.", "",
             "| Mode | Native AUROC | Native AP | Accuracy | BA | Softmax AUROC | Softmax AP | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for result in results:
        accepted = result["status"] in ("ok", "valid_negative")
        metrics = result.get("test", {}).get("softmax_metrics", {}) if accepted else {}
        native = result.get("test", {}).get("native_metrics", {}) if accepted else {}
        values = [f"{native[key]:.6f}" if key in native else "—" for key in
                  ("test_BinaryAUROC", "test_BinaryAveragePrecision", "test_BinaryAccuracy", "test_MulticlassRecall")]
        values += [f"{metrics[key]:.6f}" if key in metrics else "—" for key in ("auroc", "average_precision")]
        lines.append(f"| {result['variant']['name']} | {' | '.join(values)} | {result['status']} |")
    with (output / "summary.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError("Output root already exists; use a new directory")
    variants = selected_variants(getattr(args, "variants", None))
    context = _initialize_runtime(args)
    context.variants = variants
    _load_data(context)
    args.output_root.mkdir(parents=True, exist_ok=False)
    cohort = balanced_indices(context.labels["train"].tolist())
    _manifest(context, args.output_root, cohort)
    parity = _run_gates(context, cohort, args.output_root)
    results = []
    for variant in variants:
        reason = gate_failure_reason(variant, parity)
        if reason:
            blocked = {**_row_header(context, variant), "status": "gate_failed", "reason": reason}
            (args.output_root / variant.name).mkdir()
            write_json(args.output_root / variant.name / "result.json", blocked)
            results.append(blocked)
            continue
        baseline = results[0]["test"] if results else None
        result = _run_variant(context, variant, cohort, args.output_root, baseline)
        results.append(result)
        if variant.name == "fp32":
            write_json(args.output_root / "fp32-parent-check.json", _verify_fp32(context, result))
        if variant.name == "fp16-direct" and result["status"] in ("numeric_failure", "error"):
            results.extend(_run_variant(context, diagnostic, cohort, args.output_root, baseline) for diagnostic in FP16_DIAGNOSTICS)
    _finish(context, args.output_root, results, parity)
    print("PRECISION_SUITE_COMPLETE", args.output_root, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--variants", nargs="+", choices=[variant.name for variant in VARIANTS],
                        help="Explicit subset in evaluation order; fp32 must be first")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    existed_before = args.output_root.exists()
    try:
        run(args)
    except Exception as error:
        if not existed_before and args.output_root.is_dir() and not (args.output_root / "failure.json").exists():
            write_json(args.output_root / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
