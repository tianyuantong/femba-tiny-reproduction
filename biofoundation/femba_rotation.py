"""Fixed FP32 output-projection rotation and non-negotiable parity checks.

Call ``prepare_rotations`` after placing a fresh FP32 model on its device,
transform its weights once, and only then apply weight QDQ. The explicit
Mamba path must call ``rotate_activation`` BEFORE the scan-output A8 QDQ.
This module does not change scan, gates, normalization, or model outputs.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
import hashlib
import math
from typing import Mapping
import weakref

import torch
from torch import nn


TARGETS = (
    "mamba_blocks.0.mamba_fwd",
    "mamba_blocks.0.mamba_rev",
    "mamba_blocks.1.mamba_fwd",
    "mamba_blocks.1.mamba_rev",
    "classifier.mamba_1",
)
EXPECTED_DIMENSIONS = (1540, 1540, 1540, 1540, 512)
BLOCK_SIZE = 128
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5
ELEMENTWISE_PROTOCOL = "elementwise-v1"
VECTOR_RMS_PROTOCOL = "vector-rms-2026-09-22"
BASIS_PROTOCOL = "fixed-pdh-basis-v1"
NEGATIVE_PROTOCOL = "fixed-pdh-negative-v1"


def _check_fp32(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise TypeError(f"Rotation requires FP32, got {tensor.dtype}")
    if tensor.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable CUDA matmul TF32 before rotation")


def _hadamard(order: int) -> torch.Tensor:
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1),
                            torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(order)


def _rotation(dimension: int, seed: int, identity: bool) -> tuple[torch.Tensor, dict]:
    blocks = [BLOCK_SIZE] * (dimension // BLOCK_SIZE)
    if dimension % BLOCK_SIZE:
        blocks.append(dimension % BLOCK_SIZE)
    if not blocks:
        raise ValueError("Rotation dimension must be positive")
    hadamard = torch.block_diag(*(_hadamard(size) for size in blocks))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(dimension, generator=generator)
    signs = torch.randint(0, 2, (dimension,), generator=generator).float() * 2 - 1
    # P = I[permutation]; selecting signed rows is exactly P @ D @ H.
    matrix = (hadamard[permutation] * signs[permutation, None]).contiguous()
    if identity:
        matrix = torch.eye(dimension, dtype=torch.float32)
        permutation = torch.arange(dimension)
        signs = torch.ones(dimension, dtype=torch.float32)
    metadata = {"dimension": dimension, "seed": seed, "blocks": blocks,
                "permutation": permutation.tolist(), "signs": signs.int().tolist(),
                "matrix_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest()}
    return matrix, metadata


def _projections(model: nn.Module) -> dict[str, nn.Linear]:
    modules = dict(model.named_modules())
    projections = {}
    for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
        projection = modules.get(f"{name}.out_proj")
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"Missing Linear {name}.out_proj")
        if projection.in_features != dimension:
            raise ValueError(f"Unexpected {name} dimension: {projection.in_features}")
        _check_fp32(projection.weight)
        projections[name] = projection
    return projections


@dataclass
class _BlockRotation:
    inverse_permutation: torch.Tensor
    signs: torch.Tensor
    full_block: torch.Tensor
    tail_block: torch.Tensor | None
    identity: bool

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        _check_fp32(value)
        if value.device != self.signs.device:
            raise ValueError("Prepare rotations on the model's final device")
        if self.identity:
            return value
        # For P = I[perm], x @ P selects x[inverse_perm], then D signs columns.
        with torch.autocast(value.device.type, enabled=False):
            signed = value.index_select(-1, self.inverse_permutation) * self.signs
            full_width = value.shape[-1] // BLOCK_SIZE * BLOCK_SIZE
            pieces = []
            if full_width:
                grouped = signed[..., :full_width].reshape(*value.shape[:-1], -1, BLOCK_SIZE)
                pieces.append((grouped @ self.full_block).reshape(*value.shape[:-1], full_width))
            if self.tail_block is not None:
                pieces.append(signed[..., full_width:] @ self.tail_block)
            return torch.cat(pieces, dim=-1)


@dataclass
class RotationPlan:
    mode: str
    seed: int
    transforms: dict[str, _BlockRotation]
    metadata: dict[str, dict]
    _transformed: weakref.WeakSet = field(default_factory=weakref.WeakSet, repr=False)

    def rotate_weight(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Apply the same feature transform to a [output, input] weight."""
        if name not in TARGETS:
            raise ValueError(f"Unknown rotation target: {name}")
        if value.ndim != 2 or value.shape[1] != EXPECTED_DIMENSIONS[TARGETS.index(name)]:
            raise ValueError(f"Unexpected projection weight shape at {name}")
        _check_fp32(value)
        return value if self.mode == "none" else self.transforms[name].apply(value)

    def transform_out_proj_weights_(self, model: nn.Module) -> None:
        """Mutate a fresh model once; subsequent weight QDQ sees W @ R."""
        if model in self._transformed:
            raise ValueError("Weights already transformed; load a fresh FP32 model")
        projections = _projections(model)
        replacements = {}
        for name, projection in projections.items():
            if self.mode == "none":
                continue
            with torch.no_grad():
                replacements[name] = self.rotate_weight(name, projection.weight)
        # Validate every target before changing any model parameter.
        with torch.no_grad():
            for name, weight in replacements.items():
                projections[name].weight.copy_(weight)
        self._transformed.add(model)

    def rotate_activation(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Rotate [batch, feature, time] once, before this boundary's A8 QDQ."""
        name = name.removesuffix(".scan_gate_output")
        if name not in TARGETS:
            raise ValueError(f"Unknown rotation boundary: {name}")
        if value.ndim != 3 or value.shape[1] != EXPECTED_DIMENSIONS[TARGETS.index(name)]:
            raise ValueError(f"Expected [B, D, T] at {name}, got {tuple(value.shape)}")
        _check_fp32(value)
        if self.mode in {"none", "identity"}:
            # Identity must preserve layout as well as values for the same GEMM path.
            return value
        rotated = self.transforms[name].apply(value.transpose(1, 2))
        return rotated.transpose(1, 2).contiguous()

    def to_dict(self) -> dict:
        return {"scheme": "P-D-H_block", "mode": self.mode, "seed": self.seed,
                "block_size": BLOCK_SIZE, "compute_dtype": "float32",
                "targets": list(TARGETS), "layers": self.metadata}


def prepare_rotations(model: nn.Module, *, seed: int = 42, block: int = BLOCK_SIZE,
                      mode: str = "outproj_h128") -> RotationPlan:
    if mode not in {"none", "identity", "outproj_h128"}:
        raise ValueError(f"Unsupported rotation mode: {mode}")
    if block != BLOCK_SIZE:
        raise ValueError("RotOut-H128 fixes the block size at 128")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Rotation seed must be a non-negative integer")
    projections = _projections(model)
    transforms, metadata = {}, {}
    for index, (name, projection) in enumerate(projections.items()):
        _, layer_metadata = _rotation(projection.in_features, seed + index,
                                      identity=mode == "identity")
        device = projection.weight.device
        permutation = torch.tensor(layer_metadata["permutation"], device=device)
        tail = projection.in_features % BLOCK_SIZE
        transforms[name] = _BlockRotation(
            inverse_permutation=permutation.argsort(),
            signs=torch.tensor(layer_metadata["signs"], dtype=torch.float32, device=device),
            full_block=_hadamard(BLOCK_SIZE).to(device),
            tail_block=_hadamard(tail).to(device) if tail else None,
            identity=mode == "identity",
        )
        metadata[name] = layer_metadata
    return RotationPlan(mode, seed, transforms, metadata)


def _fixed_basis_matrix(index: int) -> tuple[torch.Tensor, dict]:
    """Independent Kronecker construction of the frozen seed-42 P D H."""
    dimension = EXPECTED_DIMENSIONS[index]
    generator = torch.Generator(device="cpu").manual_seed(42 + index)
    permutation = torch.randperm(dimension, generator=generator)
    signs = torch.randint(0, 2, (dimension,), generator=generator).float() * 2 - 1
    h2 = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float32)
    blocks = []
    sizes = [128] * (dimension // 128) + ([4] if dimension % 128 else [])
    for size in sizes:
        block = torch.ones((1, 1), dtype=torch.float32)
        while block.shape[0] < size:
            block = torch.kron(block, h2)
        blocks.append(block / math.sqrt(size))
    hadamard = torch.block_diag(*blocks)
    # R[i,j] = D[perm[i],perm[i]] H[perm[i],j], from literal P @ D @ H.
    matrix = (signs[:, None] * hadamard).index_select(0, permutation)
    metadata = {"dimension": dimension, "seed": 42 + index, "blocks": sizes,
                "permutation": permutation.tolist(), "signs": signs.int().tolist(),
                "matrix_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest()}
    return matrix, metadata


def _basis_model(plan: RotationPlan) -> nn.Module:
    """Fresh identity projections exercise the real weight-mutation loop."""
    model = nn.Module()
    for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
        branch = model
        for part in name.split("."):
            if part not in branch._modules:
                branch.add_module(part, nn.Module())
            branch = branch.get_submodule(part)
        projection = nn.Linear(dimension, dimension, bias=False, device="meta", dtype=torch.float32)
        basis = torch.eye(dimension, device=plan.transforms[name].signs.device, dtype=torch.float32)
        projection.weight = nn.Parameter(basis, requires_grad=False)
        branch.add_module("out_proj", projection)
    return model


def _exact_basis_report(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    if actual.shape != expected.shape or actual.dtype != torch.float32:
        raise ValueError("Basis output shape or dtype differs from the fixed matrix")
    failures = int((actual.detach().cpu() != expected).sum())
    return {"exact_equal": failures == 0, "failure_count": failures,
            "elements": expected.numel()}


def verify_fixed_basis(plan: RotationPlan) -> dict:
    """Check every input basis vector through separate weight/activation APIs."""
    if plan.mode != "outproj_h128" or set(plan.transforms) != set(TARGETS):
        raise ValueError("Fixed-basis verification requires all five H128 targets")
    layers = {}
    with torch.inference_mode():
        model = _basis_model(plan)
        plan.transform_out_proj_weights_(model)
        for index, name in enumerate(TARGETS):
            dimension = EXPECTED_DIMENSIONS[index]
            basis = torch.eye(dimension, device=plan.transforms[name].signs.device, dtype=torch.float32)
            expected, metadata = _fixed_basis_matrix(index)
            activation = plan.rotate_activation(name, basis[:, :, None]).squeeze(-1)
            weight = model.get_submodule(name + ".out_proj").weight
            layers[name] = {"dimension": dimension, "metadata_matches": plan.metadata.get(name) == metadata,
                            "activation": _exact_basis_report(expected, activation),
                            "weight": _exact_basis_report(expected, weight)}
    passed = plan.seed == 42 and all(row["metadata_matches"] and row[path]["exact_equal"]
                                    for row in layers.values() for path in ("activation", "weight"))
    return {"protocol": BASIS_PROTOCOL, "allclose": passed, "mode": plan.mode,
            "seed": plan.seed, "layers": layers}


def require_fixed_basis(report: Mapping) -> None:
    if (report.get("protocol") != BASIS_PROTOCOL or report.get("seed") != 42
            or report.get("mode") != "outproj_h128" or set(report.get("layers", {})) != set(TARGETS)):
        raise ValueError("Incomplete or noncanonical fixed-basis report")
    for name, dimension in zip(TARGETS, EXPECTED_DIMENSIONS):
        layer = report["layers"][name]
        if layer.get("dimension") != dimension or layer.get("metadata_matches") is not True:
            raise ValueError("Fixed-basis dimension differs")
        for path in ("activation", "weight"):
            row = layer.get(path, {})
            if (row.get("exact_equal") is not True or row.get("failure_count") != 0
                    or row.get("elements") != dimension ** 2):
                raise RuntimeError(f"Fixed-basis check failed: {name} {path}")
    if report.get("allclose") is not True:
        raise RuntimeError("Fixed-basis aggregate failed")


def _control_plan(plan: RotationPlan) -> RotationPlan:
    mutated = copy(plan)
    mutated.transforms = {name: copy(transform) for name, transform in plan.transforms.items()}
    mutated._transformed = weakref.WeakSet()
    return mutated


def _transpose_activation(name: str, value: torch.Tensor) -> torch.Tensor:
    name = name.removesuffix(".scan_gate_output")
    matrix = _fixed_basis_matrix(TARGETS.index(name))[0].to(value.device)
    return (value.transpose(1, 2) @ matrix.T).transpose(1, 2)


def audit_rotation_negative_controls(plan: RotationPlan) -> dict:
    """Actually execute fixed mutations; never retry seeds or modify the model."""
    require_fixed_basis(verify_fixed_basis(plan))
    mutations = []
    for name in TARGETS[:-1]:
        mutated = _control_plan(plan)
        mutated.transforms[name].tail_block = torch.eye(4, device=plan.transforms[name].signs.device,
                                                       dtype=torch.float32)
        mutations.append(("h4_identity:" + name, "h4_identity", [name], mutated))
    transposed = _control_plan(plan)
    transposed.rotate_activation = _transpose_activation
    mutations.append(("activation_transpose", "activation_transpose", list(TARGETS), transposed))
    swapped = _control_plan(plan)
    for transform in swapped.transforms.values():
        transform.signs = transform.signs[transform.inverse_permutation]
    mutations.append(("swapped_pd_both", "swapped_pd_both", list(TARGETS), swapped))
    controls = {}
    for name, mutation, targets, mutated in mutations:
        report = verify_fixed_basis(mutated)
        failures = sum(row[path]["failure_count"] for row in report["layers"].values()
                       for path in ("activation", "weight"))
        controls[name] = {"mutation": mutation, "targets": targets,
                          "expected_rejection": "fixed_basis", "which_gate": "fixed_basis",
                          "failure_count": failures, "rejected": not report["allclose"] and failures > 0,
                          "gate_report": report}
    rejected = all(row["rejected"] for row in controls.values())
    return {"protocol": NEGATIVE_PROTOCOL, "all_rejected": rejected, "passed": rejected,
            "controls": controls}


def tensor_parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Report every tolerance failure; the second is compared to the reference."""
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("Parity tensors must have identical non-empty shapes")
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise TypeError("FP32 parity requires two FP32 tensors")
    reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    difference = (candidate - reference).abs()
    tolerance = PARITY_ATOL + PARITY_RTOL * reference.abs()
    passed = finite & (difference <= tolerance)
    failure_count = int((~passed).sum())
    finite_difference = difference[finite]
    ratios = torch.where(finite, difference / tolerance, torch.full_like(difference, float("inf")))
    worst_flat = int(ratios.reshape(-1).argmax())
    worst_index = list(torch.unravel_index(torch.tensor(worst_flat), reference.shape))
    def number(value: torch.Tensor) -> float | None:
        return float(value) if torch.isfinite(value) else None
    return {"allclose": failure_count == 0, "shape": list(reference.shape),
            "elements": reference.numel(), "failure_count": failure_count,
            "nonfinite_count": int((~finite).sum()), "rtol": PARITY_RTOL,
            "atol": PARITY_ATOL,
            "max_abs_diff": number(finite_difference.max()) if finite_difference.numel() else None,
            "mean_abs_diff": number(finite_difference.mean()) if finite_difference.numel() else None,
            "max_error_over_tolerance": number(ratios.max()),
            "worst_index": [int(index) for index in worst_index],
            "worst_reference": number(reference.reshape(-1)[worst_flat]),
            "worst_candidate": number(candidate.reshape(-1)[worst_flat])}


def vector_rms_parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Audit every [batch,time] vector; FP64 reduction never changes inference."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.numel() == 0:
        raise ValueError("Vector RMS parity requires identical non-empty [B,T,C] tensors")
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise TypeError("Vector RMS parity requires FP32 inference outputs")
    reference, candidate = reference.detach().cpu().double(), candidate.detach().cpu().double()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    reference_rms = reference.square().mean(dim=-1).sqrt()
    error_rms = (candidate - reference).square().mean(dim=-1).sqrt()
    passed = finite.all(dim=-1) & (error_rms <= PARITY_ATOL + PARITY_RTOL * reference_rms)
    failures = int((~passed).sum())
    return {"allclose": failures == 0, "shape": list(reference.shape),
            "vectors": reference.shape[0] * reference.shape[1], "elements": reference.numel(),
            "failure_count": failures, "nonfinite_count": int((~finite).sum()),
            "rtol": PARITY_RTOL, "atol": PARITY_ATOL}


def compare_parity(reference: Mapping[str, torch.Tensor],
                   candidate: Mapping[str, torch.Tensor], *, protocol: str = ELEMENTWISE_PROTOCOL) -> dict:
    if protocol not in {ELEMENTWISE_PROTOCOL, VECTOR_RMS_PROTOCOL}:
        raise ValueError(f"Unknown parity protocol: {protocol}")
    expected = {"logits", *TARGETS}
    if set(reference) != expected or set(candidate) != expected:
        raise ValueError("Parity requires logits and every one of the five out_proj outputs")
    outputs = {name: tensor_parity(reference[name], candidate[name])
               for name in ("logits", *TARGETS)}
    report = {"allclose": all(row["allclose"] for row in outputs.values()),
              "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "outputs": outputs}
    if protocol == VECTOR_RMS_PROTOCOL:
        vectors = {name: vector_rms_parity(reference[name], candidate[name]) for name in TARGETS}
        report.update(protocol=protocol, legacy_allclose=report["allclose"], vector_rms=vectors,
                      allclose=outputs["logits"]["allclose"] and all(row["allclose"] for row in vectors.values()))
    return report


def require_parity(report: Mapping) -> None:
    """Call after saving the report; a failed layer must block quantized runs."""
    if report.get("rtol") != PARITY_RTOL or report.get("atol") != PARITY_ATOL:
        raise ValueError("Parity tolerance differs from the frozen protocol")
    outputs = report.get("outputs", {})
    if set(outputs) != {"logits", *TARGETS}:
        raise ValueError("Incomplete parity report")
    protocol = report.get("protocol", ELEMENTWISE_PROTOCOL)
    if protocol not in {ELEMENTWISE_PROTOCOL, VECTOR_RMS_PROTOCOL}:
        raise ValueError("Unknown parity protocol")
    checked = outputs
    if protocol == VECTOR_RMS_PROTOCOL:
        vectors = report.get("vector_rms", {})
        if set(vectors) != set(TARGETS):
            raise ValueError("Incomplete vector RMS report")
        if report.get("legacy_allclose") is not all(row.get("allclose") for row in outputs.values()):
            raise ValueError("Legacy parity summary differs from retained outputs")
        for name, row in vectors.items():
            shape = row.get("shape", [])
            if (len(shape) != 3 or any(d <= 0 for d in shape)
                    or shape != outputs[name].get("shape")
                    or row.get("vectors") != shape[0] * shape[1]
                    or row.get("elements") != math.prod(shape)
                    or row.get("rtol") != PARITY_RTOL or row.get("atol") != PARITY_ATOL):
                raise ValueError(f"Invalid vector RMS report: {name}")
        checked = {"logits": outputs["logits"], **vectors}
    failed = [name for name, row in checked.items()
              if not row.get("allclose") or row.get("failure_count") != 0
              or row.get("nonfinite_count") != 0]
    if failed or not report.get("allclose"):
        raise RuntimeError(f"Rotation parity failed: {', '.join(failed)}")
