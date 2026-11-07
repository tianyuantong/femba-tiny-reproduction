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
"""H128 Hadamard rotation of the second Mamba block's scan outputs, folded into out_proj.

Layout and checks follow the 2026-09-22 rotation protocol of the TUAR archive
(reproduction/PROTOCOL.md): R = P D H with a seeded random permutation P, random signs D and
a block-diagonal Hadamard H of twelve 128-blocks plus one 4-block (1540 = 12 x 128 + 4). Each
target's seed is 42 plus its index in that protocol's target list, so block 1 forward and
reverse keep 44 and 45. Before its activation quantizer the gated scan output y [B, L, 1540]
becomes y @ R, and out_proj's weight W [385, 1540] becomes W @ R once, so the float model is
unchanged: (y R)(W R)^T = y W^T. The rotation itself always runs in FP32 with TF32 off.
"""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
import hashlib
import math
import weakref

import torch
from torch import nn

TARGETS = ("mamba_blocks.1.mamba_fwd", "mamba_blocks.1.mamba_rev")
TARGET_SEEDS = {"mamba_blocks.1.mamba_fwd": 44, "mamba_blocks.1.mamba_rev": 45}  # 42 + protocol index
DIMENSION = 1540
BLOCK_SIZE = 128
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5
BASIS_PROTOCOL = "fixed-pdh-basis-v1"
NEGATIVE_PROTOCOL = "fixed-pdh-negative-v1"
VECTOR_RMS_PROTOCOL = "vector-rms-2026-09-22"


@contextmanager
def strict_fp32_matmul():
    """Run matmuls at true FP32 even when the formal runtime enables TF32."""
    prior = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior


def hadamard(order: int) -> torch.Tensor:
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(order)


def block_sizes(dimension: int) -> list[int]:
    sizes = [BLOCK_SIZE] * (dimension // BLOCK_SIZE)
    if dimension % BLOCK_SIZE:
        sizes.append(dimension % BLOCK_SIZE)
    return sizes


def rotation_matrix(dimension: int, seed: int) -> tuple[torch.Tensor, dict]:
    """The protocol's P @ D @ H as a dense FP32 matrix, with the drawn permutation and signs."""
    blocks = block_sizes(dimension)
    stacked = torch.block_diag(*(hadamard(size) for size in blocks))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(dimension, generator=generator)
    signs = torch.randint(0, 2, (dimension,), generator=generator).float() * 2 - 1
    # P = I[permutation]; selecting signed rows is exactly P @ D @ H.
    matrix = (stacked[permutation] * signs[permutation, None]).contiguous()
    metadata = {"dimension": dimension, "seed": seed, "blocks": blocks,
                "permutation": permutation.tolist(), "signs": signs.int().tolist(),
                "matrix_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest()}
    return matrix, metadata


def fixed_basis_matrix(dimension: int, seed: int) -> tuple[torch.Tensor, dict]:
    """Independent Kronecker construction of the same P D H, for exact comparison."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(dimension, generator=generator)
    signs = torch.randint(0, 2, (dimension,), generator=generator).float() * 2 - 1
    h2 = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float32)
    blocks = []
    for size in block_sizes(dimension):
        block = torch.ones((1, 1), dtype=torch.float32)
        while block.shape[0] < size:
            block = torch.kron(block, h2)
        blocks.append(block / math.sqrt(size))
    stacked = torch.block_diag(*blocks)
    # R[i, j] = D[perm[i], perm[i]] H[perm[i], j], from literal P @ D @ H.
    matrix = (signs[:, None] * stacked).index_select(0, permutation)
    metadata = {"dimension": dimension, "seed": seed, "blocks": block_sizes(dimension),
                "permutation": permutation.tolist(), "signs": signs.int().tolist(),
                "matrix_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest()}
    return matrix, metadata


@dataclass
class BlockRotation:
    inverse_permutation: torch.Tensor
    signs: torch.Tensor
    full_block: torch.Tensor
    tail_block: torch.Tensor | None
    identity: bool

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        """value @ R along the last axis, in FP32 with TF32 off."""
        if value.dtype != torch.float32:
            raise TypeError(f"Rotation requires FP32, got {value.dtype}")
        if value.device != self.signs.device:
            raise ValueError("Prepare the rotation on the model's final device")
        if value.shape[-1] != self.inverse_permutation.numel():
            raise ValueError(f"Expected {self.inverse_permutation.numel()} features, got {value.shape[-1]}")
        if self.identity:
            return value
        with strict_fp32_matmul(), torch.autocast(value.device.type, enabled=False):
            # For P = I[perm], x @ P selects x[inverse_perm]; D then signs the columns.
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
    mode: str  # "identity" keeps values and layout; "h128" rotates
    transforms: dict[str, BlockRotation]
    metadata: dict[str, dict]
    _folded: weakref.WeakSet = field(default_factory=weakref.WeakSet, repr=False)

    def rotate_activation(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """y @ R for the [B, L, 1540] input of this target's scan-output quantizer."""
        if name not in self.transforms:
            raise ValueError(f"Unknown rotation target: {name}")
        if value.ndim != 3:
            raise ValueError(f"Expected [B, L, D] at {name}, got {tuple(value.shape)}")
        return self.transforms[name].apply(value)

    def rotate_weight(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        """W @ R for this target's [out, in] out_proj weight."""
        if name not in self.transforms:
            raise ValueError(f"Unknown rotation target: {name}")
        if weight.ndim != 2:
            raise ValueError(f"Expected a [out, in] weight at {name}, got {tuple(weight.shape)}")
        return self.transforms[name].apply(weight)

    def matrix(self, name: str) -> torch.Tensor:
        """The dense R actually applied, read off the identity through the activation path."""
        basis = torch.eye(self.transforms[name].inverse_permutation.numel(),
                          device=self.transforms[name].signs.device, dtype=torch.float32)
        return self.rotate_activation(name, basis[None]).squeeze(0)

    def projections(self, model: nn.Module) -> dict[str, nn.Linear]:
        modules = dict(model.named_modules())
        projections = {}
        for name in self.transforms:
            projection = modules.get(name + ".out_proj")
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"Missing Linear {name}.out_proj")
            if projection.weight.shape[1] != self.transforms[name].inverse_permutation.numel():
                raise ValueError(f"Unexpected {name}.out_proj input width {projection.weight.shape[1]}")
            if projection.weight.dtype != torch.float32:
                raise TypeError(f"{name}.out_proj must hold FP32 weights")
            projections[name] = projection
        return projections

    def fold_into_model_(self, model: nn.Module) -> None:
        """Replace each target's out_proj weight by W @ R, once per model instance."""
        if model in self._folded:
            raise ValueError("Weights already rotated; load a fresh FP32 model")
        projections = self.projections(model)
        with torch.no_grad():
            rotated = {name: self.rotate_weight(name, projection.weight) for name, projection in projections.items()}
            for name, weight in rotated.items():  # validate every target before changing any weight
                projections[name].weight.copy_(weight)
        self._folded.add(model)

    def install(self, model: nn.Module) -> list:
        """Rotate the input of each target's scan-output quantizer; returns the hook handles."""
        modules = dict(model.named_modules())
        handles = []
        for name in self.transforms:
            quantizer = modules.get(name + ".ssm.output_quant")
            if quantizer is None:
                raise ValueError(f"Missing {name}.ssm.output_quant")
            handles.append(quantizer.register_forward_pre_hook(
                lambda module, inputs, name=name: (self.rotate_activation(name, inputs[0]), *inputs[1:])))
        return handles

    def to_dict(self) -> dict:
        return {"scheme": "P-D-H_block", "mode": self.mode, "block_size": BLOCK_SIZE, "compute_dtype": "float32",
                "tf32": False, "targets": list(self.transforms), "layers": self.metadata}


def prepare_rotation(model: nn.Module, mode: str = "h128", targets: tuple = TARGETS) -> RotationPlan:
    if mode not in {"identity", "h128"}:
        raise ValueError(f"Unsupported rotation mode: {mode}")
    modules = dict(model.named_modules())
    transforms, metadata = {}, {}
    for name in targets:
        projection = modules.get(name + ".out_proj")
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"Missing Linear {name}.out_proj")
        dimension, device = projection.weight.shape[1], projection.weight.device
        _, layer = rotation_matrix(dimension, TARGET_SEEDS[name])
        if mode == "identity":
            layer = {**layer, "permutation": list(range(dimension)), "signs": [1] * dimension,
                     "matrix_sha256": hashlib.sha256(torch.eye(dimension).numpy().tobytes()).hexdigest()}
        permutation = torch.tensor(layer["permutation"], device=device)
        tail = dimension % BLOCK_SIZE
        transforms[name] = BlockRotation(
            inverse_permutation=permutation.argsort(),
            signs=torch.tensor(layer["signs"], dtype=torch.float32, device=device),
            full_block=hadamard(BLOCK_SIZE).to(device),
            tail_block=hadamard(tail).to(device) if tail else None,
            identity=mode == "identity")
        metadata[name] = layer
    return RotationPlan(mode, transforms, metadata)


def _basis_model(plan: RotationPlan) -> nn.Module:
    """Fresh identity projections exercise the real weight-folding loop."""
    model = nn.Module()
    for name, transform in plan.transforms.items():
        branch = model
        for part in name.split("."):
            if part not in branch._modules:
                branch.add_module(part, nn.Module())
            branch = branch.get_submodule(part)
        dimension = transform.inverse_permutation.numel()
        projection = nn.Linear(dimension, dimension, bias=False, device="meta", dtype=torch.float32)
        projection.weight = nn.Parameter(torch.eye(dimension, device=transform.signs.device), requires_grad=False)
        branch.add_module("out_proj", projection)
    return model


def _exact_report(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    if actual.shape != expected.shape or actual.dtype != torch.float32:
        raise ValueError("Basis output shape or dtype differs from the fixed matrix")
    failures = int((actual.detach().cpu() != expected).sum())
    return {"exact_equal": failures == 0, "failure_count": failures, "elements": expected.numel()}


def verify_fixed_basis(plan: RotationPlan) -> dict:
    """Push every unit vector through the activation and the weight path; compare exactly."""
    if plan.mode != "h128":
        raise ValueError("Fixed-basis verification requires the H128 plan")
    layers = {}
    with torch.inference_mode():
        model = _basis_model(plan)
        plan.fold_into_model_(model)
        for name, transform in plan.transforms.items():
            dimension = transform.inverse_permutation.numel()
            expected, metadata = fixed_basis_matrix(dimension, TARGET_SEEDS[name])
            basis = torch.eye(dimension, device=transform.signs.device, dtype=torch.float32)
            activation = plan.rotate_activation(name, basis[None]).squeeze(0)
            weight = model.get_submodule(name + ".out_proj").weight
            layers[name] = {"dimension": dimension, "seed": TARGET_SEEDS[name], "blocks": metadata["blocks"],
                            "metadata_matches": plan.metadata.get(name) == metadata,
                            "activation": _exact_report(expected, activation), "weight": _exact_report(expected, weight)}
    passed = all(row["metadata_matches"] and row[path]["exact_equal"] for row in layers.values()
                 for path in ("activation", "weight"))
    return {"protocol": BASIS_PROTOCOL, "allclose": passed, "mode": plan.mode, "layers": layers}


def require_fixed_basis(report: dict) -> None:
    if report.get("protocol") != BASIS_PROTOCOL or set(report.get("layers", {})) != set(TARGETS):
        raise ValueError("Incomplete or noncanonical fixed-basis report")
    for name, layer in report["layers"].items():
        if layer.get("dimension") != DIMENSION or layer.get("metadata_matches") is not True:
            raise RuntimeError(f"Fixed-basis metadata differs: {name}")
        for path in ("activation", "weight"):
            row = layer.get(path, {})
            if row.get("exact_equal") is not True or row.get("failure_count") != 0 or row.get("elements") != DIMENSION ** 2:
                raise RuntimeError(f"Fixed-basis check failed: {name} {path}")
    if report.get("allclose") is not True:
        raise RuntimeError("Fixed-basis aggregate failed")


def _control_plan(plan: RotationPlan) -> RotationPlan:
    mutated = copy(plan)
    mutated.transforms = {name: copy(transform) for name, transform in plan.transforms.items()}
    mutated._folded = weakref.WeakSet()
    return mutated


class _TransposedPlan(RotationPlan):
    """Activations through R^T instead of R; the weight path is untouched."""

    def rotate_activation(self, name: str, value: torch.Tensor) -> torch.Tensor:
        matrix = fixed_basis_matrix(value.shape[-1], TARGET_SEEDS[name])[0].to(value.device)
        with strict_fp32_matmul():
            return value @ matrix.T


def negative_controls(plan: RotationPlan) -> dict:
    """Execute fixed mutations that the basis check must reject; never retries or alters the plan."""
    require_fixed_basis(verify_fixed_basis(plan))
    mutations = []
    for name in plan.transforms:
        mutated = _control_plan(plan)
        mutated.transforms[name].tail_block = torch.eye(DIMENSION % BLOCK_SIZE, device=plan.transforms[name].signs.device)
        mutations.append(("h4_identity:" + name, "h4_identity", [name], mutated))
    transposed = _TransposedPlan(plan.mode, {name: copy(t) for name, t in plan.transforms.items()}, plan.metadata)
    mutations.append(("activation_transpose", "activation_transpose", list(plan.transforms), transposed))
    swapped = _control_plan(plan)
    for transform in swapped.transforms.values():
        transform.signs = transform.signs[transform.inverse_permutation]
    mutations.append(("swapped_pd_both", "swapped_pd_both", list(plan.transforms), swapped))
    controls = {}
    for key, mutation, targets, mutated in mutations:
        report = verify_fixed_basis(mutated)
        failures = sum(row[path]["failure_count"] for row in report["layers"].values() for path in ("activation", "weight"))
        controls[key] = {"mutation": mutation, "targets": targets, "which_gate": "fixed_basis",
                         "failure_count": failures, "rejected": not report["allclose"] and failures > 0}
    rejected = all(row["rejected"] for row in controls.values())
    return {"protocol": NEGATIVE_PROTOCOL, "all_rejected": rejected, "passed": rejected, "controls": controls}


def tensor_parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Elementwise |candidate - reference| <= atol + rtol |reference|, every failure counted."""
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("Parity tensors must have identical non-empty shapes")
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise TypeError("FP32 parity requires two FP32 tensors")
    reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    difference = (candidate - reference).abs()
    tolerance = PARITY_ATOL + PARITY_RTOL * reference.abs()
    passed = finite & (difference <= tolerance)
    ratios = torch.where(finite, difference / tolerance, torch.full_like(difference, float("inf")))
    worst = int(ratios.reshape(-1).argmax())

    def number(value: torch.Tensor):
        return float(value) if torch.isfinite(value) else None
    finite_difference = difference[finite]
    return {"allclose": bool(passed.all()), "shape": list(reference.shape), "elements": reference.numel(),
            "failure_count": int((~passed).sum()), "nonfinite_count": int((~finite).sum()),
            "rtol": PARITY_RTOL, "atol": PARITY_ATOL,
            "max_abs_diff": number(finite_difference.max()) if finite_difference.numel() else None,
            "mean_abs_diff": number(finite_difference.mean()) if finite_difference.numel() else None,
            "max_error_over_tolerance": number(ratios.max()),
            "worst_index": [int(index) for index in torch.unravel_index(torch.tensor(worst), reference.shape)],
            "worst_reference": number(reference.reshape(-1)[worst]), "worst_candidate": number(candidate.reshape(-1)[worst])}


def vector_rms_parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Every [batch, time] feature vector: RMS(error) <= atol + rtol RMS(reference), reduced in FP64."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.numel() == 0:
        raise ValueError("Vector RMS parity requires identical non-empty [B, T, C] tensors")
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32:
        raise TypeError("Vector RMS parity requires FP32 inference outputs")
    reference, candidate = reference.detach().cpu().double(), candidate.detach().cpu().double()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    reference_rms = reference.square().mean(dim=-1).sqrt()
    error_rms = (candidate - reference).square().mean(dim=-1).sqrt()
    ratio = error_rms / (PARITY_ATOL + PARITY_RTOL * reference_rms)
    passed = finite.all(dim=-1) & (ratio <= 1)
    return {"allclose": bool(passed.all()), "shape": list(reference.shape), "vectors": int(passed.numel()),
            "elements": reference.numel(), "failure_count": int((~passed).sum()),
            "nonfinite_count": int((~finite).sum()), "rtol": PARITY_RTOL, "atol": PARITY_ATOL,
            "max_rms_ratio": float(ratio.max())}


def compare_parity(reference: dict, candidate: dict, targets: tuple = TARGETS) -> dict:
    """Logits elementwise; each target's out_proj output per vector (and elementwise, for the record)."""
    expected = {"logits", *targets}
    if set(reference) != expected or set(candidate) != expected:
        raise ValueError("Parity requires logits and every target's out_proj output")
    outputs = {name: tensor_parity(reference[name], candidate[name]) for name in ("logits", *targets)}
    vectors = {name: vector_rms_parity(reference[name], candidate[name]) for name in targets}
    return {"protocol": VECTOR_RMS_PROTOCOL, "rtol": PARITY_RTOL, "atol": PARITY_ATOL,
            "allclose": outputs["logits"]["allclose"] and all(row["allclose"] for row in vectors.values()),
            "legacy_allclose": all(row["allclose"] for row in outputs.values()),
            "outputs": outputs, "vector_rms": vectors}


def require_parity(report: dict) -> None:
    if report.get("protocol") != VECTOR_RMS_PROTOCOL or report.get("rtol") != PARITY_RTOL or report.get("atol") != PARITY_ATOL:
        raise ValueError("Parity report protocol or tolerance differs from the frozen one")
    checked = {"logits": report["outputs"]["logits"], **report["vector_rms"]}
    failed = [name for name, row in checked.items()
              if not row.get("allclose") or row.get("failure_count") != 0 or row.get("nonfinite_count") != 0]
    if failed or not report.get("allclose"):
        raise RuntimeError(f"Rotation parity failed: {', '.join(failed)}")
