"""Fixed FP32 output-projection rotation and non-negotiable parity checks.

Call ``prepare_rotations`` after placing a fresh FP32 model on its device,
transform its weights once, and only then apply weight QDQ. The explicit
Mamba path must call ``rotate_activation`` BEFORE the scan-output A8 QDQ.
This module does not change scan, gates, normalization, or model outputs.
"""

from __future__ import annotations

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
                replacements[name] = self.transforms[name].apply(projection.weight)
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


def compare_parity(reference: Mapping[str, torch.Tensor],
                   candidate: Mapping[str, torch.Tensor]) -> dict:
    expected = {"logits", *TARGETS}
    if set(reference) != expected or set(candidate) != expected:
        raise ValueError("Parity requires logits and every one of the five out_proj outputs")
    outputs = {name: tensor_parity(reference[name], candidate[name])
               for name in ("logits", *TARGETS)}
    return {"allclose": all(row["allclose"] for row in outputs.values()),
            "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "outputs": outputs}


def require_parity(report: Mapping) -> None:
    """Call after saving the report; a failed layer must block quantized runs."""
    if report.get("rtol") != PARITY_RTOL or report.get("atol") != PARITY_ATOL:
        raise ValueError("Parity tolerance differs from the frozen protocol")
    outputs = report.get("outputs", {})
    if set(outputs) != {"logits", *TARGETS}:
        raise ValueError("Incomplete parity report")
    failed = [name for name, row in outputs.items()
              if not row.get("allclose") or row.get("failure_count") != 0
              or row.get("nonfinite_count") != 0]
    if failed or not report.get("allclose"):
        raise RuntimeError(f"Rotation parity failed: {', '.join(failed)}")
