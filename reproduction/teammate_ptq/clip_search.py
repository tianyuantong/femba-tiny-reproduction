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
"""Per-batch activation clipping thresholds chosen by quantization MSE or KL divergence.

Takes the place of a Brevitas runtime-stats percentile observer: forward returns the
clipping threshold for one calibration batch, and Brevitas averages the per-batch
thresholds over the calibration set exactly as it does for percentiles.

MSE simulates the quantizer itself; its candidates are log-spaced from 1e-8 of the batch
maximum, because a few activations exceed the 90th percentile by more than 10^5, and then
refined linearly. KL follows TensorRT's entropy calibration on a fixed histogram of
|x| over [0, max] with 128 bins per integer level, so candidate thresholds are the
multiples of 1/128 of the batch maximum.
"""
import math

import torch

MAX_SAMPLES = 1 << 22    # larger tensors are subsampled with a fixed stride
MSE_GRID = 128           # log-spaced MSE candidates over MSE_MIN_FRACTION .. 1 of the batch maximum
MSE_REFINE = 32          # linear MSE candidates between the neighbours of the best grid point
MSE_MIN_FRACTION = 1e-8
KL_STEPS = 128           # KL candidates: thresholds k/KL_STEPS of the batch maximum, k = 1 .. KL_STEPS


class ClipSearch(torch.nn.Module):
    def __init__(self, method: str, min_int: int, max_int: int, int_scaling: float):
        super().__init__()
        if method not in ("mse", "kl"):
            raise ValueError(f"Unknown clipping method: {method}")
        self.method, self.min_int, self.max_int, self.int_scaling = method, min_int, max_int, int_scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().reshape(-1).float()
        x = x[::math.ceil(x.numel() / MAX_SAMPLES)]
        peak = x.abs().max()
        if peak == 0:
            return peak
        return self.mse_threshold(x, peak) if self.method == "mse" else self.kl_threshold(x, peak)

    def quantization_mse(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        scale = threshold / self.int_scaling
        quantized = torch.clamp(torch.round(x / scale), self.min_int, self.max_int) * scale
        return torch.mean((x - quantized) ** 2)

    def mse_threshold(self, x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
        grid = peak * torch.logspace(math.log10(MSE_MIN_FRACTION), 0, MSE_GRID, device=x.device)
        best = int(torch.stack([self.quantization_mse(x, t) for t in grid]).argmin())
        low, high = grid[max(best - 1, 0)].item(), grid[min(best + 1, MSE_GRID - 1)].item()
        fine = torch.linspace(low, high, MSE_REFINE, device=x.device)
        return fine[int(torch.stack([self.quantization_mse(x, t) for t in fine]).argmin())]

    def kl_threshold(self, x: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
        # For each candidate: P is the histogram up to the threshold with the clipped mass added
        # to its last bin; Q merges the same bins, without the clipped mass, into the integer
        # levels and spreads each level evenly over P's nonzero bins.
        levels = round(self.int_scaling)
        hist = torch.histc(x.abs(), bins=levels * KL_STEPS, min=0, max=peak.item())
        divergences = []
        for k in range(1, KL_STEPS + 1):
            inside = hist[:k * levels]
            p = inside.clone()
            p[-1] += hist[k * levels:].sum()
            nonzero = (p > 0).float().view(levels, k)
            level_mass = inside.view(levels, k).sum(dim=1, keepdim=True)
            q = (level_mass / nonzero.sum(dim=1, keepdim=True).clamp(min=1) * nonzero).view(-1)
            p, q = p / p.sum(), q / q.sum().clamp(min=1)
            mask = p > 0
            divergences.append(torch.sum(p[mask] * torch.log(p[mask] / q[mask].clamp(min=1e-12))))
        return peak * (int(torch.stack(divergences).argmin()) + 1) / KL_STEPS
