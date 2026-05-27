from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from hidden_repro.noise_layers.base import BaseNoiseLayer


class GaussianBlurNoise(BaseNoiseLayer):
    """Differentiable Gaussian blur used in the HiDDeN paper."""

    def __init__(self, sigma: float = 2.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be greater than 0")
        self.sigma = float(sigma)
        self.kernel_size = 2 * int(4 * self.sigma) + 1
        super().__init__(name=f"gaussian_sigma_{self.sigma:g}")
        kernel = self._build_kernel(self.kernel_size, self.sigma)
        self.register_buffer("kernel", kernel, persistent=False)

    @staticmethod
    def _build_kernel(kernel_size: int, sigma: float) -> Tensor:
        center = (kernel_size - 1) / 2.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - center
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2.0 * sigma * sigma))
        kernel = kernel / (2.0 * math.pi * sigma * sigma)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        del cover
        channels = encoded.shape[1]
        kernel = self.kernel.to(device=encoded.device, dtype=encoded.dtype).expand(
            channels, 1, self.kernel_size, self.kernel_size
        )
        padding = self.kernel_size // 2
        padded = F.pad(encoded, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(padded, kernel, groups=channels)
