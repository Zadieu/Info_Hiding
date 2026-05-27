from __future__ import annotations

import math

import torch
from torch import Tensor

from hidden_repro.noise_layers.base import BaseNoiseLayer


class CropNoise(BaseNoiseLayer):
    """Random square crop that keeps an area ratio p of the encoded image."""

    def __init__(self, p: float = 0.035) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)
        super().__init__(name=f"crop_p_{self.p:g}")

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        del cover
        _, _, height, width = encoded.shape
        crop_size = max(1, int(round(height * math.sqrt(self.p))))
        crop_size = min(crop_size, height, width)

        max_x = width - crop_size
        max_y = height - crop_size
        x1 = int(torch.randint(0, max_x + 1, (1,), device=encoded.device).item())
        y1 = int(torch.randint(0, max_y + 1, (1,), device=encoded.device).item())

        return encoded[:, :, y1 : y1 + crop_size, x1 : x1 + crop_size]
