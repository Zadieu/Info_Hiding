from __future__ import annotations

import math

import torch
from torch import Tensor

from hidden_repro.noise_layers.base import BaseNoiseLayer


class CropoutNoise(BaseNoiseLayer):
    """Keep a random square crop from the encoded image and use cover elsewhere."""

    def __init__(self, p: float = 0.3) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)
        super().__init__(name=f"cropout_p_{self.p:g}")

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        _, _, height, width = encoded.shape
        crop_size = max(1, int(round(height * math.sqrt(self.p))))
        crop_size = min(crop_size, height, width)

        max_x = width - crop_size
        max_y = height - crop_size
        x1 = int(torch.randint(0, max_x + 1, (1,), device=encoded.device).item())
        y1 = int(torch.randint(0, max_y + 1, (1,), device=encoded.device).item())

        mask = torch.zeros_like(encoded)
        mask[:, :, y1 : y1 + crop_size, x1 : x1 + crop_size] = 1.0
        return encoded * mask + cover * (1.0 - mask)
