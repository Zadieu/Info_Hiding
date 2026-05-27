from __future__ import annotations

import torch
from torch import Tensor

from hidden_repro.noise_layers.base import BaseNoiseLayer


class DropoutNoise(BaseNoiseLayer):
    """Keep encoded pixels with probability p and replace the rest with cover pixels."""

    def __init__(self, p: float = 0.3) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)
        super().__init__(name=f"dropout_p_{self.p:g}")

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        batch_size, channels, height, width = encoded.shape
        mask = torch.rand(
            batch_size,
            1,
            height,
            width,
            device=encoded.device,
            dtype=encoded.dtype,
        ) < self.p
        mask = mask.expand(batch_size, channels, height, width)
        return torch.where(mask, encoded, cover)
