from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class BaseNoiseLayer(nn.Module):
    """Abstract base class for HiDDeN noise layers."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, cover: Tensor, encoded: Tensor) -> Tensor:
        self._validate_inputs(cover, encoded)
        return self.apply_noise(cover=cover, encoded=encoded)

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement apply_noise().")

    @staticmethod
    def _validate_inputs(cover: Tensor, encoded: Tensor) -> None:
        if not isinstance(cover, torch.Tensor) or not isinstance(encoded, torch.Tensor):
            raise TypeError("cover and encoded must both be torch.Tensor")
        if cover.ndim != 4 or encoded.ndim != 4:
            raise ValueError("cover and encoded must both have shape [B, C, H, W]")
        if cover.shape != encoded.shape:
            raise ValueError("cover and encoded must have the same shape")
        if cover.device != encoded.device:
            raise ValueError("cover and encoded must be on the same device")
