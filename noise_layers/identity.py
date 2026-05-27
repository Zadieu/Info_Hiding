from __future__ import annotations

from torch import Tensor

from hidden_repro.noise_layers.base import BaseNoiseLayer


class IdentityNoise(BaseNoiseLayer):
    """Identity layer from HiDDeN: I_no = I_en."""

    def __init__(self) -> None:
        super().__init__(name="identity")

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        del cover
        return encoded
