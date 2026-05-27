from __future__ import annotations

from collections.abc import Iterable
import random

import torch.nn as nn

from torch import Tensor

from hidden_repro.noise_layers.base import BaseNoiseLayer
from hidden_repro.noise_layers.crop import CropNoise
from hidden_repro.noise_layers.cropout import CropoutNoise
from hidden_repro.noise_layers.dropout import DropoutNoise
from hidden_repro.noise_layers.gaussian import GaussianBlurNoise
from hidden_repro.noise_layers.identity import IdentityNoise
from hidden_repro.noise_layers.jpeg import JpegDropNoise, JpegMaskNoise


class NoiseManager(nn.Module):
    """Randomly picks one configured noise layer for each forward pass."""

    def __init__(self, noise_layers: Iterable[BaseNoiseLayer] | None = None) -> None:
        super().__init__()
        layers = list(noise_layers) if noise_layers is not None else [IdentityNoise()]
        if not layers:
            layers = [IdentityNoise()]
        self.noise_layers = nn.ModuleList(layers)

    def forward(self, cover: Tensor, encoded: Tensor) -> Tensor:
        noise_layer = random.choice(list(self.noise_layers))
        return noise_layer(cover=cover, encoded=encoded)


def build_noise_layer(name: str, **kwargs: float) -> BaseNoiseLayer:
    registry: dict[str, type[BaseNoiseLayer]] = {
        "identity": IdentityNoise,
        "gaussian": GaussianBlurNoise,
        "dropout": DropoutNoise,
        "crop": CropNoise,
        "cropout": CropoutNoise,
        "jpeg_mask": JpegMaskNoise,
        "jpeg_drop": JpegDropNoise,
    }
    normalized_name = name.strip().lower()
    if normalized_name not in registry:
        raise ValueError(f"Unsupported noise layer: {name}")
    return registry[normalized_name](**kwargs)


def build_paper_noise_layers() -> list[BaseNoiseLayer]:
    return [
        IdentityNoise(),
        DropoutNoise(p=0.3),
        DropoutNoise(p=0.7),
        CropoutNoise(p=0.3),
        CropoutNoise(p=0.7),
        CropNoise(p=0.035),
        CropNoise(p=0.3),
        GaussianBlurNoise(sigma=2.0),
        GaussianBlurNoise(sigma=4.0),
        JpegMaskNoise(),
        JpegDropNoise(),
    ]
