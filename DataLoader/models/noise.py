"""Noise layer N (parameterless) — Sec. 3."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseLayer(nn.Module):
    def __init__(
        self,
        noise_type: str = "identity",
        dropout_p: float = 0.3,
        cropout_p: float = 0.3,
        gaussian_sigma: float = 2.0,
        crop_p: float = 0.035,
    ):
        super().__init__()
        self.noise_type = noise_type
        self.dropout_p = dropout_p
        self.cropout_p = cropout_p
        self.gaussian_sigma = gaussian_sigma
        self.crop_p = crop_p

    def forward(self, cover: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "identity":
            return encoded
        if self.noise_type == "dropout":
            return self._dropout(cover, encoded, self.dropout_p)
        if self.noise_type == "cropout":
            return self._cropout(cover, encoded, self.cropout_p)
        if self.noise_type == "gaussian":
            return self._gaussian(encoded, self.gaussian_sigma)
        if self.noise_type == "crop":
            return self._crop(encoded, self.crop_p)
        raise ValueError(f"Unknown noise type: {self.noise_type}")

    @staticmethod
    def _dropout(cover: torch.Tensor, encoded: torch.Tensor, p: float) -> torch.Tensor:
        mask = (torch.rand_like(encoded) < p).float()
        return encoded * mask + cover * (1.0 - mask)

    def _cropout(self, cover: torch.Tensor, encoded: torch.Tensor, p: float) -> torch.Tensor:
        b, c, h, w = encoded.shape
        side = max(1, int((p ** 0.5) * min(h, w)))
        out = cover.clone()
        for i in range(b):
            top = torch.randint(0, h - side + 1, (1,)).item()
            left = torch.randint(0, w - side + 1, (1,)).item()
            out[i, :, top : top + side, left : left + side] = encoded[
                i, :, top : top + side, left : left + side
            ]
        return out

    @staticmethod
    def _gaussian(encoded: torch.Tensor, sigma: float) -> torch.Tensor:
        k = int(4 * sigma + 1) | 1
        x = torch.arange(k, device=encoded.device, dtype=encoded.dtype) - k // 2
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).view(1, 1, k, k)
        kernel = kernel.expand(encoded.size(1), 1, k, k)
        pad = k // 2
        return F.conv2d(encoded, kernel, padding=pad, groups=encoded.size(1))

    @staticmethod
    def _crop(encoded: torch.Tensor, p: float) -> torch.Tensor:
        b, c, h, w = encoded.shape
        h2 = max(1, int(h * (p**0.5)))
        w2 = max(1, int(w * (p**0.5)))
        out = []
        for i in range(b):
            top = torch.randint(0, h - h2 + 1, (1,)).item()
            left = torch.randint(0, w - w2 + 1, (1,)).item()
            out.append(encoded[i : i + 1, :, top : top + h2, left : left + w2])
        return torch.cat(out, dim=0)
