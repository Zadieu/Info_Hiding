from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from hidden_repro.noise_layers.base import BaseNoiseLayer


def _build_ortho_dct_matrix(block_size: int = 8) -> Tensor:
    matrix = torch.empty(block_size, block_size, dtype=torch.float32)
    scale0 = math.sqrt(1.0 / block_size)
    scale = math.sqrt(2.0 / block_size)
    for k in range(block_size):
        alpha = scale0 if k == 0 else scale
        for n in range(block_size):
            matrix[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * block_size))
    return matrix


def _zigzag_indices(block_size: int = 8) -> list[tuple[int, int]]:
    return sorted(
        ((row, col) for row in range(block_size) for col in range(block_size)),
        key=lambda item: (item[0] + item[1], -item[1] if (item[0] + item[1]) % 2 else item[1]),
    )


def _build_keep_mask(keep_count: int, block_size: int = 8) -> Tensor:
    mask = torch.zeros(block_size, block_size, dtype=torch.float32)
    for row, col in _zigzag_indices(block_size)[:keep_count]:
        mask[row, col] = 1.0
    return mask


def _rgb_to_yuv(image: Tensor) -> Tensor:
    if image.shape[1] != 3:
        return image
    yuv = torch.empty_like(image)
    yuv[:, 0] = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    yuv[:, 1] = -0.14713 * image[:, 0] - 0.28886 * image[:, 1] + 0.436 * image[:, 2]
    yuv[:, 2] = 0.615 * image[:, 0] - 0.51499 * image[:, 1] - 0.10001 * image[:, 2]
    return yuv


def _yuv_to_rgb(image: Tensor) -> Tensor:
    if image.shape[1] != 3:
        return image
    rgb = torch.empty_like(image)
    rgb[:, 0] = image[:, 0] + 1.13983 * image[:, 2]
    rgb[:, 1] = image[:, 0] - 0.39465 * image[:, 1] - 0.58060 * image[:, 2]
    rgb[:, 2] = image[:, 0] + 2.03211 * image[:, 1]
    return rgb


class _BaseJpegApproxNoise(BaseNoiseLayer):
    """Shared differentiable JPEG approximation on 8x8 DCT blocks."""

    block_size = 8

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        dct_matrix = _build_ortho_dct_matrix(self.block_size)
        self.register_buffer("dct_matrix", dct_matrix, persistent=False)
        self.register_buffer("idct_matrix", dct_matrix.transpose(0, 1), persistent=False)

    def apply_noise(self, cover: Tensor, encoded: Tensor) -> Tensor:
        del cover
        original_height, original_width = encoded.shape[-2:]
        padded = self._pad_to_block_size(encoded)
        transformed = _rgb_to_yuv(padded)
        dct_blocks = self._block_dct(transformed)
        filtered_blocks = self._filter_dct_blocks(dct_blocks)
        reconstructed = self._block_idct(filtered_blocks)
        reconstructed = _yuv_to_rgb(reconstructed)
        return reconstructed[:, :, :original_height, :original_width]

    def _pad_to_block_size(self, image: Tensor) -> Tensor:
        pad_h = (self.block_size - image.shape[-2] % self.block_size) % self.block_size
        pad_w = (self.block_size - image.shape[-1] % self.block_size) % self.block_size
        return F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def _block_dct(self, image: Tensor) -> Tensor:
        block = self.block_size
        blocks = image.unfold(2, block, block).unfold(3, block, block)
        dct = torch.matmul(self.dct_matrix.to(device=image.device, dtype=image.dtype), blocks)
        dct = torch.matmul(dct, self.dct_matrix.transpose(0, 1).to(device=image.device, dtype=image.dtype))
        return dct

    def _block_idct(self, coefficients: Tensor) -> Tensor:
        block = self.block_size
        image = torch.matmul(self.idct_matrix.to(device=coefficients.device, dtype=coefficients.dtype), coefficients)
        image = torch.matmul(image, self.idct_matrix.transpose(0, 1).to(device=coefficients.device, dtype=coefficients.dtype))
        image = image.permute(0, 1, 2, 4, 3, 5).contiguous()
        return image.view(
            image.shape[0],
            image.shape[1],
            image.shape[2] * block,
            image.shape[4] * block,
        )

    def _filter_dct_blocks(self, dct_blocks: Tensor) -> Tensor:
        raise NotImplementedError


class JpegMaskNoise(_BaseJpegApproxNoise):
    """JPEG-Mask: keep low-frequency DCT coefficients and zero out the rest."""

    def __init__(self, y_keep_count: int = 25, uv_keep_count: int = 9) -> None:
        super().__init__(name="jpeg_mask")
        mask = torch.stack(
            [
                _build_keep_mask(y_keep_count, self.block_size),
                _build_keep_mask(uv_keep_count, self.block_size),
                _build_keep_mask(uv_keep_count, self.block_size),
            ]
        )
        self.register_buffer("mask", mask, persistent=False)

    def _filter_dct_blocks(self, dct_blocks: Tensor) -> Tensor:
        channel_count = dct_blocks.shape[1]
        mask = self.mask[:channel_count].to(device=dct_blocks.device, dtype=dct_blocks.dtype)
        mask = mask.view(1, channel_count, 1, 1, self.block_size, self.block_size)
        return dct_blocks * mask


class JpegDropNoise(_BaseJpegApproxNoise):
    """JPEG-Drop: progressively drop higher-frequency DCT coefficients."""

    def __init__(self) -> None:
        super().__init__(name="jpeg_drop")
        y_table = torch.tensor(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=torch.float32,
        )
        uv_table = torch.tensor(
            [
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ],
            dtype=torch.float32,
        )
        keep_prob = torch.stack(
            [
                y_table.min() / y_table,
                uv_table.min() / uv_table,
                uv_table.min() / uv_table,
            ]
        )
        self.register_buffer("keep_prob", keep_prob.clamp(0.0, 1.0), persistent=False)

    def _filter_dct_blocks(self, dct_blocks: Tensor) -> Tensor:
        channel_count = dct_blocks.shape[1]
        keep_prob = self.keep_prob[:channel_count].to(device=dct_blocks.device, dtype=dct_blocks.dtype)
        keep_prob = keep_prob.view(1, channel_count, 1, 1, self.block_size, self.block_size)
        random_mask = torch.rand(
            dct_blocks.shape,
            device=dct_blocks.device,
            dtype=dct_blocks.dtype,
        )
        keep_mask = (random_mask < keep_prob).to(dct_blocks.dtype)
        return dct_blocks * keep_mask
