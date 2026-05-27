"""Decoder D_phi — Appendix A."""
import torch
import torch.nn as nn

from models.blocks import conv_bn_relu


class Decoder(nn.Module):
    def __init__(self, C: int, L: int, n_blocks: int = 7):
        super().__init__()
        blocks = [conv_bn_relu(C if i == 0 else 64, 64) for i in range(n_blocks)]
        self.conv = nn.Sequential(*blocks)
        self.final_conv = conv_bn_relu(64, L)
        self.linear = nn.Linear(L, L)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        x = self.final_conv(x)
        x = x.mean(dim=(2, 3))
        return self.linear(x)
