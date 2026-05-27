"""Adversary A_gamma — Appendix A (3 conv blocks, GAP, linear -> P(encoded))."""
import torch
import torch.nn as nn

from models.blocks import conv_bn_relu


class Discriminator(nn.Module):
    def __init__(self, C: int, n_blocks: int = 3):
        super().__init__()
        blocks = [conv_bn_relu(C if i == 0 else 64, 64) for i in range(n_blocks)]
        self.conv = nn.Sequential(*blocks)
        self.linear = nn.Linear(64, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        x = x.mean(dim=(2, 3))
        logits = self.linear(x)
        return torch.sigmoid(logits)
