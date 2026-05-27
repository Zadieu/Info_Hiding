"""Encoder E_theta — Appendix A."""
import torch
import torch.nn as nn

from models.blocks import conv_bn_relu


class Encoder(nn.Module):
    def __init__(self, C: int, H: int, W: int, L: int, n_pre: int = 4, n_mid: int = 1):
        super().__init__()
        self.C, self.L = C, L
        pre = [conv_bn_relu(C if i == 0 else 64, 64) for i in range(n_pre)]
        self.pre = nn.Sequential(*pre)
        self.mid = conv_bn_relu(64 + L + C, 64)
        self.final = nn.Conv2d(64, C, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        b, _, h, w = cover.shape
        h_feat = self.pre(cover)
        msg = message.view(b, self.L, 1, 1).expand(b, self.L, h, w)
        x = torch.cat([h_feat, msg, cover], dim=1)
        x = self.mid(x)
        return self.final(x)
