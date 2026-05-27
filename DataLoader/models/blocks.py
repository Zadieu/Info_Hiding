import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    """Appendix A: Conv-BN-ReLU, 3x3, stride 1, padding 1 (unless noted)."""
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
