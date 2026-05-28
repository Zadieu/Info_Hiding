"""
Basic building block of the HiDDeN networks: Conv-BN-ReLU.

Reference
---------
HiDDeN: Hiding Data With Deep Networks (Zhu, Kaplan, Johnson, Fei-Fei; ECCV 2018),
Appendix A "Model Architecture":

    "We denote the combination of Convolution, Batch Normalization and ReLU as
     a Conv-BN-ReLU block. In our experiments, all Conv-BN-ReLU blocks, unless
     otherwise specified, have 3x3 kernels, stride 1, and padding 1."

A 3x3 / stride-1 / padding-1 convolution preserves the spatial size (H, W), which
is what lets the Encoder concatenate the message volume with the image features
and keep producing same-size tensors throughout the network.

Note on `bias=False`
---------------------
Every convolution here is immediately followed by BatchNorm. BatchNorm has its own
learnable affine shift (beta), so a convolution bias would be mathematically
redundant (BN subtracts the running mean anyway). We therefore disable it. This is
standard practice and matches the official authoritative PyTorch reference
(ando-khachatryan/HiDDeN), which the original authors link to from their repo.
"""
import torch.nn as nn


class ConvBNRelu(nn.Module):
    """Conv(3x3, stride 1, pad 1) -> BatchNorm2d -> ReLU.

    Args:
        channels_in:  number of input channels.
        channels_out: number of output channels.
    """

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


def conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Module:
    """Functional alias kept for backward compatibility with the rest of the repo.

    `kernel_size` is accepted for flexibility; the paper always uses 3x3 with
    padding = kernel_size // 2 so that the spatial dimensions are preserved.
    """
    if kernel_size == 3:
        return ConvBNRelu(in_ch, out_ch)
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
