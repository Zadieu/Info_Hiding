"""
Adversary / Discriminator  A_gamma  -- predicts whether an image is encoded.

Paper: HiDDeN (Zhu et al., ECCV 2018), Section 3 + Appendix A.

Appendix A, verbatim
--------------------
    "The adversary has a structure similar to the decoder but has fewer
     convolutions. It contains 3 Conv-BN-ReLU blocks with 64 filters each. The
     activation volume is averaged over spatial dimensions and a linear layer with
     two output units produces the logits for the two-class classification problem."

Output convention used in THIS project
---------------------------------------
The two-class problem "cover vs. encoded" is equivalent to predicting a single
probability A(I) = P(I is encoded) in [0, 1]. Our team's losses.py implements the
paper's adversarial objective directly in that probability form:

    L_G(I_en) = log(1 - A(I_en))                          (Eq. encoder fools A)
    L_A       = log(1 - A(I_co)) + log(A(I_en))           (Eq. 2, A detects)

Both require A(.) to be a probability, so this module ends with a Sigmoid and
returns shape (B, 1) with values in [0, 1]. (A 2-logit + softmax head would be
numerically identical for two classes; we use the single-probability head to match
losses.py and the authoritative reference ando-khachatryan/HiDDeN.)
"""
import torch
import torch.nn as nn

from models.blocks import ConvBNRelu


class Discriminator(nn.Module):
    """HiDDeN adversary A_gamma.

    Args:
        C: number of image channels.
        discriminator_blocks: number of Conv-BN-ReLU blocks (paper: 3).
        channels: number of intermediate feature channels (paper: 64).
    """

    def __init__(self, C: int, discriminator_blocks: int = 3, channels: int = 64):
        super().__init__()
        self.C = C
        self.channels = channels

        layers = [ConvBNRelu(C, channels)]
        for _ in range(discriminator_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))  # GAP -> (B, channels, 1, 1)
        self.before_linear = nn.Sequential(*layers)

        self.linear = nn.Linear(channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) cover or encoded image.
        Returns:
            (B, 1) probability A(image) = P(image is encoded), in [0, 1].
        """
        x = self.before_linear(image)     # (B, channels, 1, 1)
        x = x.flatten(start_dim=1)        # (B, channels)  -- safe even when B == 1
        logits = self.linear(x)           # (B, 1)
        prob = torch.sigmoid(logits)      # probability form expected by losses.py
        return prob
