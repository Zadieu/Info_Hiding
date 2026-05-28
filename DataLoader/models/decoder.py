"""
Decoder  D_phi  -- recovers the hidden message from a (possibly noised) image.

Paper: HiDDeN (Zhu et al., ECCV 2018), Section 3 + Appendix A.

Appendix A, verbatim
--------------------
    "The decoder contains 7 Conv-BN-ReLU blocks with 64 filters each and one last
     Conv-BN-ReLU block with L filters. An average pooling is performed over all
     spatial dimensions and a final (L x L) linear layer produces the predicted
     message M_out. Because of the use of average pooling, the decoder assumes
     nothing about H' and W'."

Why global average pooling matters
----------------------------------
The decoder must read images of *unknown* spatial size, because the noise layer
may crop the encoded image (e.g. Crop keeps only 3.5% of the area). Global average
pooling collapses any H' x W' feature map to a single vector of length L, so the
decoder is input-size agnostic and can therefore survive cropping.
"""
import torch
import torch.nn as nn

from models.blocks import ConvBNRelu


class Decoder(nn.Module):
    """HiDDeN decoder D_phi.

    Args:
        C: number of image channels.
        L: message length in bits.
        decoder_blocks: number of 64-channel Conv-BN-ReLU blocks (paper: 7).
        channels: number of intermediate feature channels (paper: 64).
    """

    def __init__(self, C: int, L: int, decoder_blocks: int = 7, channels: int = 64):
        super().__init__()
        self.C = C
        self.L = L
        self.channels = channels

        # 7 blocks of `channels`, the first mapping C -> channels.
        layers = [ConvBNRelu(C, channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))
        # one last block producing L feature channels (one per message bit).
        layers.append(ConvBNRelu(channels, L))
        # global spatial average pooling -> (B, L, 1, 1), size-agnostic.
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        # final L x L linear layer -> predicted message logits/values.
        self.linear = nn.Linear(L, L)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H', W') noised (or clean) encoded image. H', W' may differ
                   from training size because of cropping.
        Returns:
            (B, L) predicted message (real-valued; rounded to {0,1} only at eval time).
        """
        x = self.layers(image)            # (B, L, 1, 1)
        x = x.flatten(start_dim=1)        # (B, L)  -- safe even when B == 1
        x = self.linear(x)                # (B, L)
        return x
