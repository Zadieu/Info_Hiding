"""
Encoder  E_theta  -- inserts a binary message into a cover image.

Paper: HiDDeN (Zhu et al., ECCV 2018), Section 3 + Appendix A.

Appendix A, verbatim
--------------------
    "Given input I_co of shape C x H x W, the encoder applies four Conv-BN-ReLU
     blocks with 64 output filters to the input image, resulting in a 64 x H x W
     image activation volume. It then replicates M spatially H x W times to form
     a L x H x W message volume. The two volumes, along with the original image,
     are concatenated channel-wise into a single (64 + L + C) x H x W activation.
     The encoder then applies a Conv-BN-ReLU block with 64 output filters. A final
     convolution layer with a 1 x 1 kernel, stride 1, no padding and C output
     filters is used to produce I_en with shape C x H x W. No activation function
     is applied after the final convolution."

Data flow (this is the core of assignment part 2)
--------------------------------------------------
    cover  I_co : (B, C, H, W)
    message M   : (B, L)              binary bits in {0, 1}

    1) feature extraction : conv_layers(I_co) -> (B, 64, H, W)
    2) MESSAGE REPLICATION ("message volume"):
         reshape M to (B, L, 1, 1) and broadcast it to (B, L, H, W) so that the
         SAME L-bit message is visible at *every* spatial location. This is what
         lets each convolutional filter in the next layer read the whole message
         no matter where it sits on the image.
    3) CHANNEL CONCATENATION:
         cat([ features(64) , message_volume(L) , cover(C) ], dim=1)
         -> (B, 64 + L + C, H, W)
    4) one more Conv-BN-ReLU (-> 64 channels)
    5) 1x1 conv (-> C channels), NO activation  ->  encoded image I_en (B, C, H, W)
"""
import torch
import torch.nn as nn

from models.blocks import ConvBNRelu


class Encoder(nn.Module):
    """HiDDeN encoder E_theta.

    Args:
        C: number of image channels (1 for grayscale steganography, 3 for YUV/RGB watermarking).
        H, W: spatial size of the cover images used at training time. Only used to keep the
              configuration explicit; the forward pass reads the actual H/W from the input
              tensor, so the module also works on images of a different size.
        L: message length in bits.
        encoder_blocks: number of Conv-BN-ReLU feature blocks before concatenation (paper: 4).
        channels: number of intermediate feature channels (paper: 64).
    """

    def __init__(self, C: int, H: int, W: int, L: int,
                 encoder_blocks: int = 4, channels: int = 64):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.L = L
        self.channels = channels

        # (1) feature-extraction trunk: first block maps C -> channels, the rest keep `channels`.
        layers = [ConvBNRelu(C, channels)]
        for _ in range(encoder_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))
        self.conv_layers = nn.Sequential(*layers)

        # (4) block applied AFTER concatenation: (channels + L + C) -> channels
        self.after_concat_layer = ConvBNRelu(channels + L + C, channels)

        # (5) final 1x1 conv -> C channels, no activation (produces the encoded image)
        self.final_layer = nn.Conv2d(channels, C, kernel_size=1, stride=1, padding=0)

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cover:   (B, C, H, W) cover image, pixel values in [0, 1].
            message: (B, L) binary message.
        Returns:
            (B, C, H, W) encoded (stego) image.
        """
        b, _, h, w = cover.shape

        # (1) image features
        features = self.conv_layers(cover)

        # (2) replicate the message over the whole spatial grid -> "message volume"
        #     (B, L) -> (B, L, 1, 1) -> (B, L, H, W) via broadcasting (no data copy).
        message_volume = message.view(b, self.L, 1, 1).expand(b, self.L, h, w)

        # (3) concatenate features + message volume + original cover along channels
        concat = torch.cat([features, message_volume, cover], dim=1)

        # (4) + (5)
        x = self.after_concat_layer(concat)
        encoded_image = self.final_layer(x)
        return encoded_image
