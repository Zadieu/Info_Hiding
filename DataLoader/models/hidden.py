import torch
import torch.nn as nn

from models.decoder import Decoder
from models.discriminator import Discriminator
from models.encoder import Encoder
from models.noise import NoiseLayer


class HiDDeNModel(nn.Module):
    def __init__(self, C: int, H: int, W: int, L: int, noise_type: str, noise_kwargs: dict):
        super().__init__()
        self.encoder = Encoder(C, H, W, L)
        self.noise = NoiseLayer(noise_type=noise_type, **noise_kwargs)
        self.decoder = Decoder(C, L)
        self.discriminator = Discriminator(C)

    def forward(self, cover: torch.Tensor, message: torch.Tensor):
        encoded = self.encoder(cover, message)
        noised = self.noise(cover, encoded)
        decoded = self.decoder(noised)
        return encoded, noised, decoded
