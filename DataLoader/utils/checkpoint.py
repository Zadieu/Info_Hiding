import logging
from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    epoch: int,
    encoder_decoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_ed: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    config_dict: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder_decoder.encoder.state_dict(),
            "noise": encoder_decoder.noise.state_dict(),
            "decoder": encoder_decoder.decoder.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_ed": opt_ed.state_dict(),
            "opt_d": opt_d.state_dict(),
            "config": config_dict,
        },
        path,
    )
    logging.info("Saved checkpoint: %s", path)


def load_checkpoint(
    path: Path,
    encoder_decoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_ed: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    encoder_decoder.encoder.load_state_dict(ckpt["encoder"])
    encoder_decoder.noise.load_state_dict(ckpt["noise"])
    encoder_decoder.decoder.load_state_dict(ckpt["decoder"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_ed.load_state_dict(ckpt["opt_ed"])
    opt_d.load_state_dict(ckpt["opt_d"])
    return int(ckpt["epoch"])
