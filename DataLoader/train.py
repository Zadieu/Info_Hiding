"""
HiDDeN training — strict paper losses (Sec. 3), Adam lr=1e-3, batch=12.

Usage (GPU, COCO):
  python train.py --data_root data/coco/train2017

Smoke test without COCO:
  python train.py --use_synthetic --num_epochs 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainConfig
from dataset import build_loaders, sample_messages
from losses import bit_accuracy, loss_A, loss_G, loss_I, loss_M
from models.hidden import HiDDeNModel
from utils.checkpoint import load_checkpoint, save_checkpoint


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train HiDDeN (ECCV 2018)")
    p.add_argument("--data_root", type=str, default="data/coco/train2017")
    p.add_argument("--use_synthetic", action="store_true")
    p.add_argument("--train_size", type=int, default=10_000)
    p.add_argument("--val_size", type=int, default=1_000)
    p.add_argument("--C", type=int, default=1)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--W", type=int, default=16)
    p.add_argument("--L", type=int, default=52)
    p.add_argument("--noise", type=str, default="identity")
    p.add_argument("--lambda_I", type=float, default=1.0)
    p.add_argument("--lambda_G", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--experiment_name", type=str, default="hidden_stego")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        use_synthetic=args.use_synthetic,
        train_size=args.train_size,
        val_size=args.val_size,
        image_channels=args.C,
        image_height=args.H,
        image_width=args.W,
        message_length=args.L,
        noise_type=args.noise,
        lambda_I=args.lambda_I,
        lambda_G=args.lambda_G,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
    )


def resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.warning("CUDA not available — training on CPU.")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, discriminator, loader, cfg, device):
    model.eval()
    discriminator.eval()
    totals = {"L_M": 0.0, "L_I": 0.0, "L_G": 0.0, "L_A": 0.0, "bit_acc": 0.0}
    n = 0
    for cover, _ in loader:
        cover = cover.to(device, non_blocking=True)
        message = sample_messages(cover.size(0), cfg.L, device)
        encoded, noised, decoded = model(cover, message)
        p_cover = discriminator(cover)
        p_enc = discriminator(encoded)
        totals["L_M"] += loss_M(message, decoded).item()
        totals["L_I"] += loss_I(cover, encoded).item()
        totals["L_G"] += loss_G(p_enc).item()
        totals["L_A"] += loss_A(p_cover, p_enc).item()
        totals["bit_acc"] += bit_accuracy(message, decoded)
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def train_one_epoch(model, discriminator, opt_ed, opt_d, loader, cfg, device, epoch):
    model.train()
    discriminator.train()
    enc_dec = model
    stats = {"L_M": 0.0, "L_I": 0.0, "L_G": 0.0, "L_A": 0.0, "bit_acc": 0.0}
    steps = 0

    for step, (cover, _) in enumerate(loader, start=1):
        cover = cover.to(device, non_blocking=True)
        message = sample_messages(cover.size(0), cfg.L, device)

        # --- Eq. (2): train discriminator A ---
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            encoded = enc_dec.encoder(cover, message)
        p_cover = discriminator(cover)
        p_enc = discriminator(encoded.detach())
        la = loss_A(p_cover, p_enc)
        la.backward()
        opt_d.step()

        # --- Eq. (1): train encoder + decoder ---
        opt_ed.zero_grad(set_to_none=True)
        encoded, noised, decoded = enc_dec(cover, message)
        p_enc_g = discriminator(encoded)
        lm = loss_M(message, decoded)
        li = loss_I(cover, encoded)
        lg = loss_G(p_enc_g)
        loss_ed = lm + cfg.lambda_I * li + cfg.lambda_G * lg
        loss_ed.backward()
        opt_ed.step()

        stats["L_M"] += lm.item()
        stats["L_I"] += li.item()
        stats["L_G"] += lg.item()
        stats["L_A"] += la.item()
        stats["bit_acc"] += bit_accuracy(message, decoded)
        steps += 1

        if step % cfg.log_every == 0:
            logging.info(
                "epoch %d step %d | L_M %.4f L_I %.4f L_G %.4f L_A %.4f bit_acc %.4f",
                epoch,
                step,
                lm.item(),
                li.item(),
                lg.item(),
                la.item(),
                bit_accuracy(message, decoded),
            )

    return {k: v / max(steps, 1) for k, v in stats.items()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args()
    device = resolve_device(cfg)
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(device))

    train_loader, val_loader = build_loaders(
        cfg.data_root,
        cfg.C,
        cfg.H,
        cfg.W,
        cfg.train_size,
        cfg.val_size,
        cfg.batch_size,
        cfg.num_workers,
        cfg.use_synthetic,
    )

    noise_kwargs = {
        "dropout_p": cfg.noise_dropout_p,
        "cropout_p": cfg.noise_cropout_p,
        "gaussian_sigma": cfg.noise_gaussian_sigma,
        "crop_p": cfg.noise_crop_p,
    }
    model = HiDDeNModel(cfg.C, cfg.H, cfg.W, cfg.L, cfg.noise_type, noise_kwargs).to(device)
    discriminator = model.discriminator
    enc_dec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_ed = Adam(enc_dec_params, lr=cfg.lr)
    opt_d = Adam(discriminator.parameters(), lr=cfg.lr)

    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    resume_path = None
    for i, a in enumerate(sys.argv):
        if a == "--resume" and i + 1 < len(sys.argv):
            resume_path = sys.argv[i + 1]
    if resume_path:
        start_epoch = load_checkpoint(Path(resume_path), model, discriminator, opt_ed, opt_d) + 1
        logging.info("Resumed from epoch %d", start_epoch - 1)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_stats = train_one_epoch(model, discriminator, opt_ed, opt_d, train_loader, cfg, device, epoch)
        val_stats = validate(model, discriminator, val_loader, cfg, device)
        logging.info(
            "Epoch %d/%d | train bit_acc %.4f | val bit_acc %.4f | val L_M %.4f",
            epoch,
            cfg.num_epochs,
            train_stats["bit_acc"],
            val_stats["bit_acc"],
            val_stats["L_M"],
        )
        if epoch % cfg.save_every == 0 or epoch == cfg.num_epochs:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                epoch,
                model,
                discriminator,
                opt_ed,
                opt_d,
                asdict(cfg),
            )


if __name__ == "__main__":
    main()
