"""COCO cover images for HiDDeN (paper: 10k train, 1k val)."""
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class SyntheticCoverDataset(Dataset):
    """Fallback when COCO is unavailable (smoke test only)."""

    def __init__(self, n: int, C: int, H: int, W: int):
        self.n = n
        self.C, self.H, self.W = C, H, W

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        img = torch.rand(self.C, self.H, self.W, generator=g)
        return img, 0


def _image_transform(C: int, H: int, W: int, train: bool):
    tf = [transforms.Resize((H, W))]
    if C == 1:
        tf.append(transforms.Grayscale(num_output_channels=1))
    tf.append(transforms.ToTensor())  # [0, 1]
    return transforms.Compose(tf)


def _load_image_dataset(root: str, C: int, H: int, W: int, train: bool) -> Dataset:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data root not found: {root}\n"
            "Download COCO train2017 to data/coco/train2017 or pass --use_synthetic."
        )
    tf = _image_transform(C, H, W, train)
    if _has_subdirs(root):
        return datasets.ImageFolder(str(root), transform=tf)
    return _FlatImageDataset(root, tf)


class _FlatImageDataset(Dataset):
    """Images directly under root (COCO train2017 layout)."""

    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted(
            p for p in root.iterdir() if p.suffix.lower() in exts
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), 0


def _has_subdirs(root: Path) -> bool:
    return any(p.is_dir() for p in root.iterdir())


def build_loaders(
    data_root: str,
    C: int,
    H: int,
    W: int,
    train_size: int,
    val_size: int,
    batch_size: int,
    num_workers: int,
    use_synthetic: bool,
) -> Tuple[DataLoader, DataLoader]:
    if use_synthetic:
        train_ds = SyntheticCoverDataset(train_size, C, H, W)
        val_ds = SyntheticCoverDataset(val_size, C, H, W)
    else:
        full = _load_image_dataset(data_root, C, H, W, train=True)
        n = len(full)
        if n < train_size + val_size:
            raise ValueError(
                f"Need at least {train_size + val_size} images, found {n} in {data_root}"
            )
        indices = list(range(train_size + val_size))
        train_ds = Subset(full, indices[:train_size])
        val_ds = Subset(full, indices[train_size : train_size + val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def sample_messages(batch_size: int, L: int, device: torch.device) -> torch.Tensor:
    """Paper: each bit drawn uniformly at random from {0, 1}."""
    return torch.randint(0, 2, (batch_size, L), device=device, dtype=torch.float32)
