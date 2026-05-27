"""HiDDeN training configuration (paper defaults where specified)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # --- Data (paper: 10k train, 1k val from COCO) ---
    data_root: str = "data/coco/train2017"
    train_size: int = 10_000
    val_size: int = 1_000
    use_synthetic: bool = False  # True: random tensors when COCO missing (smoke test)

    # --- Steganography experiment (Sec. 4.1): grayscale 16x16, L=52, Identity noise ---
    image_channels: int = 1
    image_height: int = 16
    image_width: int = 16
    message_length: int = 52
    noise_type: str = "identity"  # identity | dropout | cropout | gaussian | crop

    # Noise hyperparameters (Sec. 3)
    noise_dropout_p: float = 0.3
    noise_cropout_p: float = 0.3
    noise_gaussian_sigma: float = 2.0
    noise_crop_p: float = 0.035

    # --- Loss weights (Eq. 1); tune if needed — not fixed in main paper ---
    lambda_I: float = 1.0
    lambda_G: float = 0.1

    # --- Optimizer (paper: Adam lr=1e-3, batch=12, 200 epochs) ---
    lr: float = 1e-3
    batch_size: int = 12
    num_epochs: int = 200
    num_workers: int = 4

    # --- IO ---
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "hidden_stego"
    save_every: int = 10
    log_every: int = 50

    # --- Device ---
    device: Optional[str] = None  # None -> auto cuda/cpu

    @property
    def H(self) -> int:
        return self.image_height

    @property
    def W(self) -> int:
        return self.image_width

    @property
    def C(self) -> int:
        return self.image_channels

    @property
    def L(self) -> int:
        return self.message_length
