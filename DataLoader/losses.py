"""Paper Sec. 3 losses: L_I, L_M, L_G, L_A."""
import torch
import torch.nn.functional as F

_EPS = 1e-8


def loss_I(cover: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
    """L_I = ||I_co - I_en||_2^2 / (C*H*W), per sample then mean over batch."""
    b, c, h, w = cover.shape
    per_sample = (cover - encoded).pow(2).view(b, -1).sum(dim=1) / (c * h * w)
    return per_sample.mean()


def loss_M(msg_in: torch.Tensor, msg_out: torch.Tensor) -> torch.Tensor:
    """L_M = ||M_in - M_out||_2^2 / L."""
    b, L = msg_in.shape
    per_sample = (msg_in - msg_out).pow(2).sum(dim=1) / L
    return per_sample.mean()


def loss_G(adv_prob_encoded: torch.Tensor) -> torch.Tensor:
    """L_G = log(1 - A(I_en)); encoder minimizes this (A -> 0)."""
    p = adv_prob_encoded.clamp(0.0, 1.0 - _EPS)
    return torch.mean(torch.log(1.0 - p + _EPS))


def loss_A(adv_prob_cover: torch.Tensor, adv_prob_encoded: torch.Tensor) -> torch.Tensor:
    """L_A = log(1 - A(I_co)) + log(A(I_en)); discriminator minimizes this."""
    p_cover = adv_prob_cover.clamp(0.0, 1.0 - _EPS)
    p_enc = adv_prob_encoded.clamp(_EPS, 1.0)
    return torch.mean(torch.log(1.0 - p_cover + _EPS) + torch.log(p_enc))


def bit_accuracy(msg_in: torch.Tensor, msg_out: torch.Tensor) -> float:
    """Paper Appendix A: round M_out to {0,1} before compare."""
    pred = msg_out.detach().round().clamp(0, 1)
    correct = (pred == msg_in).float().mean().item()
    return correct
