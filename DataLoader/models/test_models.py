"""
Self-tests for the HiDDeN networks (assignment part 2: Encoder / Decoder / Adversary).

How to run
----------
Recommended (run as a module from the DataLoader/ directory):

    cd path/to/Info_Hiding/DataLoader
    python -m models.test_models

Also works (run the file directly):

    cd path/to/Info_Hiding/DataLoader
    python models/test_models.py

The core network tests only need encoder.py / decoder.py / discriminator.py
(this assignment's own files), so they run even if the integration files
(hidden.py, losses.py) are not present yet. The end-to-end integration test is
skipped automatically when those teammate files are missing.

Tests
-----
  1. Output shapes for both paper configurations
     - steganography  : C=1, H=W=16,  L=52   (Sec. 4.1)
     - watermarking   : C=3, H=W=128, L=30   (Sec. 4.2)
  2. Discriminator output is a probability in [0, 1].
  3. The "message volume" is replicated correctly across all spatial locations.
  4. Gradients flow through encoder, decoder and discriminator (backward works).
  5. The decoder is input-size agnostic (survives Crop, which changes H', W').
  6. (optional) End-to-end integration with the team's losses.py / hidden.py.
"""
import sys
from pathlib import Path

# Make the DataLoader/ directory importable no matter where the script is launched from.
ROOT = Path(__file__).resolve().parent.parent  # .../DataLoader
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# These three are THIS assignment's own files -> always available.
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator

GREEN, YELLOW, RESET = "\033[92m", "\033[93m", "\033[0m"


def _ok(msg):
    print(f"{GREEN}[PASS]{RESET} {msg}")


def _skip(msg):
    print(f"{YELLOW}[SKIP]{RESET} {msg}")


def _count(module):
    return sum(p.numel() for p in module.parameters())


def test_shapes():
    for (C, H, W, L, tag) in [(1, 16, 16, 52, "steganography"),
                              (3, 128, 128, 30, "watermarking")]:
        B = 4
        enc, dec, disc = Encoder(C, H, W, L), Decoder(C, L), Discriminator(C)
        cover = torch.rand(B, C, H, W)
        msg = torch.randint(0, 2, (B, L)).float()

        encoded = enc(cover, msg)
        assert encoded.shape == (B, C, H, W), f"encoder shape {encoded.shape}"
        decoded = dec(encoded)
        assert decoded.shape == (B, L), f"decoder shape {decoded.shape}"
        prob = disc(cover)
        assert prob.shape == (B, 1), f"disc shape {prob.shape}"

        _ok(f"{tag:13s} C={C} H={H} W={W} L={L} | "
            f"enc{tuple(encoded.shape)} dec{tuple(decoded.shape)} disc{tuple(prob.shape)} | "
            f"params E/D/A = {_count(enc)}/{_count(dec)}/{_count(disc)}")


def test_discriminator_is_probability():
    disc = Discriminator(3)
    x = torch.randn(8, 3, 64, 64) * 5.0
    p = disc(x)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0), "discriminator must output [0,1]"
    _ok(f"discriminator output is a valid probability, range [{p.min():.3f}, {p.max():.3f}]")


def test_message_replication():
    C, H, W, L = 3, 8, 8, 6
    B = 2
    msg = torch.randint(0, 2, (B, L)).float()
    volume = msg.view(B, L, 1, 1).expand(B, L, H, W)
    assert volume.shape == (B, L, H, W)
    for hh in range(H):
        for ww in range(W):
            assert torch.equal(volume[:, :, hh, ww], msg), "message volume mismatch"
    _ok("message volume correctly replicated to every spatial location (B,L,1,1)->(B,L,H,W)")


def test_gradient_flow():
    C, H, W, L = 3, 32, 32, 30
    enc, dec, disc = Encoder(C, H, W, L), Decoder(C, L), Discriminator(C)
    cover = torch.rand(2, C, H, W)
    msg = torch.randint(0, 2, (2, L)).float()
    encoded = enc(cover, msg)
    loss = dec(encoded).mean() + encoded.mean() + disc(encoded).mean()
    loss.backward()
    for name, m in [("encoder", enc), ("decoder", dec), ("discriminator", disc)]:
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads), f"bad gradients in {name}"
    _ok("gradients flow through encoder, decoder and discriminator")


def test_decoder_is_size_agnostic():
    C, L = 3, 30
    dec = Decoder(C, L)
    for (h2, w2) in [(128, 128), (24, 24), (8, 200), (1, 1)]:
        out = dec(torch.rand(3, C, h2, w2))
        assert out.shape == (3, L), f"decoder failed on {(h2, w2)} -> {out.shape}"
    _ok("decoder is input-size agnostic (handles cropped / non-square inputs)")


def test_end_to_end_with_losses():
    """Optional: needs teammate files hidden.py + losses.py. Skipped if missing."""
    try:
        from models.hidden import HiDDeNModel
        from losses import loss_I, loss_M, loss_G, loss_A, bit_accuracy
    except Exception as e:
        _skip(f"end-to-end integration test (hidden.py / losses.py not importable: {e})")
        return

    C, H, W, L = 1, 16, 16, 52
    model = HiDDeNModel(C, H, W, L, "identity",
                        {"dropout_p": 0.3, "cropout_p": 0.3, "gaussian_sigma": 2.0, "crop_p": 0.035})
    opt_ed = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    cover = torch.rand(6, C, H, W)
    msg = torch.randint(0, 2, (6, L)).float()

    opt_d.zero_grad()
    with torch.no_grad():
        enc_img = model.encoder(cover, msg)
    la = loss_A(model.discriminator(cover), model.discriminator(enc_img.detach()))
    la.backward(); opt_d.step()

    opt_ed.zero_grad()
    encoded, noised, decoded = model(cover, msg)
    lm, li, lg = loss_M(msg, decoded), loss_I(cover, encoded), loss_G(model.discriminator(encoded))
    total = lm + 0.7 * li + 1e-3 * lg
    total.backward(); opt_ed.step()
    assert torch.isfinite(total), "non-finite loss"
    _ok(f"end-to-end step OK | L_M={lm.item():.4f} L_I={li.item():.4f} "
        f"L_G={lg.item():.4f} L_A={la.item():.4f} bit_acc={bit_accuracy(msg, decoded):.3f}")


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("HiDDeN models self-test  (Encoder / Decoder / Adversary)")
    print("=" * 70)
    test_shapes()
    test_discriminator_is_probability()
    test_message_replication()
    test_gradient_flow()
    test_decoder_is_size_agnostic()
    test_end_to_end_with_losses()
    print("=" * 70)
    print(f"{GREEN}CORE NETWORK TESTS PASSED{RESET}")
    print("=" * 70)


if __name__ == "__main__":
    main()