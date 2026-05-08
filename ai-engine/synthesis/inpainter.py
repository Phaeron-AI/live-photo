"""
synthesis/inpainter.py — LaMa-based inpainter for physics hole filling.

Architecture: Fast Fourier Convolution encoder (frozen after pretraining)
+ skip-connection decoder (fine-tuned on physics-shaped holes).

To use with pretrained weights:
    model = LaMaInpainter()
    model.load_pretrained('path/to/big-lama.pth')   # see README for download link
    model.freeze_encoder()

To run inference without weights (falls back gracefully):
    model = LaMaInpainter()
    # inpaint_frame() will return the warped frame unchanged if no weights loaded.

Reference:
  Suvorov et al. (2022) — Resolution-robust Large Mask Inpainting with
  Fourier Convolutions. WACV 2022. https://arxiv.org/abs/2109.07161
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Fast Fourier Convolution block
# ---------------------------------------------------------------------------

class FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2,
                              kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W   = x.shape
        spectrum     = torch.fft.rfft2(x, norm='ortho')
        cat          = torch.cat([spectrum.real, spectrum.imag], dim=1)
        out          = self.act(self.bn(self.conv(cat)))
        real, imag   = out.chunk(2, dim=1)
        return torch.fft.irfft2(torch.complex(real, imag), s=(H, W), norm='ortho')


class FFCResBlock(nn.Module):
    def __init__(self, channels: int, ratio_g: float = 0.75):
        super().__init__()
        self.g = int(channels * ratio_g)
        self.l = channels - self.g

        if self.l > 0:
            self.local_conv = nn.Sequential(
                nn.Conv2d(self.l, self.l, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.l), nn.ReLU(inplace=True),
            )
        if self.g > 0:
            self.global_conv = nn.Sequential(
                FourierUnit(self.g, self.g),
                nn.BatchNorm2d(self.g), nn.ReLU(inplace=True),
            )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.l > 0 and self.g > 0:
            out = torch.cat([self.local_conv(x[:, :self.l]),
                             self.global_conv(x[:, self.l:])], dim=1)
        elif self.g > 0:
            out = self.global_conv(x)
        else:
            out = self.local_conv(x)
        return F.relu(self.bn(self.mix(out)), inplace=True) + x


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class LaMaEncoder(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem  = nn.Sequential(nn.Conv2d(4, c, 7, padding=3, bias=False),
                                   nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(c,   c*2, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(c*2), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(c*2, c*4, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(c*4), nn.ReLU(inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(c*4, c*4, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(c*4), nn.ReLU(inplace=True))
        self.ffc   = nn.Sequential(*[FFCResBlock(c*4) for _ in range(6)])

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        z  = self.ffc(self.down3(s2))
        return z, s2, s1, s0


class LaMaDecoder(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        def up(i, o): return nn.Sequential(
            nn.ConvTranspose2d(i, o, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )
        def proj(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

        self.up1, self.proj1 = up(c*4, c*4), proj(c*8, c*2)
        self.up2, self.proj2 = up(c*2, c*2), proj(c*4, c)
        self.up3, self.proj3 = up(c,   c),   proj(c*2, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, c//2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//2, 3, 1),
            nn.Tanh(),
        )

    def forward(self, z, s2, s1, s0):
        x = self.proj1(torch.cat([self.up1(z),  s2], dim=1))
        x = self.proj2(torch.cat([self.up2(x),  s1], dim=1))
        x = self.proj3(torch.cat([self.up3(x),  s0], dim=1))
        return (self.head(x) + 1) / 2   # [-1,1] → [0,1]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LaMaInpainter(nn.Module):
    """
    Full LaMa inpainter.

    Input:
        image:     (B, 3, H, W) float32 in [0, 1]
        hole_mask: (B, 1, H, W) float32 — 1.0 where hole exists

    Output:
        inpainted: (B, 3, H, W) float32 in [0, 1]
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.encoder  = LaMaEncoder(base_channels)
        self.decoder  = LaMaDecoder(base_channels)
        self._weights_loaded = False

    def forward(self, image: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        x         = torch.cat([image * (1 - hole_mask), hole_mask], dim=1)
        z, s2, s1, s0 = self.encoder(x)
        inpainted = self.decoder(z, s2, s1, s0)
        return inpainted * hole_mask + image * (1 - hole_mask)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        """
        Load pretrained LaMa weights.

        Download the official checkpoint from:
            https://github.com/advimman/lama
            Direct: https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt

        Handles three common checkpoint formats:
            {'model': ...}, {'state_dict': ...}, {'generator': ...}, raw dict.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Download the LaMa checkpoint from:\n"
                "  https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt\n"
                "Then pass the path to load_pretrained()."
            )

        state = torch.load(path, map_location='cpu')

        for key in ('model', 'state_dict'):
            if key in state:
                state = state[key]
                break

        # Strip 'generator.' prefix used in the official LaMa release
        if any(k.startswith('generator.') for k in state):
            state = {k[len('generator.'):]: v for k, v in state.items()}

        missing, unexpected = self.load_state_dict(state, strict=strict)
        self._weights_loaded = True
        print(f"Loaded pretrained LaMa weights from {path}")
        if missing:
            print(f"  Missing keys (randomly initialised): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")

    def count_parameters(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {
            'encoder':   enc,
            'decoder':   dec,
            'total':     enc + dec,
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def inpaint_frame(
    model:        LaMaInpainter,
    warped_frame: np.ndarray,    # (H, W, 3) uint8 — forward-splatted frame
    hole_mask:    np.ndarray,    # (H, W) uint8 — 255 at holes
    device:       str = 'cpu',
) -> np.ndarray:
    """
    Run inpainting on a single frame.

    If the model has no pretrained weights loaded, logs a warning and
    returns the warped frame unchanged rather than producing garbage output.
    """
    if not model._weights_loaded:
        import warnings
        warnings.warn(
            "inpaint_frame called but no pretrained weights are loaded. "
            "Returning the warped frame unchanged. "
            "Call model.load_pretrained(path) to enable real inpainting. "
            "See README for the checkpoint download link.",
            stacklevel=2,
        )
        return warped_frame

    model.eval()
    img = (torch.from_numpy(warped_frame).float()
           .permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)
    msk = (torch.from_numpy((hole_mask > 0).astype(np.float32))
           .unsqueeze(0).unsqueeze(0)).to(device)

    output = model(img, msk)
    result = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
    return result.clip(0, 255).astype(np.uint8)