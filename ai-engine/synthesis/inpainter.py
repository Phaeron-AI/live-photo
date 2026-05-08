"""
inpainter.py — LaMa-based inpainter fine-tuned for physics hole filling.

Architecture overview:
  LaMa (Large Mask inpainting) uses Fast Fourier Convolutions (FFC) which
  give the network a global receptive field from the first layer. This is
  critical for inpainting — local convolutions can only fill holes using
  nearby pixels, but backgrounds often have structure (sky gradients, floor
  textures, wall patterns) that requires understanding the full image.

  Fast Fourier Convolution splits feature maps into:
    - Local branch:  standard convolution (spatial detail)
    - Global branch: FFT → complex multiply → IFFT (global structure)
  These are summed and passed to the next layer.

Our approach (hybrid):
  1. Load pretrained LaMa weights (trained on Places2 — diverse backgrounds)
  2. Freeze the FFC encoder — it already knows global structure
  3. Fine-tune only the decoder on physics-specific hole shapes
     (edge-adjacent, 3-8% of image area, organically shaped)

Reference:
  Suvorov et al. (2022) — Resolution-robust Large Mask Inpainting with
  Fourier Convolutions. WACV 2022. https://arxiv.org/abs/2109.07161
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Fast Fourier Convolution block
# ---------------------------------------------------------------------------

class FourierUnit(nn.Module):
    """
    Global branch of FFC: operates in frequency domain.

    Forward pass:
        1. FFT2 the input feature map → complex spectrum
        2. Apply a learnable complex-valued linear transform
        3. IFFT2 back to spatial domain
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels * 2, out_channels * 2,
            kernel_size=1, bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_channels * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        spectrum     = torch.fft.rfft2(x, norm='ortho')
        spectrum_cat = torch.cat([spectrum.real, spectrum.imag], dim=1)

        out        = self.act(self.bn(self.conv(spectrum_cat)))
        real, imag = out.chunk(2, dim=1)

        spectrum_out = torch.complex(real, imag)
        out_spatial  = torch.fft.irfft2(spectrum_out, s=(H, W), norm='ortho')
        return out_spatial


class FFCResBlock(nn.Module):
    """
    FFC Residual Block — core building block of LaMa.

    ratio_g: fraction of channels assigned to global branch.
             0.75 in original LaMa.
    """

    def __init__(self, channels: int, ratio_g: float = 0.75):
        super().__init__()
        self.global_channels = int(channels * ratio_g)
        self.local_channels  = channels - self.global_channels

        if self.local_channels > 0:
            self.local_conv = nn.Sequential(
                nn.Conv2d(self.local_channels, self.local_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.local_channels),
                nn.ReLU(inplace=True),
            )

        if self.global_channels > 0:
            self.global_conv = nn.Sequential(
                FourierUnit(self.global_channels, self.global_channels),
                nn.BatchNorm2d(self.global_channels),
                nn.ReLU(inplace=True),
            )

        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.local_channels > 0 and self.global_channels > 0:
            x_local  = self.local_conv(x[:, :self.local_channels])
            x_global = self.global_conv(x[:, self.local_channels:])
            x_out    = torch.cat([x_local, x_global], dim=1)
        elif self.global_channels > 0:
            x_out = self.global_conv(x)
        else:
            x_out = self.local_conv(x)

        x_out = F.relu(self.bn(self.mix(x_out)), inplace=True)
        return x_out + residual


# ---------------------------------------------------------------------------
# Encoder (frozen after LaMa weight loading)
# ---------------------------------------------------------------------------

class LaMaEncoder(nn.Module):
    """
    LaMa encoder: downsample + FFC residual blocks.

    Input:  (B, 4, H, W) — RGB image concatenated with binary hole mask
    Output: (B, 256, H/8, W/8) — latent feature map
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.stem  = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )

        self.ffc_blocks = nn.Sequential(
            *[FFCResBlock(c * 4, ratio_g=0.75) for _ in range(6)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
      s0 = self.stem(x)
      s1 = self.down1(s0)
      s2 = self.down2(s1)
      s3 = self.down3(s2)
      z  = self.ffc_blocks(s3)
      return z, s2, s1, s0


# ---------------------------------------------------------------------------
# Decoder (fine-tuned)
# ---------------------------------------------------------------------------

class LaMaDecoder(nn.Module):
  """
  Decoder with skip connections from encoder.

  Input:  (B, 256, H/8, W/8) latent + skip connections
  Output: (B, 3, H, W) inpainted RGB
  """

  def __init__(self, base_channels: int = 64):
    super().__init__()
    c = base_channels

    self.up1   = nn.Sequential(
      nn.ConvTranspose2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(c * 4),
      nn.ReLU(inplace=True),
    )
    self.proj1 = nn.Sequential(
      nn.Conv2d(c * 8, c * 2, kernel_size=1, bias=False),
      nn.BatchNorm2d(c * 2),
      nn.ReLU(inplace=True),
    )

    self.up2   = nn.Sequential(
      nn.ConvTranspose2d(c * 2, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(c * 2),
      nn.ReLU(inplace=True),
    )
    self.proj2 = nn.Sequential(
      nn.Conv2d(c * 4, c, kernel_size=1, bias=False),
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
    )

    self.up3   = nn.Sequential(
      nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
    )
    self.proj3 = nn.Sequential(
      nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
      nn.BatchNorm2d(c),
      nn.ReLU(inplace=True),
    )

    self.head = nn.Sequential(
      nn.Conv2d(c, c // 2, kernel_size=3, padding=1, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(c // 2, 3, kernel_size=1),
      nn.Tanh(),
    )

  def forward(
    self,
    z: torch.Tensor,
    s2: torch.Tensor,
    s1: torch.Tensor,
    s0: torch.Tensor,
  ) -> torch.Tensor:
    x = self.up1(z)
    x = self.proj1(torch.cat([x, s2], dim=1))

    x = self.up2(x)
    x = self.proj2(torch.cat([x, s1], dim=1))

    x = self.up3(x)
    x = self.proj3(torch.cat([x, s0], dim=1))

    x = self.head(x)
    return (x + 1) / 2  # [-1,1] → [0,1]


# ---------------------------------------------------------------------------
# Full inpainter model
# ---------------------------------------------------------------------------

class LaMaInpainter(nn.Module):
  """
  Full LaMa inpainter: encoder (frozen) + decoder (fine-tuned).

  Input tensors:
      image: (B, 3, H, W) float32 in [0, 1]
      mask:  (B, 1, H, W) float32 — 1.0 where hole exists, 0.0 elsewhere

  Output:
      inpainted: (B, 3, H, W) float32 in [0, 1]
  """

  def __init__(self, base_channels: int = 64):
    super().__init__()
    self.encoder = LaMaEncoder(base_channels)
    self.decoder = LaMaDecoder(base_channels)

  def forward(
    self,
    image: torch.Tensor,
    hole_mask: torch.Tensor,
  ) -> torch.Tensor:
    masked_image = image * (1 - hole_mask)
    x = torch.cat([masked_image, hole_mask], dim=1)  # (B, 4, H, W)

    z, s2, s1, s0 = self.encoder(x)
    inpainted = self.decoder(z, s2, s1, s0)

    output = inpainted * hole_mask + image * (1 - hole_mask)
    return output

  def freeze_encoder(self) -> None:
    """Freeze encoder weights for fine-tuning phase."""
    for param in self.encoder.parameters():
      param.requires_grad = False
    print("Encoder frozen. Only decoder will be updated during fine-tuning.")

  def unfreeze_encoder(self) -> None:
    """Unfreeze encoder for full fine-tuning (later stage)."""
    for param in self.encoder.parameters():
      param.requires_grad = True
    print("Encoder unfrozen. Full model will be updated.")

  def load_pretrained(self, path: str, strict: bool = False) -> None:
    """
    Load pretrained LaMa weights.

    Handles three common checkpoint formats:
      - Raw state dict
      - {'model': state_dict}
      - {'state_dict': state_dict}
      - {'generator': state_dict}  ← official LaMa release format

    FIX: The official LaMa release wraps all weights under a 'generator.'
    key prefix. Without stripping it, load_state_dict produces only
    unexpected/missing key warnings and the model stays randomly
    initialised, which is a silent failure.
    """
    state = torch.load(path, map_location='cpu')

    # Unwrap common top-level wrapper keys
    if 'model' in state:
      state = state['model']
    elif 'state_dict' in state:
      state = state['state_dict']

    # Strip 'generator.' prefix used in the official LaMa checkpoint
    if any(k.startswith('generator.') for k in state):
      state = {k[len('generator.'):]: v for k, v in state.items()}

    missing, unexpected = self.load_state_dict(state, strict=strict)
    print(f"Loaded pretrained weights from {path}")
    if missing:
      print(f"  Missing keys (will be randomly init): {len(missing)}")
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
  model: LaMaInpainter,
  warped_frame: np.ndarray,    # (H, W, 3) uint8 — forward-splatted frame
  hole_mask: np.ndarray,       # (H, W) uint8 — 255 at holes
  device: str = 'cpu',
) -> np.ndarray:
  """
  Run inpainting on a single frame.

  Args:
      model:        Trained LaMaInpainter.
      warped_frame: Forward-splatted frame with black holes.
      hole_mask:    Binary hole mask, 255 = hole.
      device:       'cpu' or 'cuda'.

  Returns:
      Inpainted frame, uint8 (H, W, 3).
  """
  model.eval()

  img = torch.from_numpy(warped_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
  msk = torch.from_numpy((hole_mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

  img = img.to(device)
  msk = msk.to(device)

  output = model(img, msk)

  result = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
  return result.astype(np.uint8)