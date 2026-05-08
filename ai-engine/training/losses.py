"""
losses.py — Loss functions for LaMa inpainter fine-tuning.

Loss composition:
  L_total = λ_rec  * L_reconstruction   (pixel fidelity in hole region)
          + λ_perc * L_perceptual        (feature-space similarity)
          + λ_adv  * L_adversarial       (realism — optional, PatchGAN)
          + λ_temp * L_temporal          (frame-to-frame consistency)

Why each matters:
  L_reconstruction: ensures hole fill matches ground truth pixel values.
                    L1 preferred over L2 — less blurring.
  L_perceptual:     VGG feature matching prevents texture collapse.
                    Without it, the network learns mean colours.
  L_adversarial:    Forces locally realistic textures. Optional for
                    small holes (3-8%) but important for quality.
  L_temporal:       Penalises flickering between adjacent frames.
                    Critical for video output — holes should fill
                    consistently across time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
  """
  L1 loss computed only inside the hole region.

  We weight hole pixels more heavily than valid pixels because:
    1. The network already sees valid pixels (they're in the input)
    2. The only unknown is what goes in the holes

  L_rec = mean(|output[hole] - target[hole]|)
        + λ_valid * mean(|output[valid] - target[valid]|)

  λ_valid = 0.1 — small penalty for disturbing valid regions.
  """

  def __init__(self, valid_weight: float = 0.1):
    super().__init__()
    self.valid_weight = valid_weight

  def forward(
    self,
    output: torch.Tensor,       # (B, 3, H, W) [0, 1]
    target: torch.Tensor,       # (B, 3, H, W) [0, 1]
    hole_mask: torch.Tensor,    # (B, 1, H, W) — 1.0 at holes
  ) -> torch.Tensor:
    hole_loss  = F.l1_loss(output * hole_mask,  target * hole_mask)
    valid_mask = 1 - hole_mask
    valid_loss = F.l1_loss(output * valid_mask, target * valid_mask)
    return hole_loss + self.valid_weight * valid_loss


# ---------------------------------------------------------------------------
# Perceptual loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
  """
  VGG-16 feature matching loss.

  Computes L1 distance between VGG features of output and target.
  Uses relu1_2, relu2_2, relu3_3 — captures texture at multiple scales.

  The VGG encoder is frozen — we never update its weights.

  L_perc = Σ_l w_l * ||φ_l(output) - φ_l(target)||_1

  where φ_l is the feature map at VGG layer l.
  """

  def __init__(self, device: str = 'cpu'):
    super().__init__()

    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
    self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
    self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
    self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3

    # Freeze — never train VGG
    for param in self.parameters():
      param.requires_grad = False

    self.weights = [1.0, 0.75, 0.5]

    # VGG normalisation (ImageNet mean/std)
    self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

  def _normalise(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std # type: ignore[union-attr]

  def forward(
    self,
    output: torch.Tensor,  # (B, 3, H, W) [0, 1]
    target: torch.Tensor,  # (B, 3, H, W) [0, 1]
  ) -> torch.Tensor:
    out_n = self._normalise(output)
    tgt_n = self._normalise(target)

    loss = torch.tensor(0.0, device=output.device)

    o1 = self.slice1(out_n);  t1 = self.slice1(tgt_n)
    o2 = self.slice2(o1);     t2 = self.slice2(t1)
    o3 = self.slice3(o2);     t3 = self.slice3(t2)

    for o, t, w in zip([o1, o2, o3], [t1, t2, t3], self.weights):
      loss = loss + w * F.l1_loss(o, t)

    return loss


# ---------------------------------------------------------------------------
# Adversarial loss — PatchGAN discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
  """
  70×70 PatchGAN discriminator.

  Classifies overlapping 70×70 patches as real or fake.
  Better than full-image discriminator for texture realism —
  each patch decision is local, so the network focuses on
  texture plausibility rather than global composition.

  Input:  (B, 3+1, H, W) — image concatenated with hole mask
          (conditioning on mask makes discrimination easier to learn)
  Output: (B, 1, H', W') — patch-level real/fake scores
  """

  def __init__(self):
    super().__init__()

    def block(in_c, out_c, stride=2, bn=True):
      layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not bn)]
      if bn:
        layers.append(nn.BatchNorm2d(out_c)) # type: ignore[union-attr]
      layers.append(nn.LeakyReLU(0.2, inplace=True)) # type: ignore[union-attr]
      return nn.Sequential(*layers)

    self.model = nn.Sequential(
      block(4,   64,  stride=2, bn=False),  # 4 = 3 (RGB) + 1 (mask)
      block(64,  128, stride=2),
      block(128, 256, stride=2),
      block(256, 512, stride=1),
      nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
    )

  def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = torch.cat([image, mask], dim=1)
    return self.model(x)


class AdversarialLoss(nn.Module):
  """
  Hinge adversarial loss for generator and discriminator.

  Generator loss:   L_adv_G = -mean(D(G(x)))
  Discriminator:    L_adv_D = mean(relu(1 - D(real)))
                             + mean(relu(1 + D(fake)))

  Hinge loss is more stable than BCE and avoids mode collapse
  better than Wasserstein for this task.
  """

  def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()

  def discriminator_loss(
    self,
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
  ) -> torch.Tensor:
    real_loss = F.relu(1.0 - real_scores).mean()
    fake_loss = F.relu(1.0 + fake_scores).mean()
    return (real_loss + fake_loss) * 0.5


# ---------------------------------------------------------------------------
# Temporal consistency loss
# ---------------------------------------------------------------------------

class TemporalLoss(nn.Module):
  """
  Penalises flickering between consecutive inpainted frames.

  For adjacent frames t and t+1:
    L_temp = ||output_t+1 - warp(output_t, flow_t→t+1)||_1

  where warp uses the same backward warping as warper.py.
  Applied only inside the hole region — valid pixels don't flicker
  because they come directly from the original image.

  At training time we approximate this with random pairs of
  synthetic hole frames from the same video sequence.
  """

  def forward(
    self,
    output_curr: torch.Tensor,  # (B, 3, H, W) current frame output
    output_prev: torch.Tensor,  # (B, 3, H, W) previous frame output
    hole_mask: torch.Tensor,    # (B, 1, H, W)
  ) -> torch.Tensor:
    # Simple frame difference inside holes — no optical flow needed
    # at training time (we use pairs from the same video)
    diff = torch.abs(output_curr - output_prev) * hole_mask
    return diff.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class InpainterLoss(nn.Module):
  """
  Combines all losses with configurable weights.

  Default weights are tuned for physics hole sizes (3-8% of image).
  For larger holes, increase λ_perc and λ_adv.
  """

  def __init__(
    self,
    lambda_rec:  float = 1.0,
    lambda_perc: float = 0.1,
    lambda_adv:  float = 0.01,
    lambda_temp: float = 0.05,
    use_adversarial: bool = True,
    device: str = 'cpu',
  ):
    super().__init__()
    self.lambda_rec  = lambda_rec
    self.lambda_perc = lambda_perc
    self.lambda_adv  = lambda_adv
    self.lambda_temp = lambda_temp
    self.use_adv     = use_adversarial

    self.rec  = ReconstructionLoss()
    self.perc = PerceptualLoss(device)
    self.temp = TemporalLoss()
    if use_adversarial:
      self.adv = AdversarialLoss()

  def forward(
    self,
    output: torch.Tensor,
    target: torch.Tensor,
    hole_mask: torch.Tensor,
    fake_scores: Optional[torch.Tensor] = None,
    output_prev: Optional[torch.Tensor] = None,
  ) -> dict[str, torch.Tensor]:
    """
    Returns dict of individual losses and total.
    Logging individual losses is essential for debugging training.
    """
    losses = {}

    losses['rec']  = self.lambda_rec  * self.rec(output, target, hole_mask)
    losses['perc'] = self.lambda_perc * self.perc(output, target)

    if self.use_adv and fake_scores is not None:
      losses['adv'] = self.lambda_adv * self.adv.generator_loss(fake_scores)

    if output_prev is not None:
      losses['temp'] = self.lambda_temp * self.temp(output, output_prev, hole_mask)

    losses['total'] = sum(losses.values())
    return losses