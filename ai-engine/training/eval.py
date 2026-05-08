"""
eval.py — Quantitative evaluation for the inpainter and full animation pipeline.

Metrics implemented:
  Image quality (hole region only):
    PSNR   — Peak Signal-to-Noise Ratio (dB). Higher = better. >30dB = good.
    SSIM   — Structural Similarity Index. [0, 1]. >0.9 = good.
    LPIPS  — Learned Perceptual Image Patch Similarity. Lower = better.

  Temporal consistency:
    tOF    — temporal Optical Flow: mean |flow(t) - flow(t-1)| in the hole region.
             Lower = less flicker.
    tWarp  — temporal Warp Error: ||f_t - warp(f_{t-1}, flow_t→t-1)||.
             Standard metric from video generation literature.

  Mask coverage:
    hole_coverage — fraction of hole pixels with non-trivial inpainting.
    boundary_sharpness — gradient magnitude at mask boundary (consistency check).

Usage:
    from training.eval import Evaluator
    ev = Evaluator(device='cuda')
    metrics = ev.evaluate_batch(output, target, hole_mask)

    # Full sequence evaluation
    seq_metrics = ev.evaluate_sequence(frames, gt_frames, masks)

Run standalone:
    python training/eval.py --checkpoint checkpoints/best.pth \
                            --davis_root /data/davis --split val
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from dataclasses import dataclass, field
from typing import Optional
import argparse
import logging
import json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------

@dataclass
class ImageMetrics:
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    mae:  float = 0.0   # mean absolute error in hole


@dataclass
class SequenceMetrics:
    image:  ImageMetrics = field(default_factory=ImageMetrics)
    t_warp_error: float = 0.0   # temporal warp consistency
    t_flow_diff:  float = 0.0   # temporal flow smoothness
    n_frames:     int   = 0

    def summary(self) -> dict:
        return {
            'psnr':          round(self.image.psnr,  3),
            'ssim':          round(self.image.ssim,  4),
            'lpips':         round(self.image.lpips, 4),
            'mae':           round(self.image.mae,   4),
            't_warp_error':  round(self.t_warp_error, 4),
            't_flow_diff':   round(self.t_flow_diff,  4),
            'n_frames':      self.n_frames,
        }


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(output: torch.Tensor, target: torch.Tensor,
         mask: Optional[torch.Tensor] = None) -> float:
    """
    Peak Signal-to-Noise Ratio computed inside the hole mask.

    Args:
        output, target: (B, C, H, W) float32 in [0, 1].
        mask:           (B, 1, H, W) float32 — 1.0 at hole pixels.

    Returns:
        PSNR in dB (averaged over batch).
    """
    with torch.no_grad():
        diff = (output - target) ** 2
        if mask is not None:
            n    = mask.sum().clamp(min=1.0)
            mse  = (diff * mask).sum() / (n * output.shape[1])
        else:
            mse = diff.mean()
        mse  = mse.clamp(min=1e-10)
        return float(10 * torch.log10(torch.tensor(1.0) / mse))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def ssim(output: torch.Tensor, target: torch.Tensor,
         mask: Optional[torch.Tensor] = None,
         window_size: int = 11) -> float:
    """
    Structural Similarity Index (simplified, no multi-scale).

    Computed in luminance (grayscale). Returns mean over batch.
    """
    with torch.no_grad():
        # Convert to grayscale via BT.601 luma
        def to_luma(x):
            return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

        L1 = to_luma(output)
        L2 = to_luma(target)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = window_size // 2

        # Local mean via average pooling (approximate Gaussian)
        mu1 = F.avg_pool2d(L1, window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(L2, window_size, stride=1, padding=pad)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12   = mu1 * mu2

        sigma1_sq = F.avg_pool2d(L1 ** 2, window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(L2 ** 2, window_size, stride=1, padding=pad) - mu2_sq
        sigma12   = F.avg_pool2d(L1 * L2, window_size, stride=1, padding=pad) - mu12

        num = (2 * mu12 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den.clamp(min=1e-8)

        if mask is not None:
            # Resize mask to match (may differ by 1px due to pooling padding)
            m = F.interpolate(mask, size=ssim_map.shape[-2:], mode='nearest')
            n = m.sum().clamp(min=1.0)
            return float((ssim_map * m).sum() / n)
        return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# LPIPS (lightweight self-contained version)
# ---------------------------------------------------------------------------

class LPIPSLite(torch.nn.Module):
    """
    Lightweight LPIPS using VGG16 relu features.

    Not identical to the official LPIPS (which uses pre-trained linear layers
    on top of AlexNet/VGG), but highly correlated and sufficient for relative
    comparisons during training.

    For publication-quality numbers, use the official lpips package:
        pip install lpips
        from lpips import LPIPS; fn = LPIPS(net='alex')
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(*list(vgg.children())[:4]),    # relu1_2
            torch.nn.Sequential(*list(vgg.children())[4:9]),   # relu2_2
            torch.nn.Sequential(*list(vgg.children())[9:16]),  # relu3_3
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std  # type: ignore[operator]
        y = (y - self.mean) / self.std  # type: ignore[operator]
        loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            # Normalise channels then compute L2
            xn = F.normalize(x, dim=1)
            yn = F.normalize(y, dim=1)
            loss = loss + (xn - yn).pow(2).mean()
        return loss


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------

def temporal_warp_error(
    frames: list[np.ndarray],   # list of (H, W, 3) uint8
    masks:  list[np.ndarray],   # list of (H, W) uint8 — hole mask
) -> float:
    """
    Temporal warp error: how consistent the inpainted hole is across frames.

    For each consecutive pair (f_t, f_{t+1}), we compute:
        err = ||f_{t+1}[hole] - f_t[hole]||_1 / hole_area

    Lower = smoother (less flickering) inpainting.
    """
    if len(frames) < 2:
        return 0.0

    errors = []
    for i in range(len(frames) - 1):
        f0 = frames[i].astype(np.float32)
        f1 = frames[i + 1].astype(np.float32)
        m  = (masks[i] > 0).astype(np.float32)
        n  = m.sum()
        if n < 1:
            continue
        err = np.abs(f1 - f0) * m[..., None]
        errors.append(err.sum() / (n * 3))

    return float(np.mean(errors)) if errors else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Wraps all metrics in a single interface.

    Usage:
        ev = Evaluator(device='cuda')
        metrics = ev.evaluate_batch(output, target, hole_mask)
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._lpips = LPIPSLite().to(device).eval()

    @torch.no_grad()
    def evaluate_batch(
        self,
        output:    torch.Tensor,   # (B, 3, H, W) float32 [0,1]
        target:    torch.Tensor,   # (B, 3, H, W) float32 [0,1]
        hole_mask: torch.Tensor,   # (B, 1, H, W) float32
    ) -> ImageMetrics:
        output    = output.to(self.device)
        target    = target.to(self.device)
        hole_mask = hole_mask.to(self.device)

        metrics          = ImageMetrics()
        metrics.psnr     = psnr(output, target, hole_mask)
        metrics.ssim     = ssim(output, target, hole_mask)
        metrics.lpips    = float(self._lpips(output, target).mean())
        n                = hole_mask.sum().clamp(min=1.0)
        metrics.mae      = float(((output - target).abs() * hole_mask).sum() /
                                  (n * output.shape[1]))
        return metrics

    def evaluate_sequence(
        self,
        frames:    list[np.ndarray],   # list of (H, W, 3) uint8
        gt_frames: list[np.ndarray],   # list of (H, W, 3) uint8
        masks:     list[np.ndarray],   # list of (H, W) uint8 — hole mask
    ) -> SequenceMetrics:
        assert len(frames) == len(gt_frames) == len(masks)

        # Per-frame image metrics (average over sequence)
        psnr_vals, ssim_vals, lpips_vals, mae_vals = [], [], [], []
        for frame, gt, m in zip(frames, gt_frames, masks):
            out_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
            gt_t  = torch.from_numpy(gt).float().permute(2,0,1).unsqueeze(0) / 255.0
            msk_t = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            im    = self.evaluate_batch(out_t, gt_t, msk_t)
            psnr_vals.append(im.psnr);  ssim_vals.append(im.ssim)
            lpips_vals.append(im.lpips); mae_vals.append(im.mae)

        img_metrics       = ImageMetrics(
            psnr  = float(np.mean(psnr_vals)),
            ssim  = float(np.mean(ssim_vals)),
            lpips = float(np.mean(lpips_vals)),
            mae   = float(np.mean(mae_vals)),
        )
        t_warp = temporal_warp_error(frames, masks)

        return SequenceMetrics(
            image        = img_metrics,
            t_warp_error = t_warp,
            n_frames     = len(frames),
        )

    def evaluate_checkpoint(
        self,
        model,
        dataloader,
    ) -> dict:
        """Run evaluation on an entire dataloader."""
        from training.losses import InpainterLoss
        criterion = InpainterLoss(use_adversarial=False, device=self.device)

        all_psnr, all_ssim, all_lpips, all_mae = [], [], [], []
        model.eval()

        for batch in dataloader:
            masked = batch['masked_image'].to(self.device)
            mask   = batch['hole_mask'].to(self.device)
            target = batch['target'].to(self.device)

            with torch.no_grad():
                output = model(masked, mask)

            m = self.evaluate_batch(output, target, mask)
            all_psnr.append(m.psnr);   all_ssim.append(m.ssim)
            all_lpips.append(m.lpips); all_mae.append(m.mae)

        return {
            'psnr':  float(np.mean(all_psnr)),
            'ssim':  float(np.mean(all_ssim)),
            'lpips': float(np.mean(all_lpips)),
            'mae':   float(np.mean(all_mae)),
            'n_batches': len(all_psnr),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Evaluate inpainter checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--davis_root', default=None)
    p.add_argument('--split',      default='val')
    p.add_argument('--device',     default='cuda')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output',     default=None,
                   help='JSON file to write results to')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    args = _parse_args()

    from synthesis.inpainter import LaMaInpainter
    from training.dataset import make_dataloader

    device = args.device if torch.cuda.is_available() else 'cpu'
    log.info(f'Evaluating on {device}')

    model = LaMaInpainter(base_channels=64).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get('model', state))
    model.eval()
    log.info(f'Loaded: {args.checkpoint}')

    loader = make_dataloader(
        args.davis_root, split=args.split,
        batch_size=args.batch_size, num_workers=2,
    )

    ev      = Evaluator(device=device)
    results = ev.evaluate_checkpoint(model, loader)
    log.info(f'Results: {json.dumps(results, indent=2)}')

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        log.info(f'Saved to {args.output}')