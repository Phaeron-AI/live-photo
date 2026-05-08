"""
warper.py — Image warping using dense optical flow fields.

Given an original image and a (H, W, 2) flow field from dense_flow.py,
this module produces a warped frame where each pixel has moved according
to its displacement vector.

Two warping strategies:

Strategy A — Backward warping (default, no holes)
--------------------------------------------------
For each destination pixel d, find its source location:
  source = d - flow[d]
Sample the original image at source using bilinear interpolation.

Strategy B — Forward splatting (hole-aware)
-------------------------------------------
For each source pixel s, splat its colour to destination:
  dest = s + flow[s]

These holes are exactly what the inpainter (inpainter.py) fills.

References:
  Jaderberg et al. (2015) — Spatial Transformer Networks. NeurIPS.
  Niklaus & Liu (2018) — Context-aware Synthesis for Video Frame Interpolation. CVPR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backward_warp(
    image: np.ndarray,
    flow: np.ndarray,
    mask: Optional[np.ndarray] = None,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Backward warp an image using a dense flow field.

    For each destination pixel d = (x, y):
        source = (x - flow[y,x,0], y - flow[y,x,1])
        warped[y,x] = bilinear_sample(image, source)

    Pixels outside the mask are copied unchanged from the original.

    Args:
        image:  RGB image to warp, uint8 (H, W, 3).
        flow:   Dense flow field, float32 (H, W, 2). flow[y,x] = (dx, dy).
        mask:   If provided, only warp pixels inside mask; rest copied from image.
        device: 'cpu' or 'cuda'.

    Returns:
        Warped image, uint8 (H, W, 3).
    """
    H, W = image.shape[:2]

    img_tensor  = _image_to_tensor(image, device)
    flow_tensor = _flow_to_tensor(flow, device)

    grid = _build_backward_grid(flow_tensor, H, W, device)

    warped = F.grid_sample(
        img_tensor, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    warped_np = _tensor_to_image(warped)

    if mask is not None:
        binary    = (mask > 0).astype(np.uint8)
        warped_np = (warped_np * binary[:, :, None]
                     + image * (1 - binary[:, :, None]))

    return warped_np.astype(np.uint8)


def forward_splat(
    image: np.ndarray,
    flow: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward splat an image using a dense flow field.

    For each source pixel s = (x, y):
        dest = (x + flow[y,x,0], y + flow[y,x,1])
        splat image[y,x] onto warped at dest with bilinear weights

    Args:
        image:  RGB image, uint8 (H, W, 3).
        flow:   Dense flow field, float32 (H, W, 2).
        mask:   If provided, only splat pixels inside mask.

    Returns:
        warped:    (H, W, 3) uint8 — splatted image, holes filled with 0
        hole_mask: (H, W) uint8   — 255 where holes exist, 0 elsewhere
    """
    H, W = image.shape[:2]

    colour_acc = np.zeros((H, W, 3), dtype=np.float32)
    weight_acc = np.zeros((H, W),    dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W]

    if mask is not None:
        valid = mask > 0
        ys    = ys[valid]
        xs    = xs[valid]
    else:
        ys = ys.ravel()
        xs = xs.ravel()

    dx     = flow[ys, xs, 0]
    dy     = flow[ys, xs, 1]
    dest_x = xs.astype(np.float32) + dx
    dest_y = ys.astype(np.float32) + dy

    _bilinear_splat(
        colour_acc, weight_acc,
        image, ys, xs,
        dest_x, dest_y,
        H, W,
    )

    mask_binary = (mask > 0) if mask is not None else np.ones((H, W), dtype=bool)

    hole_mask = np.zeros((H, W), dtype=np.uint8)
    hole_mask[mask_binary & (weight_acc < 1e-6)] = 255

    safe_weight = np.maximum(weight_acc, 1e-6)
    warped      = (colour_acc / safe_weight[:, :, None]).clip(0, 255).astype(np.uint8)
    warped[hole_mask > 0] = 0

    return warped, hole_mask


# ---------------------------------------------------------------------------
# Backward warp helpers
# ---------------------------------------------------------------------------

def _build_backward_grid(
    flow_tensor: torch.Tensor,
    H: int,
    W: int,
    device: str,
) -> torch.Tensor:
    """
    Build the sampling grid for grid_sample.

    grid_sample convention:
        grid[b, y, x] = (u, v) where u, v in [-1, 1]
        u = -1 → leftmost column,  u = +1 → rightmost column

    For backward warping:
        grid[y, x] = normalised(pixel(x,y) - flow(x,y))
    """
    xs             = torch.linspace(-1, 1, W, device=device)
    ys             = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    identity       = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    grid           = identity - flow_tensor
    return grid


def _flow_to_tensor(flow: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert pixel-space flow (H, W, 2) to normalised flow (1, H, W, 2).

    grid_sample works in [-1, 1] space:
        flow_norm_x = flow_px_x / (W / 2)
        flow_norm_y = flow_px_y / (H / 2)
    """
    H, W        = flow.shape[:2]
    flow_tensor = torch.from_numpy(flow).float().to(device)
    norm        = torch.tensor([W / 2.0, H / 2.0], device=device)
    flow_tensor = flow_tensor / norm
    return flow_tensor.unsqueeze(0)  # (1, H, W, 2)


def _image_to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    """Convert (H, W, 3) uint8 to (1, 3, H, W) float32 in [0, 1]."""
    tensor = (torch.from_numpy(image)
              .float()
              .permute(2, 0, 1)
              .unsqueeze(0)
              / 255.0)
    return tensor.to(device)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, H, W) float32 in [0, 1] to (H, W, 3) uint8."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Forward splat helpers
# ---------------------------------------------------------------------------

def _bilinear_splat(
    colour_acc: np.ndarray,
    weight_acc: np.ndarray,
    image: np.ndarray,
    src_y: np.ndarray,
    src_x: np.ndarray,
    dest_x: np.ndarray,
    dest_y: np.ndarray,
    H: int,
    W: int,
) -> None:
    """
    Distribute each source pixel's colour to the 4 surrounding destination
    pixels with bilinear weights.

    For destination (fx, fy):
        x0, y0 = floor(fx), floor(fy)
        x1, y1 = x0+1, y0+1
        wx1 = fx - x0,  wx0 = 1 - wx1
        wy1 = fy - y0,  wy0 = 1 - wy1

    This is the transpose of bilinear interpolation — ensures
    energy conservation.
    """
    colours = image[src_y, src_x].astype(np.float32)

    x0  = np.floor(dest_x).astype(np.int32)
    y0  = np.floor(dest_y).astype(np.int32)
    x1  = x0 + 1
    y1  = y0 + 1

    wx1 = (dest_x - x0).astype(np.float32)
    wy1 = (dest_y - y0).astype(np.float32)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    for (gx, gy, w) in [
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    ]:
        valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        gx_v  = gx[valid]
        gy_v  = gy[valid]
        w_v   = w[valid]
        c_v   = colours[valid]
        np.add.at(weight_acc, (gy_v, gx_v), w_v)
        np.add.at(colour_acc, (gy_v, gx_v), w_v[:, None] * c_v)


# ---------------------------------------------------------------------------
# Temporal smoothing — reduces flicker between frames
# ---------------------------------------------------------------------------

def blend_frames(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    alpha: float = 0.85,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Exponential moving average between consecutive frames.

    Reduces temporal flicker caused by bilinear interpolation aliasing,
    floating-point jitter in particle positions, and mesh boundary artefacts.
    Only applied inside the mask — background stays sharp.

    f_out = alpha * f_curr + (1 - alpha) * f_prev

    Args:
        alpha: Blend weight for current frame. 0.85 is a good default.
               Lower = smoother but more ghosting.

    FIX: The intermediate variable `blended` was shadowed — the EMA result
    was computed then reassigned in-place, making the intent ambiguous.
    Renamed the EMA intermediate to `ema` so the final composite is explicit.
    """
    ema = (
        alpha       * frame_curr.astype(np.float32) +
        (1 - alpha) * frame_prev.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    if mask is not None:
        binary = (mask > 0).astype(np.uint8)
        # Apply EMA inside mask; copy current frame unchanged outside
        return (ema * binary[:, :, None]
                + frame_curr * (1 - binary[:, :, None]))

    return ema