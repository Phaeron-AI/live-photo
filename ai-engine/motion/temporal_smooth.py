"""
temporal_smooth.py — Frame-level temporal consistency for animated sequences.

Problem:
  Physics simulation produces per-frame displacements that can jitter slightly
  between frames due to floating-point accumulation in the Verlet integrator and
  discretisation of the mesh. This jitter produces flickering in the final video
  that is perceptually annoying even when it's sub-pixel in magnitude.

Three-stage approach:
  1. Displacement smoothing   — Savitzky-Golay filter on the particle trajectory
                                (preserves peaks, removes high-freq noise)
  2. Flow-field smoothing     — Gaussian blur on the per-pixel flow field each frame
                                (removes spatial aliasing at triangle edges)
  3. Frame blending           — Exponential moving average across rendered frames
                                (handles residual temporal flicker)

All three stages are orthogonal and can be combined or used independently.

References:
  Savitzky & Golay (1964) — Smoothing and Differentiation of Data by
  Simplified Least Squares Procedures. Analytical Chemistry.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.signal import savgol_filter
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 1 — Particle trajectory smoothing
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """
    Accumulates particle positions across frames and applies
    Savitzky-Golay smoothing to remove high-frequency jitter.

    Usage:
        buf = TrajectoryBuffer(window=15, poly=3)
        for frame in range(n_frames):
            sim.step()
            smoothed = buf.push(sim.get_positions())  # (N, 2) smoothed positions
            displacements = smoothed - sim.rest_positions
    """

    def __init__(
        self,
        window: int = 15,   # must be odd, ≥ poly+2
        poly:   int = 3,    # polynomial order
        max_frames: int = 300,
    ):
        assert window % 2 == 1, "SG window must be odd"
        assert window > poly,   "window must be > poly"
        self.window     = window
        self.poly       = poly
        self.max_frames = max_frames
        self._history:  list[np.ndarray] = []   # list of (N, 2)

    def push(self, positions: np.ndarray) -> np.ndarray:
        """
        Append new positions and return smoothed estimate for the current frame.

        During warm-up (fewer frames than window), returns the raw positions
        to avoid edge artefacts from the SG filter.
        """
        self._history.append(positions.copy())
        if len(self._history) > self.max_frames:
            self._history.pop(0)

        if len(self._history) < self.window:
            return positions.copy()

        # Stack last `window` frames: (window, N, 2)
        recent = np.stack(self._history[-self.window:], axis=0)
        # Apply SG filter along the time axis (axis=0) for each spatial dim
        smoothed = savgol_filter(recent, self.window, self.poly, axis=0)
        # Return the middle frame of the smoothed window
        return smoothed[-1].astype(np.float64)  # type: ignore[return-value]

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Stage 2 — Flow field spatial smoothing
# ---------------------------------------------------------------------------

def smooth_flow_field(
    flow: np.ndarray,               # (H, W, 2) float32
    mask: np.ndarray,               # (H, W) uint8
    spatial_sigma: float = 1.5,     # Gaussian sigma for edge smoothing
    edge_feather:  int   = 4,       # boundary feather radius (pixels)
) -> np.ndarray:
    """
    Apply Gaussian blur inside the mask to reduce triangle-edge artefacts.

    The blur is mask-aware: we weight by the mask alpha so boundary pixels
    smoothly interpolate to zero rather than mixing background pixels in.

    Args:
        flow:          Raw dense flow field from dense_flow.py.
        mask:          Binary mask of the animated region.
        spatial_sigma: Gaussian sigma. Higher = smoother edges, less crisp motion.
        edge_feather:  Additional feathering at mask boundary.

    Returns:
        (H, W, 2) float32 smoothed flow.
    """
    if spatial_sigma <= 0:
        return flow

    # Compute a smooth alpha from the mask
    binary    = (mask > 0).astype(np.float32)
    ksize     = int(6 * spatial_sigma) | 1   # nearest odd integer
    alpha     = cv2.GaussianBlur(binary, (ksize, ksize), spatial_sigma)
    alpha     = np.clip(alpha, 0, 1)

    # Blur the flow weighted by alpha (avoids black bleeding from outside mask)
    flow_x = flow[..., 0] * alpha
    flow_y = flow[..., 1] * alpha

    blur_x = cv2.GaussianBlur(flow_x, (ksize, ksize), spatial_sigma)
    blur_y = cv2.GaussianBlur(flow_y, (ksize, ksize), spatial_sigma)
    alpha_blur = np.maximum(
        cv2.GaussianBlur(alpha, (ksize, ksize), spatial_sigma), 1e-6
    )

    smooth_x = blur_x / alpha_blur
    smooth_y = blur_y / alpha_blur

    smoothed = np.stack([smooth_x, smooth_y], axis=-1).astype(np.float32)

    # Feather at the mask boundary
    if edge_feather > 0:
        k = 2 * edge_feather + 1
        feather_alpha = cv2.GaussianBlur(binary, (k, k), edge_feather)
        smoothed = smoothed * feather_alpha[..., None]

    return smoothed


# ---------------------------------------------------------------------------
# Stage 3 — Frame-level EMA blending
# ---------------------------------------------------------------------------

class FrameBlender:
    """
    Exponential moving average blender for rendered frames.

    Tracks a running estimate of the "stable" frame and blends
    each new frame against it. Applied only inside the mask.

        frame_stable = α * frame_curr + (1-α) * frame_stable_prev

    Args:
        alpha:          EMA weight for current frame [0, 1].
                        0.85 gives ~7 frames of effective history.
                        0.95 gives ~20 frames (more ghosting, less flicker).
        mask_feather:   Radius for boundary feathering (pixels).
    """

    def __init__(self, alpha: float = 0.85, mask_feather: int = 3):
        self.alpha         = alpha
        self.mask_feather  = mask_feather
        self._stable: Optional[np.ndarray] = None

    def blend(
        self,
        frame_curr: np.ndarray,    # (H, W, 3) uint8
        mask:       np.ndarray,    # (H, W) uint8
    ) -> np.ndarray:
        if self._stable is None:
            self._stable = frame_curr.astype(np.float32)
            return frame_curr.copy()

        # EMA inside mask
        curr_f   = frame_curr.astype(np.float32)
        blended  = self.alpha * curr_f + (1 - self.alpha) * self._stable

        # Soft mask alpha for feathered boundary
        binary = (mask > 0).astype(np.float32)
        if self.mask_feather > 0:
            k     = 2 * self.mask_feather + 1
            alpha = cv2.GaussianBlur(binary, (k, k), self.mask_feather)
        else:
            alpha = binary

        # Composite: blended inside mask, original outside
        result        = blended * alpha[..., None] + curr_f * (1 - alpha[..., None])
        self._stable  = blended   # update stable reference
        return result.clip(0, 255).astype(np.uint8)

    def reset(self) -> None:
        self._stable = None


# ---------------------------------------------------------------------------
# Stage 4 — Loopback smoothing (for perfectly looping GIFs)
# ---------------------------------------------------------------------------

def make_seamless_loop(
    frames: list[np.ndarray],
    mask: np.ndarray,
    blend_frames: int = 10,
) -> list[np.ndarray]:
    """
    Cross-fade the end of the sequence into the beginning so the animation
    loops without a visible jump.

    Strategy:
        The last `blend_frames` frames are mixed with the first `blend_frames`
        frames using a raised-cosine cross-fade weight. Applied only inside
        the mask — background is unchanged.

    Args:
        frames:       List of (H, W, 3) uint8 frames.
        mask:         (H, W) uint8 binary mask.
        blend_frames: Number of frames on each end to blend.

    Returns:
        List of frames with seamless loop cross-fade applied.
    """
    n = len(frames)
    if n < 2 * blend_frames:
        return frames

    binary = (mask > 0).astype(np.float32)[..., None]

    # Raised-cosine weights for smooth perceptual transition
    t = np.linspace(0, 1, blend_frames)
    weights = 0.5 * (1 - np.cos(np.pi * t))  # 0 → 1

    result = [f.copy() for f in frames]

    # Blend last section: gradually fade toward the first few frames
    for i, w in enumerate(weights):
        idx_end   = n - blend_frames + i
        idx_start = i
        f_end     = frames[idx_end].astype(np.float32)
        f_start   = frames[idx_start].astype(np.float32)
        blended   = (1 - w) * f_end + w * f_start
        result[idx_end] = (
            blended * binary + f_end * (1 - binary)
        ).clip(0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Convenience: full temporal smoothing pipeline
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """
    Wraps all three stages into a single object for use in pipeline.py.

    Usage:
        smoother = TemporalSmoother(sg_window=11, ema_alpha=0.88)
        smoother.reset()
        for frame in simulation:
            smoothed_positions = smoother.smooth_positions(sim.get_positions())
            flow               = smoother.smooth_flow(raw_flow, mask)
            output_frame       = smoother.blend_frame(rendered, mask)
        frames = smoother.make_loop(frames, mask)
    """

    def __init__(
        self,
        sg_window:      int   = 11,
        sg_poly:        int   = 3,
        flow_sigma:     float = 1.2,
        ema_alpha:      float = 0.88,
        loop_blend:     int   = 8,
    ):
        self.traj   = TrajectoryBuffer(window=sg_window, poly=sg_poly)
        self.blender = FrameBlender(alpha=ema_alpha)
        self.flow_sigma  = flow_sigma
        self.loop_blend  = loop_blend

    def reset(self) -> None:
        self.traj.reset()
        self.blender.reset()

    def smooth_positions(self, positions: np.ndarray) -> np.ndarray:
        return self.traj.push(positions)

    def smooth_flow(self, flow: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return smooth_flow_field(flow, mask, spatial_sigma=self.flow_sigma)

    def blend_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.blender.blend(frame, mask)

    def make_loop(
        self, frames: list[np.ndarray], mask: np.ndarray
    ) -> list[np.ndarray]:
        return make_seamless_loop(frames, mask, blend_frames=self.loop_blend)