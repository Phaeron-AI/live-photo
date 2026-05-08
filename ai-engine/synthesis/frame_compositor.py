"""
frame_compositor.py — Occlusion-aware compositing of warped foreground onto background.

Problem:
  The warper produces a foreground frame where the object has moved but the
  background is still the original un-animated pixels. When the object moves,
  it may:
    1. Reveal background that was previously hidden (uncovering artefacts)
    2. Cast an approximate shadow (if requested)
    3. Leave a "ghost" of itself if the background isn't properly inpainted

This module handles these concerns with four components:

  A. BackgroundExtractor   — separates object from background using the mask
  B. OcclusionMapper       — tracks which background pixels get uncovered
  C. FrameCompositor       — alpha-composites warped fg onto bg with soft edges
  D. ShadowRenderer        — optional cast shadow for realism

References:
  Porter & Duff (1984) — Compositing Digital Images. SIGGRAPH.
  Pérez et al. (2003)  — Poisson Image Editing. SIGGRAPH.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# A. Background extractor
# ---------------------------------------------------------------------------

class BackgroundExtractor:
    """
    Separates the original image into fg (object) and bg (background).

    For inpainting-quality background:
        - If an inpainter is provided, fills the mask region on the first call
        - Otherwise falls back to simple inpainting (cv2.inpaint) as a fast substitute

    The inpainted background is cached and reused across all frames.
    """

    def __init__(self, inpaint_method: str = 'ns', inpaint_radius: int = 7):
        """
        Args:
            inpaint_method: 'ns' (Navier-Stokes, best quality) or
                            'telea' (fast marching, faster).
            inpaint_radius: Radius for the inpainting algorithm.
        """
        self._method = (cv2.INPAINT_NS if inpaint_method == 'ns'
                        else cv2.INPAINT_TELEA)
        self._radius  = inpaint_radius
        self._bg_cache: Optional[np.ndarray] = None

    def extract_background(
        self,
        image: np.ndarray,   # (H, W, 3) uint8 original image
        mask:  np.ndarray,   # (H, W) uint8 — object mask (255 = object)
    ) -> np.ndarray:
        """
        Return the inpainted background: the original image with the object
        region filled in using the surrounding context.

        Result is cached — call once per animation sequence.
        """
        if self._bg_cache is not None:
            return self._bg_cache

        # Erode the mask slightly so the inpainting has clean edges to work from
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inpaint_mask = cv2.dilate((mask > 0).astype(np.uint8) * 255, k, iterations=2)

        bg = cv2.inpaint(image, inpaint_mask, self._radius, self._method)
        self._bg_cache = bg
        return bg

    def reset(self) -> None:
        self._bg_cache = None


# ---------------------------------------------------------------------------
# B. Occlusion mapper
# ---------------------------------------------------------------------------

class OcclusionMapper:
    """
    Tracks which background pixels are uncovered as the object moves.

    For each frame, we compute:
        revealed = pixels that were under the object but are no longer
        covered  = pixels that are now under the object

    The revealed region needs to show the background (inpainted bg fills it).
    The covered region shows the object.
    """

    def __init__(self, mask: np.ndarray):
        """
        Args:
            mask: Original (rest-position) binary mask of the object.
        """
        self._original_mask = (mask > 0).astype(np.uint8)
        self._prev_mask     = self._original_mask.copy()

    def update(
        self,
        current_mask: np.ndarray,   # (H, W) uint8 — mask at current displaced position
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            revealed: (H, W) uint8 — pixels newly uncovered this frame
            covered:  (H, W) uint8 — pixels now under the object
        """
        curr     = (current_mask > 0).astype(np.uint8)
        revealed = np.clip(self._prev_mask - curr, 0, 1).astype(np.uint8) * 255
        covered  = curr * 255
        self._prev_mask = curr
        return revealed, covered

    def reset(self) -> None:
        self._prev_mask = self._original_mask.copy()

    @staticmethod
    def warp_mask(
        mask: np.ndarray,
        flow: np.ndarray,
    ) -> np.ndarray:
        """
        Warp a binary mask by the given flow field to get the displaced mask.

        Uses forward mapping: for each foreground pixel (x, y), mark (x+dx, y+dy)
        as foreground in the output mask.
        """
        H, W = mask.shape
        out  = np.zeros((H, W), dtype=np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return out

        dx = flow[ys, xs, 0]
        dy = flow[ys, xs, 1]
        nx = np.round(xs + dx).astype(np.int32)
        ny = np.round(ys + dy).astype(np.int32)

        valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        out[ny[valid], nx[valid]] = 255
        return out


# ---------------------------------------------------------------------------
# C. Frame compositor
# ---------------------------------------------------------------------------

class FrameCompositor:
    """
    Alpha-composites a warped foreground onto the background.

    The compositing equation (Porter-Duff 'over'):
        result = alpha * fg + (1 - alpha) * bg

    where alpha is derived from the (possibly displaced) mask with
    feathered edges for antialiasing.

    Args:
        fg_feather: Feathering radius at object boundary (pixels).
        bg_darken:  Optional [0, 1] darkening factor on the bg under the object
                    (simulates ambient occlusion / contact shadow).
    """

    def __init__(
        self,
        fg_feather: int   = 3,
        bg_darken:  float = 0.0,
    ):
        self.fg_feather = fg_feather
        self.bg_darken  = bg_darken

    def composite(
        self,
        fg_frame:  np.ndarray,   # (H, W, 3) uint8 — warped object (bg pixels = original)
        bg_frame:  np.ndarray,   # (H, W, 3) uint8 — inpainted background
        mask:      np.ndarray,   # (H, W) uint8     — current (possibly displaced) mask
        shadow:    Optional[np.ndarray] = None,  # (H, W) float32 shadow alpha
    ) -> np.ndarray:
        """
        Composite fg onto bg, applying shadow and feathering.

        Returns:
            (H, W, 3) uint8 composited frame.
        """
        binary = (mask > 0).astype(np.float32)

        # Feather the mask boundary
        if self.fg_feather > 0:
            k     = 2 * self.fg_feather + 1
            alpha = cv2.GaussianBlur(binary, (k, k), self.fg_feather)
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = binary

        fg_f = fg_frame.astype(np.float32)
        bg_f = bg_frame.astype(np.float32)

        # Apply ambient occlusion darkening to the background in the object region
        if self.bg_darken > 0:
            bg_f = bg_f * (1.0 - self.bg_darken * alpha[..., None])

        # Apply shadow onto bg (before compositing the fg on top)
        if shadow is not None:
            bg_f = bg_f * (1.0 - shadow[..., None] * 0.5)

        result = alpha[..., None] * fg_f + (1 - alpha[..., None]) * bg_f
        return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# D. Shadow renderer
# ---------------------------------------------------------------------------

class ShadowRenderer:
    """
    Renders a directional cast shadow from the animated object.

    Strategy:
        1. Offset the displaced mask in the shadow direction
        2. Blur to soften the shadow (simulates penumbra)
        3. Return a float32 alpha mask for compositing

    This is a simple planar shadow — suitable for flat backgrounds.
    Not physically accurate for 3D scenes.

    Args:
        direction:  (dx, dy) shadow offset in pixels.
                    Positive x = shadow to the right.
        blur_sigma: Shadow softness.
        opacity:    Shadow darkness [0, 1].
    """

    def __init__(
        self,
        direction:  Tuple[float, float] = (6.0, 8.0),
        blur_sigma: float = 8.0,
        opacity:    float = 0.4,
    ):
        self.dx         = direction[0]
        self.dy         = direction[1]
        self.blur_sigma = blur_sigma
        self.opacity    = opacity

    def render(
        self,
        mask: np.ndarray,   # (H, W) uint8 — current displaced mask
    ) -> np.ndarray:
        """
        Returns:
            (H, W) float32 shadow alpha, where 1.0 = fully shadowed.
        """
        H, W = mask.shape

        # Shift the mask in the shadow direction
        M  = np.float32([[1, 0, self.dx], [0, 1, self.dy]]) # type: ignore
        shadow_mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) # type: ignore

        # Blur for penumbra
        if self.blur_sigma > 0:
            k = int(6 * self.blur_sigma) | 1
            shadow_mask = cv2.GaussianBlur(
                shadow_mask.astype(np.float32), (k, k), self.blur_sigma
            )
        else:
            shadow_mask = shadow_mask.astype(np.float32)

        # Normalise to [0, 1] and apply opacity
        shadow_alpha = np.clip(shadow_mask / 255.0, 0, 1) * self.opacity
        return shadow_alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def build_compositor(
    image: np.ndarray,
    mask:  np.ndarray,
    shadow:     bool  = False,
    bg_darken:  float = 0.0,
    fg_feather: int   = 3,
) -> Tuple[BackgroundExtractor, FrameCompositor, Optional[ShadowRenderer]]:
    """
    Create and initialise the compositing stack for one animation sequence.

    Returns:
        (bg_extractor, compositor, shadow_renderer or None)

    Usage:
        bg_ext, comp, shadow_r = build_compositor(image, mask, shadow=True)
        bg = bg_ext.extract_background(image, mask)

        for frame in frames:
            shadow_alpha = shadow_r.render(displaced_mask) if shadow_r else None
            out = comp.composite(warped_frame, bg, displaced_mask, shadow_alpha)
    """
    bg_ext  = BackgroundExtractor()
    comp    = FrameCompositor(fg_feather=fg_feather, bg_darken=bg_darken)
    shadow_r = ShadowRenderer() if shadow else None
    return bg_ext, comp, shadow_r