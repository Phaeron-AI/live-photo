"""
fix_mask_v2.py — Improved mask cleaning with convex hull corner recovery.

Fixes over v1:
  - Recovers missing corners using alpha-shape / convex hull blending
  - Smooths ragged boundaries with iterative closing
  - Fills the right-edge notch
  - Handles pole attachment region on the left

Usage:
  python fix_mask_v2.py mask.png clean_mask.png
"""

import sys
import cv2
import numpy as np
from pathlib import Path


def clean_mask(mask: np.ndarray) -> np.ndarray:
  h, w = mask.shape
  binary = (mask > 128).astype(np.uint8)

  # --- Step 1: flood-fill true background from corners ---
  bg        = (binary * 255).copy()
  flood_buf = np.zeros((h + 2, w + 2), dtype=np.uint8)
  for fy, fx in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
    if bg[fy, fx] == 0:
      cv2.floodFill(bg, flood_buf, (fx, fy), 255) # type: ignore
  not_bg = cv2.bitwise_not(bg)
  filled = cv2.bitwise_or(binary * 255, not_bg)

  # --- Step 2: close stripe gaps and Chakra ---
  for ksize in [(1, 35), (25, 1), (15, 15)]:
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)

  # --- Step 3: keep largest connected component ---
  n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
  if n > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    filled  = (labels == largest).astype(np.uint8) * 255

  # --- Step 4: convex hull fill to recover missing corners ---
  contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    c    = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)

    hull_mask = np.zeros_like(filled)
    cv2.fillPoly(hull_mask, [hull], 255)  # type: ignore

    # Blend: use hull only where it expands the mask by a bounded amount
    # This recovers corners without pulling in background objects
    expansion    = cv2.bitwise_and(hull_mask, cv2.bitwise_not(filled))
    kernel_check = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    near_mask    = cv2.dilate(filled, kernel_check)
    safe_expand  = cv2.bitwise_and(expansion, near_mask)
    filled       = cv2.bitwise_or(filled, safe_expand)

  # --- Step 5: final smooth — erode then dilate to clean jagged edges ---
  k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  filled   = cv2.morphologyEx(filled, cv2.MORPH_OPEN,  k_smooth)
  filled   = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k_smooth)

  # --- Step 6: flood-fill any remaining interior holes ---
  flood_buf2 = np.zeros((h + 2, w + 2), dtype=np.uint8)
  filled_copy = filled.copy()
  for fy, fx in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
    if filled_copy[fy, fx] == 0:
      cv2.floodFill(filled_copy, flood_buf2, (fx, fy), 128) # type: ignore
  filled = np.where(filled_copy == 0, 255, filled).astype(np.uint8)
  filled = np.where(filled == 128, 0, filled).astype(np.uint8)

  return filled


def main():
  if len(sys.argv) < 2:
    print("Usage: python fix_mask_v2.py input.png [output.png]")
    sys.exit(1)

  in_path  = sys.argv[1]
  out_path = sys.argv[2] if len(sys.argv) > 2 else in_path

  mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
  if mask is None:
    sys.exit(f"Cannot read: {in_path}")

  before = (mask > 128).sum()
  result = clean_mask(mask)
  after  = (result > 128).sum()

  print(f"White pixels: {before:,} → {after:,} (+{after-before:,})")
  cv2.imwrite(out_path, result)
  print(f"Saved: {out_path}")


if __name__ == '__main__':
  main()