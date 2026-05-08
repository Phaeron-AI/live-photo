"""
dataset.py — DAVIS dataset with synthetic physics-shaped hole augmentation.

Strategy:
  1. Load consecutive frame pairs from DAVIS video sequences
  2. Generate synthetic hole masks that mimic physics simulation holes:
     - Small (3-8% of a masked region)
     - Edge-adjacent (holes appear at object boundaries)
     - Organically shaped (use actual physics mesh boundaries)
  3. Zero out hole regions in the frame
  4. Return (masked_frame, hole_mask, original_frame) triplets

Why DAVIS?
  DAVIS contains 90 high-quality video sequences with object segmentation
  masks. This means we have ground truth for what the background looks like
  under the object — exactly what the inpainter needs to learn.

Download DAVIS:
  https://davischallenge.org/davis2017/code.html
  Or: pip install youtube-dl && scripts/download_davis.sh
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Hole mask generation — mimics physics simulation holes
# ---------------------------------------------------------------------------

class PhysicsHoleGenerator:
  """
  Generates synthetic hole masks that match the shapes produced by
  forward splatting in warper.py.

  Physics holes have distinctive properties:
    - Always adjacent to object boundary (not random interior holes)
    - Narrow and curved (follow the mesh edge)
    - Size: 3-8% of the object's masked area
    - Shape: elongated, following the direction of motion

  We approximate this by:
    1. Taking the object mask from DAVIS
    2. Eroding it slightly to get the boundary band
    3. Randomly selecting a connected sub-region of the boundary
    4. Dilating slightly to get organic shape
  """

  def __init__(
    self,
    min_hole_frac: float = 0.03,  # min hole as fraction of mask area
    max_hole_frac: float = 0.08,  # max hole as fraction of mask area
  ):
    self.min_frac = min_hole_frac
    self.max_frac = max_hole_frac

  def generate(
    self,
    object_mask: np.ndarray,   # (H, W) binary uint8 — DAVIS object mask
  ) -> np.ndarray:
    """Generate a synthetic physics hole mask."""
    H, W = object_mask.shape

    # Step 1: extract boundary band (where holes appear)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded   = cv2.erode(object_mask, kernel, iterations=2)
    boundary = object_mask - eroded  # ring around the boundary

    # Step 2: determine target hole size
    mask_area   = (object_mask > 0).sum()
    target_area = int(mask_area * np.random.uniform(self.min_frac, self.max_frac))

    # Step 3: random seed point on boundary
    boundary_pts = np.argwhere(boundary > 0)
    if len(boundary_pts) == 0:
      return np.zeros((H, W), dtype=np.uint8)

    seed = boundary_pts[np.random.randint(len(boundary_pts))]

    # Step 4: flood fill from seed with limited area
    hole = np.zeros((H, W), dtype=np.uint8)
    hole[seed[0], seed[1]] = 255

    # Grow hole by repeatedly dilating and masking to boundary+interior
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(50):  # max iterations
      hole_dilated = cv2.dilate(hole, dilate_kernel, iterations=1)
      hole = (hole_dilated & object_mask).astype(np.uint8)
      if (hole > 0).sum() >= target_area:
        break

    # Step 5: slight smoothing for organic shape
    hole = cv2.GaussianBlur(hole, (5, 5), 1)
    hole = (hole > 64).astype(np.uint8) * 255

    return hole


# ---------------------------------------------------------------------------
# DAVIS Dataset
# ---------------------------------------------------------------------------

class DAVISInpaintDataset(Dataset):
  """
  DAVIS 2017 dataset for inpainter training.

  Directory structure expected:
    davis_root/
      JPEGImages/480p/{sequence_name}/{frame_id:05d}.jpg
      Annotations/480p/{sequence_name}/{frame_id:05d}.png

  Each sample returns:
    masked_image: (3, H, W) float32 [0,1] — frame with hole zeroed
    hole_mask:    (1, H, W) float32       — 1.0 at hole pixels
    target:       (3, H, W) float32 [0,1] — original frame (ground truth)
  """

  def __init__(
    self,
    davis_root: str,
    split: str = 'train',
    image_size: Tuple[int, int] = (256, 256),
    hole_generator: Optional[PhysicsHoleGenerator] = None,
  ):
    self.root         = Path(davis_root)
    self.image_size   = image_size
    self.hole_gen     = hole_generator or PhysicsHoleGenerator()

    self.samples = self._collect_samples(split)
    print(f"DAVIS {split} set: {len(self.samples)} frame pairs")

  def _collect_samples(self, split: str) -> List[Tuple[Path, Path]]:
    """Collect (image_path, annotation_path) pairs."""
    img_root  = self.root / 'JPEGImages'  / '480p'
    ann_root  = self.root / 'Annotations' / '480p'

    # DAVIS split file
    split_file = self.root / 'ImageSets' / '2017' / f'{split}.txt'
    if split_file.exists():
      sequences = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    else:
      # Fallback: use all sequences
      sequences = sorted([d.name for d in img_root.iterdir() if d.is_dir()])

    samples = []
    for seq in sequences:
      img_dir = img_root / seq
      ann_dir = ann_root / seq
      if not img_dir.exists() or not ann_dir.exists():
        continue
      frames = sorted(img_dir.glob('*.jpg'))
      for frame in frames:
        ann = ann_dir / (frame.stem + '.png')
        if ann.exists():
          samples.append((frame, ann))

    return samples

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> dict:
    img_path, ann_path = self.samples[idx]

    # Load image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore[union-attr]
    image = cv2.resize(image, self.image_size[::-1])  # cv2 takes (W, H)

    # Load annotation mask
    ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
    ann = cv2.resize(ann, self.image_size[::-1], interpolation=cv2.INTER_NEAREST) # type: ignore[union-attr]
    ann = (ann > 0).astype(np.uint8) * 255

    # Generate synthetic hole
    hole_mask = self.hole_gen.generate(ann)

    # Apply hole to image
    masked_image         = image.copy()
    masked_image[hole_mask > 0] = 0

    # Convert to tensors
    image_t  = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    masked_t = torch.from_numpy(masked_image).float().permute(2, 0, 1) / 255.0
    hole_t   = torch.from_numpy((hole_mask > 0).astype(np.float32)).unsqueeze(0)

    return {
      'masked_image': masked_t,   # (3, H, W) [0,1] — input to inpainter
      'hole_mask':    hole_t,     # (1, H, W) — 1.0 at holes
      'target':       image_t,    # (3, H, W) [0,1] — ground truth
    }


# ---------------------------------------------------------------------------
# Synthetic fallback dataset (no DAVIS needed for initial testing)
# ---------------------------------------------------------------------------

class SyntheticHoleDataset(Dataset):
  """
  Synthetic dataset for testing the training pipeline without DAVIS.

  Generates random coloured images with synthetic holes.
  Not suitable for real training — use only to verify the
  training loop runs end to end before downloading DAVIS.
  """

  def __init__(
    self,
    size: int = 1000,
    image_size: Tuple[int, int] = (256, 256),
  ):
    self.size       = size
    self.image_size = image_size
    self.hole_gen   = PhysicsHoleGenerator()

  def __len__(self) -> int:
    return self.size

  def __getitem__(self, idx: int) -> dict:
    H, W = self.image_size

    # Random gradient image — simple but diverse enough to test
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(3):
      base  = np.random.randint(50, 200)
      grad  = np.linspace(0, np.random.randint(30, 80), W)
      image[:, :, c] = np.clip(base + grad[None, :], 0, 255)

    # Random rectangular object mask
    y1 = np.random.randint(H // 4, H // 3)
    y2 = np.random.randint(2 * H // 3, 3 * H // 4)
    x1 = np.random.randint(W // 4, W // 3)
    x2 = np.random.randint(2 * W // 3, 3 * W // 4)
    ann = np.zeros((H, W), dtype=np.uint8)
    ann[y1:y2, x1:x2] = 255

    hole_mask = self.hole_gen.generate(ann)

    masked_image = image.copy()
    masked_image[hole_mask > 0] = 0

    image_t  = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    masked_t = torch.from_numpy(masked_image).float().permute(2, 0, 1) / 255.0
    hole_t   = torch.from_numpy((hole_mask > 0).astype(np.float32)).unsqueeze(0)

    return {
      'masked_image': masked_t,
      'hole_mask':    hole_t,
      'target':       image_t,
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
  davis_root: Optional[str] = None,
  split: str = 'train',
  batch_size: int = 8,
  num_workers: int = 4,
  image_size: Tuple[int, int] = (256, 256),
  use_synthetic: bool = False,
) -> DataLoader:
  """
  Create a DataLoader for inpainter training.

  Args:
    davis_root:    Path to DAVIS dataset root. If None, uses synthetic.
    split:         'train' or 'val'.
    batch_size:    Batch size. 8 fits on 8GB VRAM at 256×256.
    num_workers:   Parallel data loading workers.
    image_size:    (H, W) to resize all frames to.
    use_synthetic: Force synthetic dataset even if davis_root provided.
  """
  if davis_root is not None and not use_synthetic:
    dataset = DAVISInpaintDataset(
      davis_root, split=split, image_size=image_size,
    )
  else:
    print("Using synthetic dataset (no DAVIS root provided)")
    dataset = SyntheticHoleDataset(
      size=1000 if split == 'train' else 100,
      image_size=image_size,
    )

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=(split == 'train'),
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
  )