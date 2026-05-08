"""
material.py — Material type classification and parameter assignment.

Given a masked image region, this module classifies what physical material
it most likely is and returns the appropriate simulation parameters.

In the hybrid approach:
  - We use a pretrained CLIP or ViT backbone to get image embeddings
  - The classification head on top is ours (trained or rule-based to start)
  - We can begin with a heuristic classifier and upgrade to learned later

Material types and their physics:
  CLOTH   → spring-mass, low stiffness, high damping, gravity + wind
  HAIR    → spring-mass, very low stiffness, extreme damping
  FLUID   → SPH (future module), gravity + buoyancy
  SMOKE   → SPH or particle advection, strong buoyancy, turbulence
  RIGID   → rigid body dynamics, high stiffness
  FIRE    → advection-diffusion PDE (future), buoyancy dominant
  LEAF    → spring-mass, low mass, high wind sensitivity
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import cv2


# ---------------------------------------------------------------------------
# Material enum and parameters
# ---------------------------------------------------------------------------

class MaterialType(Enum):
  CLOTH  = auto()
  HAIR   = auto()
  FLUID  = auto()
  SMOKE  = auto()
  RIGID  = auto()
  FIRE   = auto()
  LEAF   = auto()
  UNKNOWN = auto()


@dataclass
class MaterialParams:
  """Physics parameters for a detected material."""
  material: MaterialType

  # Spring parameters
  stiffness: float        # k — spring constant
  shear_stiffness: float
  bend_stiffness: float
  damping: float          # c — spring damping

  # Mass
  mass_density: float     # mass per pixel² (relative units)

  # Simulation config
  substeps: int           # more = stiffer springs need more substeps
  mesh_density: float     # fraction of pixels to sample as particles

  # Force preset name (maps to forces.py presets)
  force_preset: str


# Material parameter library
MATERIAL_PARAMS: dict[MaterialType, MaterialParams] = {

  MaterialType.CLOTH: MaterialParams(
    material=MaterialType.CLOTH,
    stiffness=600.0,
    shear_stiffness=300.0,
    bend_stiffness=80.0,
    damping=0.4,
    mass_density=8e-4,
    substeps=8,
    mesh_density=0.015,
    force_preset='cloth',
  ),

  MaterialType.HAIR: MaterialParams(
    material=MaterialType.HAIR,
    stiffness=200.0,
    shear_stiffness=100.0,
    bend_stiffness=30.0,
    damping=0.7,
    mass_density=3e-4,
    substeps=6,
    mesh_density=0.02,
    force_preset='hair',
  ),

  MaterialType.LEAF: MaterialParams(
    material=MaterialType.LEAF,
    stiffness=800.0,
    shear_stiffness=400.0,
    bend_stiffness=150.0,
    damping=0.3,
    mass_density=5e-4,
    substeps=6,
    mesh_density=0.012,
    force_preset='cloth',
  ),

  MaterialType.FLUID: MaterialParams(
    material=MaterialType.FLUID,
    stiffness=0.0,        # SPH handles pressure, not springs
    shear_stiffness=0.0,
    bend_stiffness=0.0,
    damping=0.1,
    mass_density=1.2e-3,
    substeps=10,
    mesh_density=0.025,
    force_preset='water',
  ),

  MaterialType.SMOKE: MaterialParams(
    material=MaterialType.SMOKE,
    stiffness=0.0,
    shear_stiffness=0.0,
    bend_stiffness=0.0,
    damping=0.05,
    mass_density=1e-4,
    substeps=4,
    mesh_density=0.03,
    force_preset='smoke',
  ),

  MaterialType.RIGID: MaterialParams(
    material=MaterialType.RIGID,
    stiffness=5000.0,
    shear_stiffness=5000.0,
    bend_stiffness=5000.0,
    damping=0.8,
    mass_density=2e-3,
    substeps=4,
    mesh_density=0.008,
    force_preset='rigid',
  ),

  MaterialType.FIRE: MaterialParams(
    material=MaterialType.FIRE,
    stiffness=0.0,
    shear_stiffness=0.0,
    bend_stiffness=0.0,
    damping=0.02,
    mass_density=5e-5,
    substeps=4,
    mesh_density=0.03,
    force_preset='smoke',    # fire uses smoke-like forces for now
  ),

  MaterialType.UNKNOWN: MaterialParams(
    material=MaterialType.UNKNOWN,
    stiffness=400.0,
    shear_stiffness=200.0,
    bend_stiffness=60.0,
    damping=0.4,
    mass_density=6e-4,
    substeps=6,
    mesh_density=0.015,
    force_preset='cloth',   # default fallback
  ),
}


# ---------------------------------------------------------------------------
# Heuristic classifier (Phase 1 — no ML needed to start)
# ---------------------------------------------------------------------------

class HeuristicMaterialClassifier:
  """
  Rule-based material classifier using visual cues from the masked region.

  Uses:
    - Aspect ratio (tall + thin → hair)
    - Color statistics (orange/red → fire, blue → water)
    - Texture variance (smooth → fluid, high variance → cloth/hair)
    - Mask solidity (holes → non-solid → fabric/hair)
    - Edge density (many edges → cloth/hair, few → fluid)

  This is good enough to start. Replace with a CLIP-based learned
  classifier once you have labeled data.
  """

  def classify(
    self,
    image: np.ndarray,   # full image, RGB, (H, W, 3)
    mask: np.ndarray,    # binary mask, (H, W)
  ) -> MaterialType:

    features = self._extract_features(image, mask)
    return self._apply_rules(features)

  def _extract_features(self, image: np.ndarray, mask: np.ndarray) -> dict:
    binary = (mask > 0).astype(np.uint8)
    region = image[binary > 0]  # (K, 3) — pixels inside mask

    # --- Shape features ---
    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0]) if contours else (0, 0, 1, 1)
    aspect_ratio = h / (w + 1e-6)

    area = binary.sum()
    hull = cv2.convexHull(contours[0]) if contours else None
    hull_area = cv2.contourArea(hull) if hull is not None and len(hull) > 2 else area + 1
    solidity = area / (hull_area + 1e-6)   # 1.0 = convex, < 0.7 = irregular

    # --- Color features (normalized 0-1) ---
    mean_rgb = region.mean(axis=0) / 255.0 if len(region) > 0 else np.zeros(3)
    r, g, b = mean_rgb

    # Color ratios
    is_fiery    = r > 0.6 and g < 0.5 and b < 0.3     # red/orange dominant
    is_watery   = b > 0.5 and r < 0.5                  # blue dominant
    is_smoky    = abs(r - g) < 0.1 and abs(g - b) < 0.1 and r < 0.5  # gray

    # --- Texture features ---
    gray_region = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_gray = gray_region * binary
    texture_var = float(np.var(masked_gray[binary > 0])) if binary.sum() > 0 else 0.0

    # Edge density inside mask
    edges = cv2.Canny(gray_region, 50, 150)
    edge_density = float((edges * binary).sum()) / (area + 1e-6)

    return {
      'aspect_ratio': aspect_ratio,
      'solidity': solidity,
      'is_fiery': is_fiery,
      'is_watery': is_watery,
      'is_smoky': is_smoky,
      'texture_var': texture_var,
      'edge_density': edge_density,
      'area': area,
    }

  def _apply_rules(self, f: dict) -> MaterialType:
    if f['is_fiery'] and f['solidity'] < 0.8:
      return MaterialType.FIRE

    if f['is_smoky'] and f['solidity'] < 0.7:
      return MaterialType.SMOKE

    if f['is_watery']:
      return MaterialType.FLUID

    if f['aspect_ratio'] > 4.0 and f['solidity'] < 0.6:
      return MaterialType.HAIR

    if f['edge_density'] > 0.15 and f['texture_var'] > 500:
      return MaterialType.CLOTH

    if f['aspect_ratio'] < 0.5 or (f['aspect_ratio'] < 2.0 and f['texture_var'] < 200):
      return MaterialType.LEAF

    if f['solidity'] > 0.9 and f['texture_var'] < 100:
      return MaterialType.RIGID

    return MaterialType.UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_material(
  image: np.ndarray,
  mask: np.ndarray,
  override: Optional[MaterialType] = None,
) -> MaterialParams:
  """
  Classify the material in the masked region and return simulation params.

  Args:
    image:    Full RGB image (H, W, 3).
    mask:     Binary mask (H, W).
    override: Force a specific material type (for debugging/testing).

  Returns:
    MaterialParams with all physics configuration for this object.
  """
  if override is not None:
    return MATERIAL_PARAMS[override]

  classifier = HeuristicMaterialClassifier()
  material_type = classifier.classify(image, mask)
  return MATERIAL_PARAMS[material_type]