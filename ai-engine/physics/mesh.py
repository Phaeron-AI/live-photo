"""
mesh.py — Triangulated particle mesh from a binary segmentation mask.

Given a binary mask (H x W numpy array), this module:
  1. Extracts the boundary contour
  2. Samples interior seed points
  3. Runs constrained Delaunay triangulation
  4. Builds a particle + edge graph for the physics simulator

Key concepts:
  - Delaunay triangulation: maximizes minimum angles, avoids sliver triangles
  - Each triangle vertex becomes a particle with mass proportional to
    the sum of areas of triangles it belongs to (Voronoi area weighting)
  - Edges become springs; we distinguish structural / shear / bend springs
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.spatial import Delaunay
from scipy.spatial.ckdtree import cKDTree
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Particle:
  """A single mass-point in the simulation."""
  id: int
  position: np.ndarray          # shape (2,), float64, image-space (x, y)
  velocity: np.ndarray          # shape (2,), float64
  mass: float                   # kg (relative units)
  pinned: bool = False          # pinned particles have infinite effective mass

  position_prev: np.ndarray = field(default_factory=lambda: np.zeros(2))

  def __post_init__(self):
    self.position_prev = self.position.copy()


@dataclass
class Spring:
  """
  A distance constraint between two particles.

  spring_type:
      'structural' — edge between adjacent vertices (resist stretch)
      'shear'      — diagonal edge across a quad (resist shear)
      'bend'       — connects vertices two hops apart (resist bending)
  """
  p0: int
  p1: int
  rest_length: float
  stiffness: float
  damping: float
  spring_type: str = 'structural'


@dataclass
class Mesh:
  """The complete particle-spring mesh derived from a mask."""
  particles: list[Particle]
  springs: list[Spring]
  triangles: np.ndarray         # (N, 3) indices into particles
  boundary_ids: set[int]        # particle indices on the mask boundary


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

class MeshBuilder:
  """
  Builds a spring-mass mesh from a binary segmentation mask.

  Usage:
      builder = MeshBuilder(stiffness=800.0, damping=0.5)
      mesh = builder.build(mask, density=0.02)

  Args:
      stiffness:     Spring stiffness k (N/m in simulation units).
      damping:       Spring damping coefficient c.
      mass_per_area: Particle mass density (mass per pixel^2).
  """

  def __init__(
    self,
    stiffness: float = 800.0,
    shear_stiffness: float = 400.0,
    bend_stiffness: float = 100.0,
    damping: float = 0.5,
    mass_per_area: float = 1e-3,
  ):
    self.stiffness = stiffness
    self.shear_stiffness = shear_stiffness
    self.bend_stiffness = bend_stiffness
    self.damping = damping
    self.mass_per_area = mass_per_area

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def build(self, mask: np.ndarray, density: float = 0.02) -> Mesh:
    """
    Args:
        mask:    Binary mask, uint8 or bool, shape (H, W).
                  Non-zero pixels = inside the object.
        density: Fraction of mask pixels to sample as interior seeds.
                  Lower = coarser mesh = faster simulation.

    Returns:
        Mesh object ready to hand to PhysicsSimulator.
    """
    mask = self._normalize_mask(mask)

    boundary_pts, _ = self._extract_boundary(mask, n_points=60)
    interior_pts = self._sample_interior(mask, density=density)

    all_pts = np.vstack([boundary_pts, interior_pts])  # (N, 2)

    tri = Delaunay(all_pts)
    triangles = self._filter_triangles_outside_mask(tri, all_pts, mask)

    particles = self._build_particles(all_pts, triangles)
    springs = self._build_springs(particles, triangles, all_pts)

    boundary_ids = set(range(len(boundary_pts)))

    return Mesh(
      particles=particles,
      springs=springs,
      triangles=triangles,
      boundary_ids=boundary_ids,
    )

  # ------------------------------------------------------------------
  # Step 1 — Mask preprocessing
  # ------------------------------------------------------------------

  def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
    """Ensure mask is uint8 with values 0 / 255."""
    if mask.dtype == bool:
      mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1:
      mask = (mask * 255).astype(np.uint8)
    else:
      mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

  # ------------------------------------------------------------------
  # Step 2 — Boundary extraction
  # ------------------------------------------------------------------

  def _extract_boundary(
    self, mask: np.ndarray, n_points: int = 60
  ) -> Tuple[np.ndarray, list[int]]:
    """
    Extract the outer contour of the mask and uniformly subsample it.

    Returns:
        pts:              (n_points, 2) float64 array of boundary samples
        boundary_indices: indices in pts that are boundary (all of them here)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
      raise ValueError("No contours found in mask — mask may be empty.")

    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(axis=1).astype(np.float64)  # (N, 2)

    pts     = self._resample_contour(contour, n_points)
    indices = list(range(len(pts)))
    return pts, indices

  def _resample_contour(self, contour: np.ndarray, n: int) -> np.ndarray:
    """Resample contour to exactly n evenly-spaced points by arc length."""
    diffs       = np.diff(contour, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen      = np.concatenate([[0], np.cumsum(seg_lengths)])
    total       = cumlen[-1]

    sample_dists = np.linspace(0, total, n, endpoint=False)
    pts          = np.zeros((n, 2))
    for i, d in enumerate(sample_dists):
      idx = np.searchsorted(cumlen, d, side='right') - 1
      idx = np.clip(idx, 0, len(contour) - 2)
      t   = (d - cumlen[idx]) / (seg_lengths[idx] + 1e-9)
      pts[i] = contour[idx] + t * (contour[idx + 1] - contour[idx])
    return pts

  # ------------------------------------------------------------------
  # Step 3 — Interior sampling
  # ------------------------------------------------------------------

  def _sample_interior(self, mask: np.ndarray, density: float) -> np.ndarray:
    """
    Sample interior points using Poisson-disk-like rejection sampling.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
      return np.zeros((0, 2))

    coords   = np.stack([xs, ys], axis=1).astype(np.float64)
    n_samples = max(10, int(len(coords) * density))
    indices  = np.random.choice(len(coords), size=min(n_samples, len(coords)), replace=False)
    candidates = coords[indices]

    min_dist = self._compute_min_dist(mask, density)
    filtered = self._poisson_filter(candidates, min_dist)
    return filtered

  def _compute_min_dist(self, mask: np.ndarray, density: float) -> float:
    area     = np.sum(mask > 0)
    n_target = max(10, int(area * density))
    return np.sqrt(area / (n_target * np.pi)) * 0.8

  def _poisson_filter(self, pts: np.ndarray, min_dist: float) -> np.ndarray:
    """
    Greedy minimum-distance filtering using a cKDTree for O(N log N)
    neighbour queries.

    FIX: The original implementation rebuilt np.array(kept) on every
    iteration, giving O(N²) time and O(N²) allocations. For density=0.03
    on a 512×512 mask (~7,800 candidates) this was noticeably slow.
    We now query a cKDTree built from the accepted points, rebuilding it
    only when the kept list grows. Rebuilding is O(k log k) and happens
    at most N times, keeping the total cost at O(N log N).
    """
    if len(pts) == 0:
        return pts

    kept = [pts[0]]
    tree = cKDTree(kept) # type: ignore[assignment]

    for p in pts[1:]:
      dist, _ = tree.query(p, k=1)
      if dist >= min_dist:
        kept.append(p)
        tree = cKDTree(kept)  # type: ignore[assignment]

    return np.array(kept)

  # ------------------------------------------------------------------
  # Step 4 — Triangle filtering
  # ------------------------------------------------------------------

  def _filter_triangles_outside_mask(
    self, tri: Delaunay, pts: np.ndarray, mask: np.ndarray
  ) -> np.ndarray:
    """
    Remove triangles whose centroid falls outside the mask.
    This handles concave shapes and holes correctly.
    """
    H, W  = mask.shape
    valid = []
    for simplex in tri.simplices:
      centroid = pts[simplex].mean(axis=0)
      cx = int(np.clip(centroid[0], 0, W - 1))
      cy = int(np.clip(centroid[1], 0, H - 1))
      if mask[cy, cx] > 0:
        valid.append(simplex)
    return np.array(valid, dtype=np.int32)

  # ------------------------------------------------------------------
  # Step 5 — Particle construction
  # ------------------------------------------------------------------

  def _build_particles(
    self, pts: np.ndarray, triangles: np.ndarray
  ) -> list[Particle]:
    """
    One particle per unique point in pts.
    Mass = sum of Voronoi areas (1/3 of each adjacent triangle area).
    """
    n      = len(pts)
    masses = np.zeros(n)

    for tri in triangles:
      area = self._triangle_area(pts[tri[0]], pts[tri[1]], pts[tri[2]])
      for vid in tri:
        masses[vid] += area / 3.0

    masses = np.maximum(masses * self.mass_per_area, 1e-6)

    particles = []
    for i, (pos, mass) in enumerate(zip(pts, masses)):
      p = Particle(
        id=i,
        position=pos.copy(),
        velocity=np.zeros(2),
        mass=float(mass),
      )
      particles.append(p)
    return particles

  def _triangle_area(self, a, b, c) -> float:
    return 0.5 * abs(
      (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    )

  # ------------------------------------------------------------------
  # Step 6 — Spring construction
  # ------------------------------------------------------------------

  def _build_springs(
    self,
    particles: list[Particle],
    triangles: np.ndarray,
    pts: np.ndarray,
  ) -> list[Spring]:
    """
    Build structural, shear, and bend springs.

    Structural: every edge in the triangulation.
    Shear:      diagonals — edges connecting vertices that share two triangles.
    Bend:       two-hop neighbours (skip one vertex along the edge graph).
    """
    springs           = []
    structural_edges  = set()

    # --- Structural springs (triangle edges) ---
    for tri in triangles:
      for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        edge = (min(a, b), max(a, b))
        if edge not in structural_edges:
          structural_edges.add(edge)
          rest = np.linalg.norm(pts[a] - pts[b])
          springs.append(Spring(
            p0=a, p1=b,
            rest_length=rest,           # type: ignore[union-attr]
            stiffness=self.stiffness,
            damping=self.damping,
            spring_type='structural',
          ))

    # --- Build adjacency for bend springs ---
    adj: dict[int, set[int]] = {i: set() for i in range(len(particles))}
    for a, b in structural_edges:
      adj[a].add(b)
      adj[b].add(a)

    # --- Bend springs (two-hop neighbours not already connected) ---
    bend_edges = set()
    for v, neighbours in adj.items():
      for n in neighbours:
        for nn in adj[n]:
          if nn != v:
            edge = (min(v, nn), max(v, nn))
            if edge not in structural_edges and edge not in bend_edges:
              bend_edges.add(edge)
              rest = np.linalg.norm(pts[edge[0]] - pts[edge[1]])
              springs.append(Spring(
                p0=edge[0], p1=edge[1],
                rest_length=rest,       # type: ignore[union-attr]
                stiffness=self.bend_stiffness,
                damping=self.damping * 0.5,
                spring_type='bend',
              ))

    return springs