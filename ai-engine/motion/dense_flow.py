"""
dense_flow.py — Sparse particle displacements → dense (H, W, 2) optical flow field.

The physics simulator gives us displacement vectors at ~200-300 particle positions.
The frame synthesizer needs a displacement at every pixel inside the mask.
This module bridges that gap using two strategies:

Strategy A — Barycentric interpolation (fast, O(pixels))
---------------------------------------------------------
We already have a Delaunay triangulation of the mask from mesh.py.
For every pixel inside the mask:
  1. Find which triangle it belongs to (point-in-triangle test)
  2. Compute its barycentric coordinates (λ₀, λ₁, λ₂) w.r.t. the triangle vertices
  3. Interpolate displacement: d = λ₀·d₀ + λ₁·d₁ + λ₂·d₂

This is exact within each triangle and C⁰ continuous across edges.
Use for: real-time preview, per-frame during training.

Strategy B — RBF thin-plate spline (smooth, O(N²) solve + O(pixels) eval)
---------------------------------------------------------------------------
Fits a smooth function through all particle displacements using
Radial Basis Functions with the thin-plate spline kernel φ(r) = r²·log(r).

Solve once:  [φ(‖xᵢ-xⱼ‖)]·w = d   →  w (N×2 weight vector)
Eval per px: f(x) = Σᵢ wᵢ·φ(‖x-xᵢ‖)

C∞ smooth everywhere, handles particles near boundaries better than barycentric.
Use for: final render quality, when smoothness at triangle edges matters.

The public API is identical for both — swap method='barycentric' ↔ method='rbf'.

References:
  Bookstein (1989) — Principal Warps: Thin-Plate Splines and the
  decomposition of deformations. IEEE TPAMI.

  Sibson (1980) — A vector identity for the Dirichlet tessellation.
  (Barycentric interpolation in Delaunay triangulations.)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from typing import Literal, Optional
import cv2

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dense_flow(
    mask: np.ndarray,                   # (H, W) binary mask, uint8 or bool
    particle_positions: np.ndarray,     # (N, 2) rest positions, image-space (x, y)
    particle_displacements: np.ndarray, # (N, 2) displacement vectors from simulator
    triangles: np.ndarray,              # (T, 3) triangle indices into particle_positions
    method: Literal['barycentric', 'rbf', 'both'] = 'barycentric',
    rbf_regularisation: float = 1e-3,   # ridge term for RBF system stability
) -> dict[str, np.ndarray]:
    """
    Convert sparse particle displacements to a dense optical flow field.

    Args:
        mask:                   Binary mask of the animated object.
        particle_positions:     Rest positions of particles — (N, 2), (x, y) in px.
        particle_displacements: Per-particle displacement vectors — (N, 2).
        triangles:              Delaunay triangle indices — (T, 3).
        method:                 'barycentric', 'rbf', or 'both'.
        rbf_regularisation:     Ridge regularisation for RBF solve. Higher = smoother
                                but less faithful to particle positions.

    Returns:
        dict with keys based on method:
          'barycentric': (H, W, 2) float32 flow field
          'rbf':         (H, W, 2) float32 flow field
        Flow is zero outside the mask in both cases.
    """
    mask = _normalise_mask(mask)
    H, W = mask.shape

    results = {}

    if method in ('barycentric', 'both'):
        results['barycentric'] = _barycentric_flow(
            mask, H, W,
            particle_positions,
            particle_displacements,
            triangles,
        )

    if method in ('rbf', 'both'):
        results['rbf'] = _rbf_flow(
            mask, H, W,
            particle_positions,
            particle_displacements,
            rbf_regularisation,
        )

    return results


def flow_to_visualisation(flow: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W, 2) flow field to an HSV colour visualisation (H, W, 3) uint8.

    Convention: hue = direction, saturation = 1, value = magnitude (normalised).
    Pixels outside the mask are black.

    Useful for debugging — sanity-check that flow direction and magnitude
    match what the physics simulator is producing.
    """
    H, W = flow.shape[:2]
    magnitude = np.linalg.norm(flow, axis=2)            # (H, W)
    angle     = np.arctan2(flow[..., 1], flow[..., 0])  # (H, W) radians

    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # [0, 179] for OpenCV
    max_mag = magnitude.max() + 1e-8
    val = (magnitude / max_mag * 255).astype(np.uint8)
    sat = np.full((H, W), 220, dtype=np.uint8)

    hsv = np.stack([hue, sat, val], axis=2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    binary = (mask > 0).astype(np.uint8)
    rgb    = rgb * binary[:, :, None]
    return rgb


# ---------------------------------------------------------------------------
# Strategy A — Barycentric interpolation
# ---------------------------------------------------------------------------

def _barycentric_flow(
    mask: np.ndarray,
    H: int, W: int,
    positions: np.ndarray,      # (N, 2) — note: (x, y) = (col, row)
    displacements: np.ndarray,  # (N, 2)
    triangles: np.ndarray,      # (T, 3) — mesh triangles from mesh.py
) -> np.ndarray:
    """
    Interpolate displacements at every masked pixel using barycentric coords.

    Mathematics:
        For a point p inside triangle (v0, v1, v2):

        [v1-v0 | v2-v0] · [λ1, λ2]ᵀ = p - v0

        Solve for λ1, λ2. Then λ0 = 1 - λ1 - λ2.
        Displacement: d(p) = λ0·d0 + λ1·d1 + λ2·d2

        Point is inside triangle iff λ0, λ1, λ2 ≥ 0.

    We vectorise this over all masked pixels simultaneously.
    For each triangle, we find all pixels inside it and compute their
    barycentric coordinates in one numpy operation.

    Implementation note:
        scipy's Delaunay.find_simplex and .transform are computed together at
        construction time and are permanently coupled — patching .simplices after
        construction causes find_simplex to return indices into the *original*
        triangulation while transform has rows only for the *filtered* set,
        producing the IndexError seen in practice.

        We avoid this entirely by precomputing the affine inverse transforms for
        the filtered triangles ourselves and doing the point-in-triangle lookup
        via a vectorised barycentric test, triangle by triangle. Each triangle
        processes all M pixels in one numpy broadcast, so the total work is
        O(T × M) elementwise ops — fast in practice because T is small (~few
        hundred) and the inner loop is pure numpy with no Python overhead.
    """
    flow = np.zeros((H, W, 2), dtype=np.float32)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return flow

    pixels = np.stack([xs, ys], axis=1).astype(np.float64)  # (M, 2)
    M = len(pixels)

    triangles = triangles.astype(np.int32)
    T = len(triangles)
    if T == 0:
        return flow

    # assigned[i] = index of the triangle that owns pixel i (-1 = none)
    assigned   = np.full(M, -1, dtype=np.int32)
    # bary[:,0..2] = barycentric coords for each pixel in its owning triangle
    bary_final = np.zeros((M, 3), dtype=np.float64)

    for t_idx in range(T):
        tri = triangles[t_idx]
        v0, v1, v2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]

        # Affine map: solve [v1-v0 | v2-v0] * [lam1, lam2]^T = p - v0
        # Precompute 2×2 inverse for this triangle
        e1 = v1 - v0   # (2,)
        e2 = v2 - v0   # (2,)
        denom = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(denom) < 1e-10:          # degenerate (zero-area) triangle
            continue

        inv_denom = 1.0 / denom
        # Rows of the 2×2 inverse of [e1 | e2]:
        # inv = (1/det) * [[e2[1], -e2[0]], [-e1[1], e1[0]]]
        dp = pixels - v0                # (M, 2)
        lam1 = ( dp[:, 0] * e2[1] - dp[:, 1] * e2[0]) * inv_denom  # (M,)
        lam2 = (-dp[:, 0] * e1[1] + dp[:, 1] * e1[0]) * inv_denom  # (M,)
        lam0 = 1.0 - lam1 - lam2                                     # (M,)

        # A pixel belongs to this triangle iff all coords >= 0
        inside = (lam0 >= 0) & (lam1 >= 0) & (lam2 >= 0) & (assigned < 0)

        assigned[inside]      = t_idx
        bary_final[inside, 0] = lam0[inside]
        bary_final[inside, 1] = lam1[inside]
        bary_final[inside, 2] = lam2[inside]

    # Interpolate displacements for all assigned pixels
    hit = assigned >= 0
    if not hit.any():
        return flow

    hit_tris  = assigned[hit]                    # (K,)
    hit_bary  = bary_final[hit]                  # (K, 3)
    hit_verts = triangles[hit_tris]              # (K, 3) — vertex indices

    v0i, v1i, v2i = hit_verts[:, 0], hit_verts[:, 1], hit_verts[:, 2]

    d = (hit_bary[:, 0:1] * displacements[v0i] +
         hit_bary[:, 1:2] * displacements[v1i] +
         hit_bary[:, 2:3] * displacements[v2i])  # (K, 2)

    flow[ys[hit], xs[hit]] = d.astype(np.float32)
    return flow


# ---------------------------------------------------------------------------
# Strategy B — RBF thin-plate spline interpolation
# ---------------------------------------------------------------------------

# Chunk size for RBF query evaluation.
# Keeps the (chunk, N, 2) intermediate array below ~100 MB even for large images.
_RBF_CHUNK_SIZE = 4096


def _rbf_flow(
    mask: np.ndarray,
    H: int, W: int,
    positions: np.ndarray,      # (N, 2) control points
    displacements: np.ndarray,  # (N, 2) target values at control points
    regularisation: float,
) -> np.ndarray:
    """
    Fit a thin-plate spline through particle displacements and evaluate at all pixels.

    Thin-plate spline kernel: φ(r) = r² · log(r),  r = ‖xᵢ - xⱼ‖

    System to solve:
        [Φ + λI] · w = d

    where Φᵢⱼ = φ(‖xᵢ - xⱼ‖), λ is regularisation, d is the displacement matrix.

    After solving for weights w (N×2), evaluate at query point x:
        f(x) = Σᵢ wᵢ · φ(‖x - xᵢ‖)

    FIX: Previously the full (M, N, 2) intermediate array was materialised at
    once. For a 512×512 image with 300 particles M ≈ 262k, producing a ~1.2 GB
    array that silently OOMs. Evaluation is now chunked in slices of
    _RBF_CHUNK_SIZE rows so peak extra memory is O(chunk × N) ≈ a few MB.
    """
    flow = np.zeros((H, W, 2), dtype=np.float32)

    N = len(positions)
    if N == 0:
        return flow

    # --- Normalise coordinates to [0, 1] for numerical stability ---
    pos_min   = positions.min(axis=0)
    pos_max   = positions.max(axis=0)
    pos_range = pos_max - pos_min + 1e-8
    pos_norm  = (positions - pos_min) / pos_range  # (N, 2)

    # --- Build kernel matrix Φ (N × N) ---
    diff = pos_norm[:, None, :] - pos_norm[None, :, :]  # (N, N, 2)
    r    = np.linalg.norm(diff, axis=2)                  # (N, N)
    Phi  = _tps_kernel(r)
    Phi += regularisation * np.eye(N)

    # --- Solve for weights ---
    try:
        w = solve(Phi, displacements, assume_a='sym')   # (N, 2)
    except Exception:
        w, _, _, _ = np.linalg.lstsq(Phi, displacements, rcond=None)

    # --- Evaluate at all masked pixels, chunked to cap memory usage ---
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return flow

    query      = np.stack([xs, ys], axis=1).astype(np.float64)  # (M, 2)
    query_norm = (query - pos_min) / pos_range                   # (M, 2)

    M = len(query_norm)
    for start in range(0, M, _RBF_CHUNK_SIZE):
        end        = min(start + _RBF_CHUNK_SIZE, M)
        chunk_norm = query_norm[start:end]              # (C, 2)

        diff_q = chunk_norm[:, None, :] - pos_norm[None, :, :]  # (C, N, 2)
        r_q    = np.linalg.norm(diff_q, axis=2)                  # (C, N)
        Phi_q  = _tps_kernel(r_q)                                 # (C, N)

        d_chunk = Phi_q @ w                             # (C, 2)

        flow[ys[start:end], xs[start:end]] = d_chunk.astype(np.float32)

    return flow


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    """
    Thin-plate spline kernel: φ(r) = r² · log(r)

    Numerically: φ(0) = 0 (L'Hôpital's rule: r²·log(r) → 0 as r → 0)
    We handle r=0 explicitly to avoid log(0) = -inf.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(r > 0, r * r * np.log(r), 0.0)
    return result


# ---------------------------------------------------------------------------
# Boundary handling — zero flow outside mask, blend at edges
# ---------------------------------------------------------------------------

def apply_boundary_mask(
    flow: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 3,
) -> np.ndarray:
    """
    Enforce zero flow outside the mask and feather the boundary to avoid
    hard edges in the warped output.

    Args:
        flow:           (H, W, 2) raw flow field.
        mask:           (H, W) binary mask.
        feather_radius: Gaussian blur radius for boundary softening (pixels).

    Returns:
        (H, W, 2) masked and feathered flow field.
    """
    binary = (mask > 0).astype(np.float32)

    if feather_radius > 0:
        kernel_size = 2 * feather_radius + 1
        alpha       = cv2.GaussianBlur(binary, (kernel_size, kernel_size), feather_radius)
        alpha       = np.clip(alpha, 0, 1)
    else:
        alpha = binary

    return flow * alpha[:, :, None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    if mask.max() <= 1:
        return (mask * 255).astype(np.uint8)
    return mask.astype(np.uint8)