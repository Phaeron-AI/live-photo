"""
pipeline.py — End-to-end: image + mask → animated video.

Usage:
    python pipeline.py --image photo.jpg --output out.gif
    python pipeline.py --image indian_flag.png --mask mask_clean.png \
        --pin left --wind right --material cloth \
        --strength 0.5 --stiffness 1500 --output flag.mp4
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import cv2
import imageio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from physics.mesh import MeshBuilder
from physics.simulator import PhysicsSimulator, ForceConfig
from physics.material import classify_material, MaterialType
from motion.dense_flow import compute_dense_flow, apply_boundary_mask
from synthesis.warper import backward_warp, blend_frames


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PARTICLES = 350
MAX_IMAGE_DIM = 512

WIND_VECTORS = {
    'right': np.array([ 1.0,  0.0]),
    'left':  np.array([-1.0,  0.0]),
    'up':    np.array([ 0.0, -1.0]),
    'down':  np.array([ 0.0,  1.0]),
    'none':  np.array([ 0.0,  0.0]),
}

MATERIAL_OVERRIDE = {
    'cloth': MaterialType.CLOTH,
    'hair':  MaterialType.HAIR,
    'leaf':  MaterialType.LEAF,
    'fluid': MaterialType.FLUID,
    'rigid': MaterialType.RIGID,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_fit(image: np.ndarray, mask: np.ndarray, max_dim: int):
    """Downscale so the longer edge <= max_dim. Returns (image, mask, scale)."""
    H, W  = image.shape[:2]
    scale = min(max_dim / max(H, W), 1.0)
    if scale == 1.0:
        return image, mask, 1.0
    new_W   = max(1, int(W * scale))
    new_H   = max(1, int(H * scale))
    image_s = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
    mask_s  = cv2.resize(mask,  (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    return image_s, mask_s, scale


def _safe_density(mask: np.ndarray, target_particles: int) -> float:
    """Back-calculate mesh density to hit target particle count."""
    area = float((mask > 0).sum())
    if area == 0:
        return 0.01
    return min(0.03, max(0.003, target_particles / area))


def _check_stability(positions: np.ndarray, rest_positions: np.ndarray,
                     max_disp_px: float = 200.0) -> bool:
    """Return False if simulation has exploded."""
    disp = np.linalg.norm(positions - rest_positions, axis=1)
    return bool(np.isfinite(disp).all() and disp.max() < max_disp_px)


def _pin_particles(mesh, pin: str) -> int:
    """Pin particles on the specified edge. Returns count pinned."""
    positions  = np.array([p.position for p in mesh.particles])
    xs         = positions[:, 0]
    ys         = positions[:, 1]
    pin_margin = 12

    predicates = {
        'top':    ys < ys.min() + pin_margin,
        'bottom': ys > ys.max() - pin_margin,
        'left':   xs < xs.min() + pin_margin,
        'right':  xs > xs.max() - pin_margin,
        'none':   np.zeros(len(mesh.particles), dtype=bool),
    }

    to_pin = predicates[pin]
    count  = 0
    for i, p in enumerate(mesh.particles):
        if to_pin[i]:
            p.pinned = True
            count   += 1
    return count


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run(
    image: np.ndarray,              # (H, W, 3) RGB uint8
    mask: np.ndarray,               # (H, W) uint8, 255 = object
    n_frames: int = 60,
    fps: int = 30,
    pin: str = 'top',
    wind: str = 'right',
    strength: float = 1.0,
    material_override: str = None,  # type: ignore[assignment]
    stiffness_override: float = None,  # type: ignore[assignment]
    damping_override: float = None,    # type: ignore[assignment]
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    Animate the masked region of `image` using physics simulation.
    Returns n_frames RGB uint8 frames at the original resolution.
    """
    H_orig, W_orig = image.shape[:2]

    # ------------------------------------------------------------------
    # 1. Resize to working resolution
    # ------------------------------------------------------------------
    image_w, mask_w, scale = _resize_to_fit(image, mask, MAX_IMAGE_DIM)
    H, W = image_w.shape[:2]
    if verbose and scale < 1.0:
        print(f"Resized to {W}x{H} (scale={scale:.2f})")

    # ------------------------------------------------------------------
    # 2. Material classification
    # ------------------------------------------------------------------
    override = MATERIAL_OVERRIDE.get(material_override) if material_override else None
    params   = classify_material(image_w, mask_w, override=override)
    if verbose:
        print(f"Material: {params.material.name}  "
              f"(stiffness={params.stiffness}, substeps={params.substeps})")

    # ------------------------------------------------------------------
    # 3. Build mesh
    #    stiffness_override / damping_override let the caller fine-tune
    #    motion quality without changing material preset globally.
    # ------------------------------------------------------------------
    stiffness      = stiffness_override if stiffness_override else params.stiffness
    shear_stiff    = (stiffness_override / 2.0) if stiffness_override else params.shear_stiffness
    bend_stiff     = (stiffness_override / 8.0) if stiffness_override else params.bend_stiffness
    damping        = damping_override   if damping_override   else params.damping

    density = _safe_density(mask_w, target_particles=MAX_PARTICLES)
    builder = MeshBuilder(
        stiffness=stiffness,
        shear_stiffness=shear_stiff,
        bend_stiffness=bend_stiff,
        damping=damping,
    )
    mesh = builder.build(mask_w, density=density)
    if verbose:
        print(f"Mesh: {len(mesh.particles)} particles, "
              f"{len(mesh.triangles)} triangles, "
              f"{len(mesh.springs)} springs")

    # ------------------------------------------------------------------
    # 4. Pin selected edge
    # ------------------------------------------------------------------
    pinned = _pin_particles(mesh, pin)
    if verbose:
        print(f"Pinned {pinned} particles ({pin} edge)")

    # ------------------------------------------------------------------
    # 5. Configure forces — scaled to working resolution
    # ------------------------------------------------------------------
    ref_dim    = 256.0
    dim_scale  = min(H, W) / ref_dim
    wind_dir   = WIND_VECTORS[wind]
    wind_force = wind_dir * 80.0 * dim_scale * strength
    gravity    = np.array([0.0, 150.0 * dim_scale * strength])

    forces = ForceConfig(
        gravity=gravity,
        wind=wind_force,
        wind_noise_scale=0.3,
        drag=0.02,
    )

    # ------------------------------------------------------------------
    # 6. Initialise simulator
    # ------------------------------------------------------------------
    sim = PhysicsSimulator(
        mesh,
        dt=0.5 / fps,
        forces=forces,
        substeps=params.substeps,
    )

    # ------------------------------------------------------------------
    # 7. Render frames
    # ------------------------------------------------------------------
    frames: list[np.ndarray] = []
    prev = image_w.copy()

    if verbose:
        print(f"Rendering {n_frames} frames at {fps}fps...")

    for f in range(n_frames):
        sim.step()

        if not _check_stability(sim.positions, sim.rest_positions):
            print(f"  WARNING: unstable at frame {f}. "
                  f"Try --strength 0.5 or --stiffness 2000.")
            last = frames[-1] if frames else image_w
            while len(frames) < n_frames:
                frames.append(
                    cv2.resize(last, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                    if scale < 1.0 else last.copy()
                )
            break

        result  = compute_dense_flow(
            mask_w,
            sim.rest_positions,
            sim.get_displacements(),
            mesh.triangles,
            method='barycentric',
        )
        flow    = apply_boundary_mask(result['barycentric'], mask_w, feather_radius=4)
        warped  = backward_warp(image_w, flow, mask=mask_w)
        blended = blend_frames(prev, warped, alpha=0.9, mask=mask_w)

        out_frame = (
            cv2.resize(blended, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            if scale < 1.0 else blended
        )
        frames.append(out_frame.copy())
        prev = blended

        if verbose and (f % 10 == 0 or f == n_frames - 1):
            max_flow = float(np.linalg.norm(flow, axis=2).max())
            print(f"  Frame {f:3d} | max_flow={max_flow:.2f}px")

    return frames


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Save RGB uint8 frames to GIF or MP4/AVI/MOV."""
    ext = Path(path).suffix.lower()
    if ext == '.gif':
        imageio.mimsave(path, frames,format='GIF', duration=int(1000 / fps), loop=0,)   # type: ignore
    elif ext in ('.mp4', '.avi', '.mov'):
        with imageio.get_writer(path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)   # type: ignore
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .gif, .mp4, .avi, or .mov.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Animate a masked object in a photo using physics simulation.'
    )
    p.add_argument('--image',     required=True,
                   help='Path to input image (JPEG, PNG, etc.)')
    p.add_argument('--mask',      default=None,
                   help='White = object to animate. Strongly recommended.')
    p.add_argument('--output',    default='output.gif',
                   help='Output path (.gif or .mp4). Default: output.gif')
    p.add_argument('--frames',    type=int,   default=60,
                   help='Number of frames (default: 60)')
    p.add_argument('--fps',       type=int,   default=30,
                   help='Frames per second (default: 30)')
    p.add_argument('--pin',       default='top',
                   choices=['top', 'bottom', 'left', 'right', 'none'],
                   help='Edge to pin still (default: top). '
                        'top=curtain, left=flag on pole, bottom=hair')
    p.add_argument('--wind',      default='right',
                   choices=['right', 'left', 'up', 'down', 'none'],
                   help='Wind direction (default: right)')
    p.add_argument('--strength',  type=float, default=1.0,
                   help='Motion strength 0.0-2.0 (default: 1.0). '
                        'Try 0.5 for subtle motion, 1.5 for dramatic.')
    p.add_argument('--stiffness', type=float, default=None,
                   help='Override spring stiffness. '
                        '600=loose cloth, 1500=flag, 3000=stiff fabric.')
    p.add_argument('--damping',   type=float, default=None,
                   help='Override damping 0.0-1.0. '
                        'Higher = motion dies out faster (default: from material).')
    p.add_argument('--material',  default=None,
                   choices=['cloth', 'hair', 'leaf', 'fluid', 'rigid'],
                   help='Override automatic material detection.')
    p.add_argument('--quiet',     action='store_true',
                   help='Suppress progress output.')
    return p.parse_args()


if __name__ == '__main__':
    args    = parse_args()
    verbose = not args.quiet

    # Load image
    raw = cv2.imread(args.image)
    if raw is None:
        sys.exit(f"Error: could not read image '{args.image}'")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Load or generate mask
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            sys.exit(f"Error: could not read mask '{args.mask}'")
        mask = (mask > 128).astype(np.uint8) * 255
    else:
        if verbose:
            print("No mask provided — using full image.")
            print("TIP: Pass --mask mask.png for better results.")
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    if verbose:
        H, W    = image.shape[:2]
        obj_pct = (mask > 0).sum() / (H * W) * 100
        print(f"Image: {W}x{H}  |  Object: {obj_pct:.1f}% of pixels")

    frames = run(
        image, mask,
        n_frames=args.frames,
        fps=args.fps,
        pin=args.pin,
        wind=args.wind,
        strength=args.strength,
        material_override=args.material,
        stiffness_override=args.stiffness,
        damping_override=args.damping,
        verbose=verbose,
    )

    save_video(frames, args.output, fps=args.fps)
    print(f"Saved {len(frames)} frames → {args.output}")