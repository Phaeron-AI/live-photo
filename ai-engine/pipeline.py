"""
pipeline.py — End-to-end: image + mask → animated video.

Integrates:
  - Config system (configs/config.py)
  - Force scheduler (motion/flow_scheduler.py)
  - Temporal smoother (motion/temporal_smooth.py)
  - Frame compositor with occlusion awareness (synthesis/frame_compositor.py)
  - Full physics + flow + warp chain

Usage:
    # Quickstart (preset)
    python pipeline.py --image flag.jpg --preset flag --output flag.gif

    # Full control via YAML
    python pipeline.py --image flag.jpg --config configs/flag.yaml --output flag.gif

    # CLI overrides on top of YAML / preset
    python pipeline.py --image flag.jpg --preset flag \\
        --override physics.stiffness=1800 output.fps=24 \\
        --output flag_stiff.gif
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import AnimationConfig
from motion.dense_flow import apply_boundary_mask, compute_dense_flow
from motion.flow_scheduler import ForceScheduler
from motion.temporal_smooth import TemporalSmoother
from physics.material import MaterialType, classify_material
from physics.mesh import MeshBuilder
from physics.simulator import PhysicsSimulator
from synthesis.frame_compositor import OcclusionMapper, build_compositor
from synthesis.warper import backward_warp

# Lazy import — inpainter weights may not be present yet.
# Only inpaint_frame is used at call sites; LaMaInpainter is referenced only
# by the type annotation on run()'s `inpainter` parameter.
try:
    from synthesis.inpainter import inpaint_frame as _inpaint_frame
    _INPAINTER_AVAILABLE = True
except ImportError:
    _INPAINTER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_fit(
    image: np.ndarray,
    mask: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    H, W  = image.shape[:2]
    scale = min(max_dim / max(H, W), 1.0)
    if scale == 1.0:
        return image, mask, 1.0
    nW = max(1, int(W * scale))
    nH = max(1, int(H * scale))
    img_s  = cv2.resize(image, (nW, nH), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask,  (nW, nH), interpolation=cv2.INTER_NEAREST)
    return img_s, mask_s, scale


def _safe_density(mask: np.ndarray, target: int) -> float:
    area = float((mask > 0).sum())
    if area == 0:
        return 0.01
    return min(0.03, max(0.003, target / area))


def _check_stability(
    positions: np.ndarray,
    rest: np.ndarray,
    max_px: float = 250.0,
) -> bool:
    d = np.linalg.norm(positions - rest, axis=1)
    return bool(np.isfinite(d).all() and d.max() < max_px)


def _pin_particles(mesh, pin_edge: str) -> int:
    positions = np.array([p.position for p in mesh.particles])
    xs, ys    = positions[:, 0], positions[:, 1]
    margin    = 12
    preds: dict[str, np.ndarray] = {
        'top':    ys < ys.min() + margin,
        'bottom': ys > ys.max() - margin,
        'left':   xs < xs.min() + margin,
        'right':  xs > xs.max() - margin,
        'none':   np.zeros(len(mesh.particles), dtype=bool),
    }
    count = 0
    for i, p in enumerate(mesh.particles):
        if preds[pin_edge][i]:
            p.pinned = True
            count   += 1
    return count


def _build_force_scheduler(cfg: AnimationConfig, dim_scale: float) -> ForceScheduler:
    fc = cfg.force
    ws = fc.wind_speed * dim_scale * fc.strength
    gy = fc.gravity_y  * dim_scale * fc.strength

    wind_vecs: dict[str, np.ndarray] = {
        'right': np.array([ 1.,  0.]),
        'left':  np.array([-1.,  0.]),
        'up':    np.array([ 0., -1.]),
        'down':  np.array([ 0.,  1.]),
        'none':  np.zeros(2),
    }
    base_wind = wind_vecs[fc.wind_direction] * ws

    preset_map = {
        'flag':  ForceScheduler.flag_in_wind,
        'hair':  ForceScheduler.hair_in_breeze,
        'leaf':  ForceScheduler.leaf_flutter,
        'smoke': ForceScheduler.smoke_rising,
    }
    if fc.preset in preset_map:
        sched = preset_map[fc.preset](wind_speed=ws)
        sched._base_gravity = np.array([0.0, gy])
        return sched

    # Manual construction (covers 'smoke', 'rigid', and None)
    sched = ForceScheduler(
        base_wind=base_wind,
        base_gravity=np.array([0.0, gy]),
        base_drag=fc.drag,
    )
    sched.add_sway(period=fc.sway_period,
                   amplitude=ws * fc.sway_amp_frac,
                   axis='x')
    sched.add_turbulence(base_amplitude=ws * fc.turbulence_amp_frac,
                         base_freq=1.8)
    if fc.gust_enabled:
        sched.add_gust(t_start=0.8, duration=0.5,
                       direction=fc.wind_direction, # type: ignore
                       peak=ws * fc.gust_peak_frac)
    return sched


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

_MATERIAL_OVERRIDE_MAP: dict[str, MaterialType] = {
    'cloth': MaterialType.CLOTH,
    'hair':  MaterialType.HAIR,
    'leaf':  MaterialType.LEAF,
    'fluid': MaterialType.FLUID,
    'rigid': MaterialType.RIGID,
    'smoke': MaterialType.SMOKE,
}


def run(
    image:     np.ndarray,        # (H, W, 3) RGB uint8
    mask:      np.ndarray,        # (H, W) uint8, 255 = object
    cfg:       AnimationConfig,
    verbose:   bool = True,
    inpainter: Any  = None,       # LaMaInpainter instance, or None
) -> list[np.ndarray]:
    """
    Animate the masked region using physics simulation.

    Returns cfg.output.n_frames RGB uint8 frames at the original resolution.
    """
    t0 = time.time()
    H_orig, W_orig = image.shape[:2]

    # ── 1. Resize to working resolution ──────────────────────────────────
    image_w, mask_w, scale = _resize_to_fit(image, mask, cfg.output.max_dim)
    H, W = image_w.shape[:2]
    if verbose and scale < 1.0:
        print(f"  Resized to {W}×{H} (scale={scale:.2f})")

    # ── 2. Material classification ────────────────────────────────────────
    mat_override = _MATERIAL_OVERRIDE_MAP.get(cfg.material.override or '')
    params = classify_material(image_w, mask_w, override=mat_override)
    if verbose:
        print(f"  Material: {params.material.name}")

    # ── 3. Build mesh ─────────────────────────────────────────────────────
    pc      = cfg.physics
    density = _safe_density(mask_w, target=pc.max_particles)
    builder = MeshBuilder(
        stiffness=pc.stiffness,
        shear_stiffness=pc.shear_stiffness,
        bend_stiffness=pc.bend_stiffness,
        damping=pc.damping,
        mass_per_area=pc.mass_density,
    )
    mesh = builder.build(mask_w, density=density)
    if verbose:
        print(f"  Mesh: {len(mesh.particles)} particles, "
              f"{len(mesh.triangles)} triangles, {len(mesh.springs)} springs")

    # ── 4. Pin selected edge ──────────────────────────────────────────────
    pinned = _pin_particles(mesh, pc.pin_edge)
    if verbose:
        print(f"  Pinned {pinned} particles ({pc.pin_edge} edge)")

    # ── 5. Force scheduler ────────────────────────────────────────────────
    dim_scale = min(H, W) / 256.0
    scheduler = _build_force_scheduler(cfg, dim_scale)

    # ── 6. Simulator ──────────────────────────────────────────────────────
    sim = PhysicsSimulator(
        mesh,
        dt=0.5 / cfg.output.fps,
        forces=scheduler.get(0.0),
        substeps=pc.substeps,
    )

    # ── 7. Compositing stack ──────────────────────────────────────────────
    sc = cfg.synthesis
    bg_ext, compositor, shadow_r = build_compositor(
        image_w, mask_w,
        shadow=sc.shadow_enabled,
        bg_darken=sc.bg_darken,
        fg_feather=sc.fg_feather,
    )
    bg_frame = bg_ext.extract_background(image_w, mask_w)

    # ── 8. Temporal smoother ──────────────────────────────────────────────
    fc = cfg.flow
    smoother = TemporalSmoother(
        sg_window=fc.sg_window,
        sg_poly=fc.sg_poly,
        flow_sigma=fc.spatial_sigma,
        ema_alpha=sc.ema_alpha,
        loop_blend=sc.loop_blend_frames,
    )
    smoother.reset()

    # ── 9. Render frames ──────────────────────────────────────────────────
    frames:  list[np.ndarray] = []
    n_frames = cfg.output.n_frames
    fps      = cfg.output.fps

    if verbose:
        print(f"  Rendering {n_frames} frames at {fps}fps...")

    for f in range(n_frames):
        t = f / fps

        sim.forces = scheduler.get(t)
        sim.step()

        smoothed_pos  = smoother.smooth_positions(sim.positions.copy())
        displacements = smoothed_pos - sim.rest_positions

        if not _check_stability(smoothed_pos, sim.rest_positions):
            if verbose:
                print(f"  WARNING: simulation unstable at frame {f}. "
                      f"Try reducing force strength or increasing stiffness.")
            last = frames[-1] if frames else image_w
            while len(frames) < n_frames:
                out = (cv2.resize(last, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                       if scale < 1.0 else last.copy())
                frames.append(out)
            break

        # Dense flow
        result = compute_dense_flow(
            mask_w, sim.rest_positions, displacements,
            mesh.triangles,
            method=fc.method,
            rbf_regularisation=fc.rbf_regularisation,
        )
        flow = result[fc.method]
        flow = smoother.smooth_flow(flow, mask_w)
        flow = apply_boundary_mask(flow, mask_w, feather_radius=fc.feather_radius)

        # Backward warp
        warped = backward_warp(image_w, flow, mask=mask_w, device=sc.warp_device)

        # Optional inpainting of forward-splat holes
        if sc.inpainter_enabled and inpainter is not None and _INPAINTER_AVAILABLE:
            from synthesis.warper import forward_splat
            _, hole_mask = forward_splat(image_w, flow, mask=mask_w)
            if (hole_mask > 0).any():
                warped = _inpaint_frame(inpainter, warped, hole_mask,
                                        device=sc.warp_device)

        # Temporal EMA blend
        blended = smoother.blend_frame(warped, mask_w)

        # Occlusion-aware composite onto inpainted background
        displaced_mask = OcclusionMapper.warp_mask(mask_w, flow)
        shadow_alpha   = shadow_r.render(displaced_mask) if shadow_r else None
        composited     = compositor.composite(blended, bg_frame,
                                              displaced_mask, shadow_alpha)

        out_frame = (cv2.resize(composited, (W_orig, H_orig),
                                interpolation=cv2.INTER_LINEAR)
                     if scale < 1.0 else composited)
        frames.append(out_frame.copy())

        if verbose and (f % 10 == 0 or f == n_frames - 1):
            max_flow = float(np.linalg.norm(flow, axis=2).max())
            print(f"    Frame {f:3d}/{n_frames} | max_flow={max_flow:.1f}px | t={t:.2f}s")

    # ── 10. Seamless loop ─────────────────────────────────────────────────
    if cfg.output.loop and len(frames) >= 2 * sc.loop_blend_frames:
        loop_mask = (cv2.resize(mask_w, (W_orig, H_orig),
                                interpolation=cv2.INTER_NEAREST)
                     if scale < 1.0 else mask_w)
        frames = smoother.make_loop(frames, loop_mask)

    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s ({len(frames)} frames)")

    return frames


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_video(
    frames: list[np.ndarray],
    path: str,
    fps: int,
    loop: bool = True,
) -> None:
    ext = Path(path).suffix.lower()
    if ext == '.gif':
        imageio.mimsave(path, frames, format='GIF', duration=int(1000 / fps), loop=0 if loop else 1)    # type: ignore
    elif ext in ('.mp4', '.avi', '.mov'):
        with imageio.get_writer(path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)   # type: ignore
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .gif, .mp4, .avi, or .mov.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Animate a masked object using physics simulation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --image flag.jpg --preset flag --output flag.gif
  python pipeline.py --image hair.jpg --preset hair --output hair.gif
  python pipeline.py --image photo.jpg --config configs/flag.yaml \\
      --override physics.stiffness=1500 output.fps=24 --output out.mp4
""",
    )
    p.add_argument('--image',    required=True, help='Input image')
    p.add_argument('--mask',     default=None,  help='Object mask (255 = object)')
    p.add_argument('--output',   default='output.gif')
    p.add_argument('--config',   default=None,  help='YAML config file')
    p.add_argument('--preset',   default=None,
                   choices=['flag', 'hair', 'leaf', 'smoke'],
                   help='Named preset (overrides --config defaults)')
    p.add_argument('--override', nargs='*', default=[],
                   metavar='key=value',
                   help='Dot-notation overrides, e.g. physics.stiffness=1500')
    p.add_argument('--quiet',    action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args    = _parse_args()
    verbose = not args.quiet

    # Load base config from preset or YAML, defaulting to AnimationConfig()
    if args.preset:
        cfg = AnimationConfig.get_preset(args.preset)
    elif args.config:
        cfg = AnimationConfig.from_yaml(args.config)
    else:
        cfg = AnimationConfig()

    # Apply dot-notation --override flags
    if args.override:
        overrides: dict[str, Any] = {}
        for kv in args.override:
            key, _, v_str = kv.partition('=')
            try:
                value: Any = int(v_str)
            except ValueError:
                try:
                    value = float(v_str)
                except ValueError:
                    value = v_str
            overrides[key] = value
        cfg = cfg.merge(overrides)

    # Sync output format with the file extension
    ext = Path(args.output).suffix.lower().lstrip('.')
    if ext in ('mp4', 'avi', 'mov', 'gif'):
        cfg = cfg.merge({'output.format': ext})

    cfg.validate()

    # Load image
    raw = cv2.imread(args.image)
    if raw is None:
        sys.exit(f"Error: could not read image '{args.image}'")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Load or synthesise mask
    if args.mask:
        raw_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            sys.exit(f"Error: could not read mask '{args.mask}'")
        mask = (raw_mask > 128).astype(np.uint8) * 255
    else:
        if verbose:
            print("No mask provided — animating full image.")
            print("TIP: Use quick_select.py to create a mask for best results.")
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    if verbose:
        H, W = image.shape[:2]
        pct  = (mask > 0).sum() / (H * W) * 100
        print(f"Image: {W}×{H} | Object: {pct:.1f}% of pixels")
        print(f"Config: '{cfg.name}'")

    frames = run(image, mask, cfg, verbose=verbose)
    save_video(frames, args.output, fps=cfg.output.fps, loop=cfg.output.loop)
    print(f"Saved {len(frames)} frames → {args.output}")