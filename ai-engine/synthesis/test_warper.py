"""
test_warper.py — First end-to-end test producing actual animated frames.

This is the first time the full chain runs:
  mask → physics sim → dense flow → warp → animated frames

Run from i2v/ root:
  python synthesis/test_warper.py

Output:
  synthesis/test_warp_frames/  — individual PNG frames
  synthesis/test_warp.gif      — animated GIF of the full sequence
  synthesis/test_warp_comparison.png — 4-panel comparison image
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
import imageio
from pathlib import Path

from physics.mesh import MeshBuilder
from physics.simulator import PhysicsSimulator, ForceConfig
from motion.dense_flow import compute_dense_flow, apply_boundary_mask
from synthesis.warper import backward_warp, forward_splat, blend_frames


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------

def make_test_image(H=256, W=256) -> np.ndarray:
  """Colourful striped flag — makes horizontal displacement very visible."""
  image       = np.ones((H, W, 3), dtype=np.uint8) * 200
  flag_region = np.zeros((120, 140, 3), dtype=np.uint8)
  stripe_h    = 120 // 5
  colours = [
    [220,  60,  60],
    [240, 160,  40],
    [240, 220,  50],
    [ 60, 180,  80],
    [ 60, 100, 220],
  ]
  for i, c in enumerate(colours):
    flag_region[i * stripe_h:(i + 1) * stripe_h, :] = c
  image[40:160, 60:200] = flag_region
  return image


def make_flag_mask(H=256, W=256) -> np.ndarray:
  mask = np.zeros((H, W), dtype=np.uint8)
  mask[40:160, 60:200] = 255
  return mask


# ---------------------------------------------------------------------------
# Run full pipeline for N frames
# ---------------------------------------------------------------------------

def run_pipeline(image, mask, n_frames=60):
  H, W = image.shape[:2]

  builder = MeshBuilder(
    stiffness=60.0,
    shear_stiffness=30.0,
    bend_stiffness=10.0,
    damping=0.08,
  )
  mesh  = builder.build(mask, density=0.02)
  top_y = min(p.position[1] for p in mesh.particles)

  for p in mesh.particles:
    if p.position[1] < top_y + 12:
      p.pinned = True

  forces = ForceConfig(
    gravity=np.array([0.0, 150.0]),
    wind=np.array([100.0, 0.0]),
    wind_noise_scale=0.0,
    drag=0.02,
  )
  sim = PhysicsSimulator(mesh, dt=1/30, forces=forces, substeps=10)

  frames_backward = []
  frames_forward  = []
  hole_masks      = []
  prev_frame      = image.copy()

  print(f"Rendering {n_frames} frames...")

  for f in range(n_frames):
    sim.step()

    result = compute_dense_flow(
      mask,
      sim.rest_positions,
      sim.get_displacements(),
      mesh.triangles,
      method='barycentric',
    )
    flow = apply_boundary_mask(result['barycentric'], mask, feather_radius=4)

    # Backward warp — no holes
    warped_bw = backward_warp(image, flow, mask=mask, device='cpu')
    warped_bw = blend_frames(prev_frame, warped_bw, alpha=0.9, mask=mask)
    frames_backward.append(warped_bw.copy())
    prev_frame = warped_bw

    # Forward splat — with holes
    warped_fw, hole_mask = forward_splat(image, flow, mask=mask)
    frames_forward.append(warped_fw.copy())
    hole_masks.append(hole_mask.copy())

    if f % 10 == 0:
      max_flow = np.linalg.norm(flow, axis=2).max()
      hole_pct = (hole_mask > 0).sum() / (mask > 0).sum() * 100
      print(f"  Frame {f:3d} | max_flow={max_flow:.2f}px | holes={hole_pct:.1f}%")

  return frames_backward, frames_forward, hole_masks


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(frames_backward, frames_forward, hole_masks, image, mask):
  out_dir = Path(__file__).parent / 'test_warp_frames'
  out_dir.mkdir(exist_ok=True)

  for i, frame in enumerate(frames_backward):
    cv2.imwrite(
      str(out_dir / f'backward_{i:03d}.png'),
      cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
    )

  mid = len(frames_backward) // 2

  orig = image.copy()
  cv2.putText(orig, 'Original', (10, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

  bw = frames_backward[mid].copy()
  cv2.putText(bw, f'Backward warp (f={mid})', (10, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

  fw = frames_forward[mid].copy()
  cv2.putText(fw, f'Forward splat (f={mid})', (10, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

  hole_vis = np.stack([hole_masks[mid]] * 3, axis=2)
  cv2.putText(hole_vis, f'Hole mask (f={mid})', (10, 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

  comparison = np.hstack([orig, bw, fw, hole_vis])
  cv2.imwrite(
    str(Path(__file__).parent / 'test_warp_comparison.png'),
    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
  )
  print("Saved comparison: synthesis/test_warp_comparison.png")

  gif_path = Path(__file__).parent / 'test_warp.gif'
  imageio.mimsave(str(gif_path), frames_backward, fps=30, loop=0)
  print(f"Saved GIF: synthesis/test_warp.gif")

  hole_pcts = [(h > 0).sum() / (mask > 0).sum() * 100 for h in hole_masks]
  print(f"\nHole mask stats across {len(hole_masks)} frames:")
  print(f"  Mean hole %: {np.mean(hole_pcts):.2f}%")
  print(f"  Max  hole %: {np.max(hole_pcts):.2f}%")
  print(f"  (These are the pixels the inpainter will need to fill)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  print("=== Warper Test — First Animated Frames ===\n")

  image = make_test_image()
  mask  = make_flag_mask()

  frames_bw, frames_fw, holes = run_pipeline(image, mask, n_frames=60)

  print("\nSaving outputs...")
  save_outputs(frames_bw, frames_fw, holes, image, mask)

  print("\nWhat to check:")
  print("  test_warp.gif         — flag should wave realistically")
  print("  test_warp_comparison  — 4 panels: original | backward | forward | holes")
  print("  Backward warp:  no black holes, slight blur at boundaries (expected)")
  print("  Forward splat:  black patches where object moved (to be filled by inpainter)")
  print("  Hole mask:      white = regions needing inpainting")
  print("  Pinned top row: should stay perfectly still in the GIF")