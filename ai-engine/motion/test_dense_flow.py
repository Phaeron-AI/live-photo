"""
test_dense_flow.py — End-to-end test: physics sim → dense flow → visualisation.

This is the first integration test connecting two modules:
  physics/simulator.py → motion/dense_flow.py

Run from the i2v/ root:
  python motion/test_dense_flow.py

Output:
  motion/test_flow_output.png  — side-by-side: mesh + barycentric + RBF flow
  motion/test_flow_overlay.png — flow vectors overlaid on the mask
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

from physics.mesh import MeshBuilder
from physics.simulator import PhysicsSimulator, ForceConfig
from motion.dense_flow import compute_dense_flow, flow_to_visualisation, apply_boundary_mask


# ---------------------------------------------------------------------------
# Synthetic test mask
# ---------------------------------------------------------------------------

def make_flag_mask(H=256, W=256) -> np.ndarray:
  mask = np.zeros((H, W), dtype=np.uint8)
  mask[40:160, 60:200] = 255
  return mask


def make_leaf_mask(H=256, W=256) -> np.ndarray:
  """Elliptical leaf shape — tests non-rectangular mesh."""
  mask = np.zeros((H, W), dtype=np.uint8)
  cv2.ellipse(mask, (128, 128), (80, 45), -20, 0, 360, 255, -1) # type: ignore[union-attr]
  return mask


# ---------------------------------------------------------------------------
# Run simulation for N frames, return final displacement
# ---------------------------------------------------------------------------

def simulate(mask: np.ndarray, n_frames: int = 90) -> tuple:
  builder = MeshBuilder(stiffness=60.0, shear_stiffness=30.0,
              bend_stiffness=10.0, damping=0.08)
  mesh = builder.build(mask, density=0.02)

  # Pin top row
  top_y = min(p.position[1] for p in mesh.particles)
  for p in mesh.particles:
    if p.position[1] < top_y + 12:
      p.pinned = True

  forces = ForceConfig(
    gravity=np.array([0.0, 150.0]),
    wind=np.array([100.0, 0.0]),
    wind_noise_scale=0.0,  # deterministic for testing
    drag=0.02,
  )

  sim = PhysicsSimulator(mesh, dt=1/60, forces=forces, substeps=10)

  for _ in range(n_frames):
    sim.step()

  positions = np.array([p.position for p in mesh.particles])   # current, not rest
  rest      = sim.rest_positions.copy()
  disps     = sim.get_displacements()
  triangles = mesh.triangles

  return rest, disps, triangles, mesh


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(mask_fn, mask_name: str, H=256, W=256):
  print(f"\n=== Testing with {mask_name} mask ===")
  mask = mask_fn(H, W)

  print("Running simulation (90 frames)...")
  rest, disps, triangles, mesh = simulate(mask)

  max_disp = np.linalg.norm(disps, axis=1).max()
  print(f"Max particle displacement: {max_disp:.2f}px")
  print(f"Particles: {len(rest)}, Triangles: {len(triangles)}")

  print("Computing barycentric flow...")
  result_bary = compute_dense_flow(
    mask, rest, disps, triangles,
    method='barycentric',
  )
  flow_bary = apply_boundary_mask(result_bary['barycentric'], mask, feather_radius=3)

  print("Computing RBF flow...")
  result_rbf = compute_dense_flow(
    mask, rest, disps, triangles,
    method='rbf',
    rbf_regularisation=1e-3,
  )
  flow_rbf = apply_boundary_mask(result_rbf['rbf'], mask, feather_radius=3)

  # --- Visualise ---
  vis_bary = flow_to_visualisation(flow_bary, mask)
  vis_rbf  = flow_to_visualisation(flow_rbf,  mask)

  fig, axes = plt.subplots(1, 4, figsize=(20, 5))
  fig.suptitle(f'Dense flow test — {mask_name}', fontsize=13)

  # Panel 1: rest mesh
  ax = axes[0]
  ax.set_title('Rest mesh + displacements')
  ax.imshow(mask, cmap='Greys', alpha=0.2, origin='upper')
  for spring in mesh.springs:
    if spring.spring_type == 'structural':
      p0, p1 = rest[spring.p0], rest[spring.p1]
      ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'b-', lw=0.3, alpha=0.4)
  ax.quiver(rest[:, 0], rest[:, 1],
        disps[:, 0], disps[:, 1],
        angles='xy', scale_units='xy', scale=1,
        color='red', width=0.003, alpha=0.8)
  ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal')

  # Panel 2: barycentric flow (colour wheel)
  ax = axes[1]
  ax.set_title('Barycentric flow')
  ax.imshow(vis_bary, origin='upper')
  ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal')

  # Panel 3: RBF flow
  ax = axes[2]
  ax.set_title('RBF (thin-plate spline) flow')
  ax.imshow(vis_rbf, origin='upper')
  ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal')

  # Panel 4: difference magnitude between the two methods
  ax = axes[3]
  diff = np.linalg.norm(flow_bary - flow_rbf, axis=2)
  im = ax.imshow(diff, origin='upper', cmap='hot', vmin=0)
  plt.colorbar(im, ax=ax, label='|bary - rbf| (px)')
  ax.set_title('Difference (px)')
  ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal')

  plt.tight_layout()
  out_path = os.path.join(os.path.dirname(__file__), f'test_flow_{mask_name}.png')
  plt.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close()
  print(f"Saved: {out_path}")

  # --- Metrics ---
  masked_px = (mask > 0).sum()
  bary_covered = (np.linalg.norm(flow_bary, axis=2) > 0).sum()
  rbf_covered  = (np.linalg.norm(flow_rbf,  axis=2) > 0).sum()
  print(f"Masked pixels:        {masked_px}")
  print(f"Barycentric coverage: {bary_covered} ({100*bary_covered/masked_px:.1f}%)")
  print(f"RBF coverage:         {rbf_covered}  ({100*rbf_covered/masked_px:.1f}%)")
  print(f"Mean |bary-rbf| diff: {diff[mask > 0].mean():.3f}px")
  print(f"Max  |bary-rbf| diff: {diff[mask > 0].max():.3f}px")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  os.makedirs(os.path.join(os.path.dirname(__file__)), exist_ok=True)

  run_test(make_flag_mask,  'flag',  H=256, W=256)
  run_test(make_leaf_mask,  'leaf',  H=256, W=256)

  print("\n=== All tests complete ===")
  print("Key things to check in the output images:")
  print("  1. Barycentric coverage should be ~90%+ of masked pixels")
  print("  2. RBF coverage should be 100% of masked pixels")
  print("  3. Flow colour should be consistent direction (wind = rightward = cyan/green)")
  print("  4. Difference image: hot spots at triangle edges = expected,")
  print("     hot spots in the interior = sign of RBF overfitting or bary gap")
  print("  5. Pinned top row should show near-zero flow (dark in colour panels)")