# Live-Photo: Selective I2V Model

This is the official Repository for the Live-Photo I2V model.

Turns a still photo into an animation by running a spring-mass physics
simulation on a segmented region and warping the image accordingly.

```
photo.jpg + mask.png  →  flag_waving.gif
```

---

## How it works

```
Input image + mask
      │
      ▼
 MeshBuilder          Delaunay triangulation of the mask region
      │
      ▼
 PhysicsSimulator     Velocity Verlet spring-mass integration
      │               Time-varying forces via ForceScheduler
      ▼
 compute_dense_flow   Sparse particle displacements → dense (H,W,2) flow
      │               Savitzky-Golay trajectory smoothing
      ▼
 backward_warp        PyTorch grid_sample — bilinear, no holes
      │
      ▼
 FrameCompositor      Occlusion-aware composite onto inpainted background
      │               Optional cast shadow, EMA temporal blending
      ▼
 save_video           GIF / MP4 / AVI / MOV
```

---

## Install

```bash
pip install -r requirements.txt
```

**CUDA (recommended for inpainting):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Server / headless:**
```bash
pip install opencv-python-headless  # instead of opencv-python
```

---

## Quickstart

### 1. Select a mask interactively
```bash
python quick_select.py photo.jpg mask.png
```
Left-click regions to add them, right-click to remove.  
Press `S` to save, `Q` to quit.

Requires a SAM checkpoint. Download once:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 2. Animate

```bash
# Use a named preset
python pipeline.py --image flag.jpg --mask mask.png --preset flag --output flag.gif

# Full YAML config
python pipeline.py --image flag.jpg --mask mask.png \
    --config configs/flag.yaml --output flag.gif

# Override individual parameters without editing the YAML
python pipeline.py --image flag.jpg --mask mask.png --preset flag \
    --override physics.stiffness=1800 output.n_frames=90 \
    --output flag_stiff.gif
```

### 3. Fix a noisy mask (optional)
```bash
python fix_mask.py mask.png mask_clean.png
```

---

## Presets

| Preset | Pin edge | Best for |
|--------|----------|----------|
| `flag` | left | Flags, banners, hanging cloth |
| `hair` | top | Hair, fur, hanging threads |
| `leaf` | top | Leaves, petals, light fabric |
| `smoke`| bottom | Smoke, fire, rising mist |

---

## Configuration

Every parameter is controlled via a YAML config. Generate the starter files:

```bash
python configs/config.py --output_dir configs/
```

Key sections:

```yaml
physics:
  stiffness: 1500.0        # spring stiffness — higher = stiffer cloth
  damping: 0.35            # energy loss — higher = motion dies faster
  pin_edge: left           # top | bottom | left | right | none
  substeps: 10             # integration substeps — more = more stable

force:
  preset: flag             # flag | hair | leaf | smoke | null
  wind_speed: 120.0
  strength: 1.0            # global scale on all forces

flow:
  method: barycentric      # barycentric | rbf
  sg_window: 13            # Savitzky-Golay window (must be odd)
  feather_radius: 5        # flow field boundary feathering

synthesis:
  ema_alpha: 0.90          # temporal frame blending (0=no blend, 1=no history)
  shadow_enabled: false
  inpainter_enabled: false # set true once LaMa weights are loaded

output:
  fps: 30
  n_frames: 90
  format: gif              # gif | mp4 | avi | mov
  loop: true
```

---

## Inpainting (optional, improves background quality)

The forward-splat warper leaves holes in the background where the
object moved away from. By default these are filled by the cv2 Navier-
Stokes inpainter (fast, reasonable quality for small holes).

For better quality, load the LaMa pretrained weights:

```bash
# Download checkpoint (~200MB)
wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt
```

```python
from synthesis.inpainter import LaMaInpainter

model = LaMaInpainter()
model.load_pretrained('big-lama.pt')
model.freeze_encoder()
```

Then pass the model to `pipeline.run()` and set `synthesis.inpainter_enabled: true`
in the config.

---

## Project structure

```
ai-engine/
├── pipeline.py               # End-to-end CLI entry point
├── quick_select.py           # Interactive mask creation (SAM)
├── fix_mask.py               # Mask cleaning / hole filling
├── segmenter.py              # SAM wrapper
├── requirements.txt
│
├── configs/
│   ├── config.py             # Typed config dataclasses + YAML I/O
│   ├── flag.yaml             # Preset configs (generated)
│   ├── hair.yaml
│   ├── leaf.yaml
│   └── smoke.yaml
│
├── physics/
│   ├── mesh.py               # Delaunay triangulation → particle mesh
│   ├── simulator.py          # Velocity Verlet integrator
│   ├── material.py           # Heuristic material classifier
│   └── forces.py             # Force primitives (gravity, wind, drag, …)
│
├── motion/
│   ├── dense_flow.py         # Sparse → dense flow (barycentric + RBF)
│   ├── flow_scheduler.py     # Time-varying force profiles + ForceConfig
│   └── temporal_smooth.py    # Savitzky-Golay + EMA + seamless loop
│
├── synthesis/
│   ├── warper.py             # Backward warp + forward splat
│   ├── frame_compositor.py   # Occlusion-aware compositing + shadow
│   └── inpainter.py          # LaMa FFC inpainter (arch + inference)
│
└── training/
    ├── dataset.py            # DAVIS dataset + synthetic hole augmentation
    ├── losses.py             # Reconstruction + perceptual + adversarial + temporal
    ├── trainer.py            # Two-phase fine-tuning loop
    └── eval.py               # PSNR / SSIM / LPIPS / temporal warp error
```

---

## Training the inpainter

```bash
# Synthetic data only (no DAVIS needed — verifies the loop runs)
python training/trainer.py --device cuda --davis_root null

# With DAVIS (recommended)
# Download: https://davischallenge.org/davis2017/code.html
python training/trainer.py \
    --davis_root /data/davis \
    --pretrained big-lama.pt \
    --device cuda \
    --batch_size 8 \
    --phase1_epochs 10 \
    --phase2_epochs 10 \
    --save_dir checkpoints/inpainter
```

Evaluate a checkpoint:
```bash
python training/eval.py \
    --checkpoint checkpoints/inpainter/best.pth \
    --davis_root /data/davis \
    --split val \
    --device cuda \
    --output results.json
```

---

## Dependency graph

```
configs.config
    ↑
motion.flow_scheduler   (owns ForceConfig)
    ↑
physics.simulator       (imports ForceConfig from motion.flow_scheduler)
physics.mesh
physics.material
    ↑
motion.dense_flow
motion.temporal_smooth
    ↑
synthesis.warper
synthesis.frame_compositor
synthesis.inpainter
    ↑
pipeline.py
```

---

## Known limitations (V1)

- Material classifier is heuristic — misclassifies on unusual colours/textures.
  Override with `--override material.override=cloth`.
- Inpainter quality depends on LaMa weights. Without them, cv2 fallback fills
  holes adequately for small displacements only.
- No multi-object support. Run pipeline.py once per object and composite manually.
- GIF palette is 256 colours — use `--output out.mp4` for better quality.