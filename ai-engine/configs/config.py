"""
config.py — Centralised configuration with validation, YAML I/O, and presets.

Motivation:
  Configuration currently lives as scattered argparse args in pipeline.py and
  hardcoded constants across multiple modules. This makes experimentation
  painful and reproducibility impossible (you can't replay a run from its CLI
  invocation alone).

  This module provides:
    - Typed, validated config dataclasses for every subsystem
    - YAML serialisation / deserialisation
    - Named presets (flag, hair, leaf, smoke, …)
    - A merge function so CLI args can override YAML defaults

Usage:
    # Load from YAML
    cfg = AnimationConfig.from_yaml('configs/flag.yaml')

    # Override from CLI
    cfg = cfg.merge({'physics.stiffness': 1500, 'output.fps': 24})

    # Use in pipeline
    from configs.config import AnimationConfig
    frames = run(image, mask, cfg)

    # Save for reproducibility
    cfg.to_yaml('runs/experiment_01.yaml')
"""

from __future__ import annotations

import yaml
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Literal


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """Controls the spring-mass physics simulation."""
    stiffness:       float = 800.0
    shear_stiffness: float = 400.0
    bend_stiffness:  float = 100.0
    damping:         float = 0.4
    mass_density:    float = 8e-4
    substeps:        int   = 8
    mesh_density:    float = 0.015
    pin_edge:        Literal['top', 'bottom', 'left', 'right', 'none'] = 'top'
    max_particles:   int   = 350

    def validate(self) -> None:
        assert self.stiffness >= 0, "stiffness must be non-negative (0 = particle-based material)"
        assert 0.0 <= self.damping <= 1.0, "damping must be in [0, 1]"
        assert self.substeps >= 1, "substeps must be >= 1"
        assert 0.001 <= self.mesh_density <= 0.1


@dataclass
class ForceConfig:
    """Controls the force scheduler applied to the simulation."""
    preset:        Optional[Literal['flag', 'hair', 'leaf', 'smoke', 'rigid']] = None
    wind_direction: Literal['right', 'left', 'up', 'down', 'none'] = 'right'
    wind_speed:    float = 100.0
    gravity_y:     float = 150.0
    strength:      float = 1.0
    drag:          float = 0.02
    sway_period:   float = 2.2
    sway_amp_frac: float = 0.35    # fraction of wind_speed
    turbulence_amp_frac: float = 0.15
    gust_enabled:  bool  = True
    gust_peak_frac: float = 2.0


@dataclass
class FlowConfig:
    """Controls the dense optical flow synthesis."""
    method:           Literal['barycentric', 'rbf'] = 'barycentric'
    rbf_regularisation: float = 1e-3
    feather_radius:   int   = 4
    sg_window:        int   = 11   # Savitzky-Golay window for trajectory smoothing
    sg_poly:          int   = 3
    spatial_sigma:    float = 1.2  # flow field spatial blur sigma


@dataclass
class SynthesisConfig:
    """Controls the warping and compositing."""
    warp_device:      Literal['cpu', 'cuda'] = 'cpu'
    ema_alpha:        float = 0.88       # frame-level EMA blending
    inpainter_enabled: bool = False      # set True when weights available
    shadow_enabled:   bool = False
    bg_darken:        float = 0.0
    fg_feather:       int  = 3
    loop_blend_frames: int = 8


@dataclass
class OutputConfig:
    """Controls the video output format."""
    fps:         int   = 30
    n_frames:    int   = 60
    format:      Literal['gif', 'mp4', 'avi', 'mov'] = 'gif'
    max_dim:     int   = 512          # working resolution (longest edge)
    quality:     int   = 8            # MP4/AVI CRF-like quality
    loop:        bool  = True         # for GIF


@dataclass
class MaterialConfig:
    """Controls material classification."""
    override:    Optional[Literal['cloth', 'hair', 'leaf', 'fluid', 'rigid', 'smoke']] = None
    # If override is None, the heuristic classifier is used


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class AnimationConfig:
    """Complete configuration for one animation run."""
    physics:   PhysicsConfig   = field(default_factory=PhysicsConfig)
    force:     ForceConfig     = field(default_factory=ForceConfig)
    flow:      FlowConfig      = field(default_factory=FlowConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    output:    OutputConfig    = field(default_factory=OutputConfig)
    material:  MaterialConfig  = field(default_factory=MaterialConfig)
    name:      str             = 'default'

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> 'AnimationConfig':
        return cls(
            physics   = PhysicsConfig(**d.get('physics',   {})),
            force     = ForceConfig(**d.get('force',       {})),
            flow      = FlowConfig(**d.get('flow',         {})),
            synthesis = SynthesisConfig(**d.get('synthesis', {})),
            output    = OutputConfig(**d.get('output',     {})),
            material  = MaterialConfig(**d.get('material', {})),
            name      = d.get('name', 'default'),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'AnimationConfig':
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> 'AnimationConfig':
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    def merge(self, overrides: Dict[str, Any]) -> 'AnimationConfig':
        """
        Apply dot-notation overrides to a copy of this config.

        Example:
            cfg.merge({'physics.stiffness': 1500, 'output.fps': 24})
        """
        d = self.to_dict()
        for key, val in overrides.items():
            parts = key.split('.')
            node  = d
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = val
        return AnimationConfig.from_dict(d)

    def validate(self) -> None:
        self.physics.validate()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def preset_flag(cls) -> 'AnimationConfig':
        return cls(
            name     = 'flag',
            physics  = PhysicsConfig(stiffness=1500, shear_stiffness=750,
                                     bend_stiffness=188, damping=0.35,
                                     pin_edge='left', substeps=10),
            force    = ForceConfig(preset='flag', wind_direction='right',
                                   wind_speed=120, gravity_y=100, strength=1.0),
            flow     = FlowConfig(feather_radius=5, sg_window=13),
            synthesis= SynthesisConfig(ema_alpha=0.90),
            output   = OutputConfig(fps=30, n_frames=90, format='gif'),
            material = MaterialConfig(override='cloth'),
        )

    @classmethod
    def preset_hair(cls) -> 'AnimationConfig':
        return cls(
            name     = 'hair',
            physics  = PhysicsConfig(stiffness=200, shear_stiffness=100,
                                     bend_stiffness=30, damping=0.7,
                                     pin_edge='top', substeps=6),
            force    = ForceConfig(preset='hair', wind_direction='right',
                                   wind_speed=80, gravity_y=200, strength=1.0),
            flow     = FlowConfig(feather_radius=3, sg_window=9),
            synthesis= SynthesisConfig(ema_alpha=0.85),
            output   = OutputConfig(fps=30, n_frames=60, format='gif'),
            material = MaterialConfig(override='hair'),
        )

    @classmethod
    def preset_leaf(cls) -> 'AnimationConfig':
        return cls(
            name     = 'leaf',
            physics  = PhysicsConfig(stiffness=800, shear_stiffness=400,
                                     bend_stiffness=150, damping=0.3,
                                     pin_edge='top', substeps=6),
            force    = ForceConfig(preset='leaf', wind_direction='right',
                                   wind_speed=60, gravity_y=80, strength=1.0),
            flow     = FlowConfig(feather_radius=4, sg_window=11),
            synthesis= SynthesisConfig(ema_alpha=0.88),
            output   = OutputConfig(fps=24, n_frames=72, format='gif'),
            material = MaterialConfig(override='leaf'),
        )

    @classmethod
    def preset_smoke(cls) -> 'AnimationConfig':
        return cls(
            name     = 'smoke',
            physics  = PhysicsConfig(stiffness=0, shear_stiffness=0,
                                     bend_stiffness=0, damping=0.05,
                                     pin_edge='bottom', substeps=4),
            force    = ForceConfig(wind_direction='up', wind_speed=50,
                                   gravity_y=-200, strength=1.0,
                                   turbulence_amp_frac=0.4),
            flow     = FlowConfig(feather_radius=6, spatial_sigma=2.0),
            synthesis= SynthesisConfig(ema_alpha=0.80),
            output   = OutputConfig(fps=24, n_frames=60, format='gif'),
            material = MaterialConfig(override='smoke'),
        )

    @classmethod
    def get_preset(cls, name: str) -> 'AnimationConfig':
        presets = {
            'flag':  cls.preset_flag,
            'hair':  cls.preset_hair,
            'leaf':  cls.preset_leaf,
            'smoke': cls.preset_smoke,
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets)}")
        return presets[name]()


# ---------------------------------------------------------------------------
# Default config YAML writer (run once to generate starter configs)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse as _argparse
    p = _argparse.ArgumentParser()
    p.add_argument('--output_dir', default='configs')
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    for name in ['flag', 'hair', 'leaf', 'smoke']:
        cfg = AnimationConfig.get_preset(name)
        cfg.to_yaml(out / f'{name}.yaml')
        print(f'Wrote {out / name}.yaml')

    # Default config
    AnimationConfig().to_yaml(out / 'default.yaml')
    print(f'Wrote {out}/default.yaml')