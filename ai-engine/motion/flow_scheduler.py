"""
flow_scheduler.py — Time-varying force profiles for the physics simulator.

Motivation:
  A constant wind force produces a dead-looking animation — the cloth just
  deflects to a fixed angle and stops. Real motion (flags, hair, leaves) is
  driven by turbulence: gusts that peak and fade, direction shifts, high-freq
  flutter superimposed on a low-freq sway.

  This module lets you define force *envelopes* — how wind/gravity magnitude
  and direction change over the animation timeline — giving the physics
  simulator time-varying inputs that produce much more natural motion.

Architecture:
  - Keyframe interpolation  (cubic Hermite spline between user-defined anchors)
  - Procedural oscillators  (sine, perlin-like, gust bursts)
  - Composite scheduler     (sum of any number of keyframe + oscillator tracks)

The scheduler is queried once per simulation step via scheduler.get(t) → ForceConfig.

Usage:
    sched = ForceScheduler()
    sched.add_wind_gust(t_start=1.0, duration=0.5, direction='right', peak=250.0)
    sched.add_sway(period=2.0, amplitude=60.0, axis='x')

    sim = PhysicsSimulator(mesh, dt=1/60, forces=sched.get(0.0))
    for f in range(n_frames):
        sim.forces = sched.get(f / fps)
        sim.step()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class ForceConfig:
    """
    External forces for one simulation step.

    Canonical definition lives here so the motion layer owns it and
    physics.simulator imports it — not the other way around, which
    would create an upward dependency.
    """
    gravity:          np.ndarray = field(default_factory=lambda: np.array([0.0, 980.0]))
    wind:             np.ndarray = field(default_factory=lambda: np.zeros(2))
    wind_noise_scale: float = 0.0
    wind_noise_freq:  float = 2.0
    drag:             float = 0.02


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _hermite_interp(t: float, t0: float, t1: float, v0: float, v1: float,
                    m0: float = 0.0, m1: float = 0.0) -> float:
    """
    Cubic Hermite spline between (t0, v0) and (t1, v1) with tangents m0, m1.
    Returns v0 for t ≤ t0, v1 for t ≥ t1.
    """
    if t <= t0: return v0
    if t >= t1: return v1
    u  = (t - t0) / (t1 - t0)
    u2, u3 = u * u, u * u * u
    h00 =  2*u3 - 3*u2 + 1
    h10 =    u3 - 2*u2 + u
    h01 = -2*u3 + 3*u2
    h11 =    u3 -   u2
    dt  = t1 - t0
    return h00*v0 + h10*dt*m0 + h01*v1 + h11*dt*m1


def _smoothstep(t: float, t0: float, t1: float) -> float:
    """Smooth 0→1 ramp from t0 to t1 (clamped)."""
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    u = (t - t0) / (t1 - t0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------
# Wind direction constants
# ---------------------------------------------------------------------------

_WIND_DIR = {
    'right': np.array([ 1.0,  0.0]),
    'left':  np.array([-1.0,  0.0]),
    'up':    np.array([ 0.0, -1.0]),
    'down':  np.array([ 0.0,  1.0]),
}


# ---------------------------------------------------------------------------
# Track types
# ---------------------------------------------------------------------------

@dataclass
class GustTrack:
    """A single wind gust: smooth ramp-up then ramp-down."""
    t_start:   float
    duration:  float
    direction: np.ndarray   # unit vector
    peak:      float        # peak force magnitude

    def sample(self, t: float) -> np.ndarray:
        t_peak = self.t_start + self.duration * 0.35
        t_end  = self.t_start + self.duration
        if t < self.t_start or t > t_end:
            return np.zeros(2)
        if t < t_peak:
            w = _smoothstep(t, self.t_start, t_peak)
        else:
            w = 1.0 - _smoothstep(t, t_peak, t_end)
        return self.direction * self.peak * w


@dataclass
class SwayTrack:
    """Sinusoidal oscillation — primary sway frequency."""
    period:    float         # seconds per cycle
    amplitude: float         # force magnitude
    axis:      np.ndarray    # oscillation axis (unit vector)
    phase:     float = 0.0   # phase offset in radians

    def sample(self, t: float) -> np.ndarray:
        v = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
        return self.axis * v


@dataclass
class TurbulenceTrack:
    """
    Sum-of-sinusoids turbulence (approximates Perlin noise cheaply).
    Produces aperiodic-looking wind variation.
    """
    base_amplitude: float
    base_freq:      float   # Hz
    n_octaves:      int = 4
    lacunarity:     float = 2.0   # frequency ratio between octaves
    persistence:    float = 0.5   # amplitude ratio between octaves
    direction:      np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    seed:           int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._phases = rng.uniform(0, 2 * np.pi, (self.n_octaves, 2))

    def sample(self, t: float) -> np.ndarray:
        result = np.zeros(2)
        amp  = self.base_amplitude
        freq = self.base_freq
        for i in range(self.n_octaves):
            result[0] += amp * np.sin(2 * np.pi * freq * t + self._phases[i, 0])
            result[1] += amp * np.sin(2 * np.pi * freq * t + self._phases[i, 1])
            amp  *= self.persistence
            freq *= self.lacunarity
        return result


@dataclass
class KeyframeTrack:
    """
    Cubic Hermite spline through a list of (time, value) keyframes.
    value is a scalar (for force magnitude) — use two of these for x/y.
    """
    times:  list[float]
    values: list[float]

    def __post_init__(self) -> None:
        assert len(self.times) == len(self.values) >= 2
        self.times  = list(self.times)
        self.values = list(self.values)

    def sample(self, t: float) -> float:
        ts, vs = self.times, self.values
        if t <= ts[0]:  return vs[0]
        if t >= ts[-1]: return vs[-1]
        # Find surrounding segment
        for i in range(len(ts) - 1):
            if ts[i] <= t < ts[i + 1]:
                # Finite-difference tangents (Catmull-Rom)
                m0 = (vs[min(i+1, len(vs)-1)] - vs[max(i-1, 0)]) / (ts[min(i+1, len(ts)-1)] - ts[max(i-1, 0)] + 1e-9)
                m1 = (vs[min(i+2, len(vs)-1)] - vs[max(i, 0)])   / (ts[min(i+2, len(ts)-1)] - ts[max(i, 0)]   + 1e-9)
                return _hermite_interp(t, ts[i], ts[i+1], vs[i], vs[i+1], m0, m1)
        return vs[-1]


# ---------------------------------------------------------------------------
# Force scheduler
# ---------------------------------------------------------------------------

class ForceScheduler:
    """
    Composes multiple force tracks into a time-varying ForceConfig.

    Usage:
        sched = ForceScheduler(base_wind=np.array([80., 0.]), base_gravity=np.array([0., 150.]))
        sched.add_gust(t_start=0.5, duration=0.8, direction='right', peak=300.)
        sched.add_sway(period=2.5, amplitude=40., axis='x')
        sched.add_turbulence(base_amplitude=25., base_freq=1.8)

        for frame in range(n_frames):
            t = frame / fps
            sim.forces = sched.get(t)
            sim.step()
    """

    def __init__(
        self,
        base_wind:    Optional[np.ndarray] = None,
        base_gravity: Optional[np.ndarray] = None,
        base_drag:    float = 0.02,
    ):
        self._base_wind    = base_wind    if base_wind    is not None else np.zeros(2)
        self._base_gravity = base_gravity if base_gravity is not None else np.array([0., 150.])
        self._base_drag    = base_drag
        self._tracks: list = []

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_gust(
        self,
        t_start:   float,
        duration:  float,
        direction: Literal['right', 'left', 'up', 'down'] | np.ndarray = 'right',
        peak:      float = 200.0,
    ) -> 'ForceScheduler':
        d = _WIND_DIR[direction] if isinstance(direction, str) else np.asarray(direction)
        self._tracks.append(GustTrack(t_start, duration, d / np.linalg.norm(d), peak))
        return self

    def add_sway(
        self,
        period:    float = 2.0,
        amplitude: float = 60.0,
        axis:      Literal['x', 'y'] | np.ndarray = 'x',
        phase:     float = 0.0,
    ) -> 'ForceScheduler':
        if isinstance(axis, str):
            ax = np.array([1.0, 0.0]) if axis == 'x' else np.array([0.0, 1.0])
        else:
            ax = np.asarray(axis, dtype=np.float64)
            ax = ax / np.linalg.norm(ax)
        self._tracks.append(SwayTrack(period, amplitude, ax, phase))
        return self

    def add_turbulence(
        self,
        base_amplitude: float = 30.0,
        base_freq:      float = 1.5,
        n_octaves:      int   = 4,
        seed:           int   = 42,
    ) -> 'ForceScheduler':
        self._tracks.append(TurbulenceTrack(
            base_amplitude=base_amplitude,
            base_freq=base_freq,
            n_octaves=n_octaves,
            seed=seed,
        ))
        return self

    def add_wind_keyframes(
        self,
        times:       list[float],
        magnitudes:  list[float],
        direction:   Literal['right', 'left', 'up', 'down'] | np.ndarray = 'right',
    ) -> 'ForceScheduler':
        d = _WIND_DIR[direction] if isinstance(direction, str) else np.asarray(direction)
        d = d / np.linalg.norm(d)
        self._tracks.append({'type': 'kf_wind', 'dir': d,
                              'kf': KeyframeTrack(times, magnitudes)})
        return self

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def flag_in_wind(cls, wind_speed: float = 100.0, fps: int = 30) -> 'ForceScheduler':
        """Realistic flag waving: constant base + sway + turbulence + occasional gusts."""
        s = cls(
            base_wind=np.array([wind_speed, 0.0]),
            base_gravity=np.array([0.0, 120.0]),
        )
        s.add_sway(period=2.2, amplitude=wind_speed * 0.35, axis='x')
        s.add_sway(period=1.1, amplitude=wind_speed * 0.08, axis='y', phase=1.2)
        s.add_turbulence(base_amplitude=wind_speed * 0.15, base_freq=2.0)
        s.add_gust(t_start=0.8, duration=0.6, direction='right', peak=wind_speed * 2.0)
        s.add_gust(t_start=2.5, duration=0.4, direction='right', peak=wind_speed * 1.6)
        return s

    @classmethod
    def hair_in_breeze(cls, wind_speed: float = 80.0) -> 'ForceScheduler':
        s = cls(
            base_wind=np.array([wind_speed * 0.4, 0.0]),
            base_gravity=np.array([0.0, 200.0]),
        )
        s.add_sway(period=1.6, amplitude=wind_speed * 0.5, axis='x')
        s.add_sway(period=0.7, amplitude=wind_speed * 0.1, axis='x', phase=0.8)
        s.add_turbulence(base_amplitude=wind_speed * 0.25, base_freq=3.0)
        return s

    @classmethod
    def leaf_flutter(cls) -> 'ForceScheduler':
        s = cls(base_gravity=np.array([0.0, 80.0]))
        s.add_sway(period=0.9, amplitude=120.0, axis='x')
        s.add_sway(period=0.45, amplitude=30.0, axis='y', phase=0.5)
        s.add_turbulence(base_amplitude=60.0, base_freq=4.0, n_octaves=5)
        s.add_gust(t_start=0.3, duration=0.25, direction='right', peak=200.0)
        return s

    @classmethod
    def smoke_rising(cls) -> 'ForceScheduler':
        """
        Smoke / fire: strong upward buoyancy, lateral drift, high turbulence.
        Gravity is negative (upward in image-space) to simulate buoyancy.
        """
        s = cls(
            base_wind=np.array([20.0, 0.0]),
            base_gravity=np.array([0.0, -180.0]),
            base_drag=0.015,
        )
        s.add_sway(period=1.4, amplitude=50.0, axis='x')
        s.add_sway(period=0.6, amplitude=15.0, axis='x', phase=1.1)
        s.add_turbulence(base_amplitude=40.0, base_freq=3.5, n_octaves=5)
        return s

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, t: float) -> ForceConfig:
        """Return a ForceConfig for simulation time `t` (seconds)."""
        wind = self._base_wind.copy()

        for track in self._tracks:
            if isinstance(track, (GustTrack, SwayTrack, TurbulenceTrack)):
                wind += track.sample(t)
            elif isinstance(track, dict) and track['type'] == 'kf_wind':
                mag = track['kf'].sample(t)
                wind += track['dir'] * mag

        return ForceConfig(
            gravity=self._base_gravity.copy(),
            wind=wind,
            wind_noise_scale=0.0,   # turbulence handled here, not in sim
            drag=self._base_drag,
        )