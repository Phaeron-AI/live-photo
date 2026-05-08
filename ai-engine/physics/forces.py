"""
forces.py — External force primitives for the physics simulator.

Each force is a callable: (positions, velocities, masses, time) → force_array (N, 2)

This design lets you compose forces cleanly and swap them out per material type
without touching the integrator.

Physics reference (what you already know from JEE, extended):
  - Gravity:      F = mg (downward, constant)
  - Drag:         F = -b*v (linear drag) or -½ρCdA*v² (quadratic, for fluids)
  - Wind:         F = constant vector + turbulence noise
  - Buoyancy:     F = ρ_fluid * V * g (upward, for floating objects)
  - Spring:       F = -k(x - x₀) - c*v (handled in simulator.py)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Force(ABC):
  @abstractmethod
  def compute(
    self,
    positions: np.ndarray,   # (N, 2)
    velocities: np.ndarray,  # (N, 2)
    masses: np.ndarray,      # (N,)
    time: float,
  ) -> np.ndarray:             # (N, 2) force vectors
    ...


# ---------------------------------------------------------------------------
# Concrete forces
# ---------------------------------------------------------------------------

class Gravity(Force):
  """
  Uniform gravitational field.

  F_i = m_i * g

  In image space, y increases downward, so g = [0, +g_magnitude].
  Default: 9.8 * scale_factor. We use 980 px/s² as a reasonable
  visual magnitude for objects spanning ~100-500 pixels.
  """

  def __init__(self, g: float = 980.0, direction: np.ndarray = None): # type: ignore[assignment]
    """
    Args:
      g:         Gravitational acceleration magnitude (pixels/s²).
      direction: Unit vector. Default is [0, 1] (downward in image-space).
    """
    self.g = g
    self.direction = direction if direction is not None else np.array([0.0, 1.0])
    self.direction = self.direction / np.linalg.norm(self.direction)

  def compute(self, positions, velocities, masses, time):
    return masses[:, None] * self.g * self.direction[None, :]


class Wind(Force):
  """
  Wind force with Perlin-like turbulence.

  Base wind: constant vector.
  Turbulence: sum of sinusoids at different frequencies (cheap but effective).

  F_i = m_i * (wind_base + turbulence(t))

  This is mass-proportional so heavier particles are less affected.
  For cloth/hair you typically want the opposite — use WindPressure instead.
  """

  def __init__(
    self,
    direction: np.ndarray = None, # type: ignore[assignment]
    speed: float = 200.0,          # pixels/s²
    turbulence: float = 0.3,       # fraction of base speed
    freq: float = 1.5,             # Hz
  ):
    self.direction = direction if direction is not None else np.array([1.0, 0.0])
    self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-9)
    self.speed = speed
    self.turbulence = turbulence
    self.freq = freq

    # Random phase offsets for turbulence harmonics
    self.phases = np.random.uniform(0, 2 * np.pi, size=(4, 2))

  def compute(self, positions, velocities, masses, time):
    # Base wind
    base = self.speed * self.direction

    # Turbulence: sum of sinusoids (mimics Perlin noise cheaply)
    turb = np.zeros(2)
    for i in range(4):
      freq = self.freq * (i + 1)
      amp = self.turbulence * self.speed / (i + 1)
      turb += amp * np.sin(2 * np.pi * freq * time + self.phases[i])

    wind_accel = base + turb
    return masses[:, None] * wind_accel[None, :]


class WindPressure(Force):
  """
  Pressure-based wind: force proportional to projected area, not mass.
  More physically accurate for cloth and thin surfaces.

  F = 0.5 * ρ_air * Cd * A * v_wind²

  Since we don't track actual areas per particle, we use a simplified
  form: uniform pressure force independent of particle mass.
  """

  def __init__(
    self,
    direction: np.ndarray = None, # type: ignore[assignment]
    pressure: float = 50.0,    # force per particle
    turbulence: float = 0.2,
    freq: float = 2.0,
  ):
    self.direction = direction if direction is not None else np.array([1.0, 0.0])
    self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-9)
    self.pressure = pressure
    self.turbulence = turbulence
    self.freq = freq

  def compute(self, positions, velocities, masses, time):
    noise = self.turbulence * np.random.randn(2)
    force_vec = self.pressure * (self.direction + noise)
    return np.tile(force_vec, (len(masses), 1))   # uniform across all particles


class LinearDrag(Force):
  """
  Linear air resistance: F = -b * v

  Valid at low Reynolds numbers (viscous flow).
  For cloth/hair simulation this is the standard choice.

  Note: this is already included in the simulator as a default,
  but you can use this class to apply different drag to different
  material types.
  """

  def __init__(self, coefficient: float = 0.02):
    self.b = coefficient

  def compute(self, positions, velocities, masses, time):
    return -self.b * velocities * masses[:, None]


class QuadraticDrag(Force):
  """
  Quadratic drag: F = -½ * ρ * Cd * A * v * |v|

  Valid at high Reynolds numbers (turbulent flow, fast motion).
  Simplified to: F = -c * v * |v| per particle.
  """

  def __init__(self, coefficient: float = 1e-4):
    self.c = coefficient

  def compute(self, positions, velocities, masses, time):
    speed = np.linalg.norm(velocities, axis=1, keepdims=True)  # (N, 1)
    return -self.c * velocities * speed


class Buoyancy(Force):
  """
  Upward buoyant force for fluid/smoke simulations.
  Counteracts gravity partially or fully depending on material density.

  F = ρ_fluid * V * g (upward)

  Approximated per-particle as: F_i = density_ratio * m_i * g * (-ĝ)
  where density_ratio < 1 means particle rises, > 1 means it sinks.
  """

  def __init__(self, density_ratio: float = 0.3, g: float = 980.0):
    """
    Args:
      density_ratio: particle_density / fluid_density.
               0.1 = very buoyant (smoke), 0.9 = slightly buoyant.
      g: gravitational magnitude matching your Gravity force.
    """
    self.density_ratio = density_ratio
    self.g = g

  def compute(self, positions, velocities, masses, time):
    # Buoyancy opposes gravity (upward = negative y in image space)
    buoyancy_accel = (1.0 - self.density_ratio) * self.g
    return masses[:, None] * np.array([0.0, -buoyancy_accel])[None, :]


class PointAttractor(Force):
  """
  Draws all particles toward a target point.
  Useful for fluid surface tension, or interactive user-guided motion.

  F = strength * (target - x_i) / ||target - x_i||²
  """

  def __init__(self, target: np.ndarray, strength: float = 100.0, falloff: float = 2.0):
    self.target = target.astype(np.float64)
    self.strength = strength
    self.falloff = falloff

  def compute(self, positions, velocities, masses, time):
    delta = self.target[None, :] - positions   # (N, 2)
    dist_sq = (delta ** 2).sum(axis=1, keepdims=True) + 1e-8
    return self.strength * delta / (dist_sq ** (self.falloff / 2))


# ---------------------------------------------------------------------------
# Force registry — maps material type to default force set
# ---------------------------------------------------------------------------

class ForceRegistry:
  """
  Composes multiple force objects and evaluates them together.

  Usage:
    registry = ForceRegistry()
    registry.add(Gravity())
    registry.add(Wind(speed=300.0))
    total_force = registry.compute(positions, velocities, masses, time)
  """

  def __init__(self):
    self._forces: list[Force] = []

  def add(self, force: Force) -> 'ForceRegistry':
    self._forces.append(force)
    return self

  def remove(self, force_type: type) -> None:
    self._forces = [f for f in self._forces if not isinstance(f, force_type)]

  def compute(
    self,
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    time: float,
  ) -> np.ndarray:
    total = np.zeros_like(positions)
    for force in self._forces:
      total += force.compute(positions, velocities, masses, time)
    return total


# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------

def cloth_forces(wind_speed: float = 200.0) -> ForceRegistry:
  """Spring forces dominate; gravity + wind pressure."""
  return (
    ForceRegistry()
    .add(Gravity(g=500.0))
    .add(WindPressure(speed=wind_speed, turbulence=0.25)) # type: ignore[union-attr]
    .add(LinearDrag(coefficient=0.03))
  )


def hair_forces(wind_speed: float = 150.0) -> ForceRegistry:
  """Lighter gravity, more wind sensitivity."""
  return (
    ForceRegistry()
    .add(Gravity(g=300.0))
    .add(WindPressure(speed=wind_speed, turbulence=0.4, freq=3.0)) # type: ignore[union-attr]
    .add(LinearDrag(coefficient=0.02))
  )


def smoke_forces() -> ForceRegistry:
  """Strong buoyancy, weak gravity, turbulent wind."""
  return (
    ForceRegistry()
    .add(Gravity(g=980.0))
    .add(Buoyancy(density_ratio=0.05))
    .add(Wind(speed=80.0, turbulence=0.6, freq=3.0)) # type: ignore[union-attr]
    .add(LinearDrag(coefficient=0.01))
  )


def water_forces() -> ForceRegistry:
  """Heavy gravity, minimal wind, quadratic drag (dense medium)."""
  return (
    ForceRegistry()
    .add(Gravity(g=980.0))
    .add(QuadraticDrag(coefficient=2e-4))
  )


def rigid_forces() -> ForceRegistry:
  """Standard gravity + linear drag for rigid bodies."""
  return (
    ForceRegistry()
    .add(Gravity(g=980.0))
    .add(LinearDrag(coefficient=0.05))
  )