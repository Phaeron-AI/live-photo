"""
simulator.py — Velocity Verlet spring-mass physics simulator.

This module takes a Mesh from mesh.py and steps it forward in time
under external forces (gravity, wind) and internal spring forces.

Why Velocity Verlet and not Euler?
-----------------------------------
Euler integration: x(t+dt) = x(t) + v(t)*dt
                   v(t+dt) = v(t) + a(t)*dt

The problem: acceleration used to update velocity is stale (from start
of step). This causes energy GAIN in oscillatory systems — springs grow
forever. You'll see this if you implement Euler first: cloth explodes.

Velocity Verlet:
  x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
  a(t+dt) = F(x(t+dt)) / m          ← recompute forces at new position
  v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

This is a symplectic integrator — it conserves a "shadow" Hamiltonian,
meaning energy stays bounded over long simulations. This is the standard
in game physics engines and cloth simulation.

References:
  Verlet, L. (1967). Computer "experiments" on classical fluids.
  Baraff & Witkin (1998). Large Steps in Cloth Simulation. SIGGRAPH.
  Müller et al. (2003). Position Based Dynamics.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from motion.flow_scheduler import ForceConfig  # canonical definition lives there
from physics.mesh import Mesh


class PhysicsSimulator:
    """
    Velocity Verlet spring-mass simulator for deformable mesh objects.

    Usage:
        sim = PhysicsSimulator(mesh, dt=1/60, forces=ForceConfig(...))
        for frame in range(n_frames):
            sim.step()
            positions = sim.get_positions()  # (N, 2) current particle positions

    Particle sync:
        The internal simulation state lives in flat numpy arrays (self.positions,
        self.velocities) for performance. The Particle objects on mesh.particles
        are NOT kept in sync automatically — call sync_particles() explicitly
        when external code needs to read per-particle attributes. This avoids
        an O(N) Python loop on every substep.
    """

    def __init__(
        self,
        mesh:      Mesh,
        dt:        float = 1.0 / 60.0,
        forces:    Optional[ForceConfig] = None,
        substeps:  int = 4,
    ):
        self.mesh      = mesh
        self.dt        = dt
        self.sub_dt    = dt / substeps
        self.substeps  = substeps
        self.forces    = forces or ForceConfig()
        self.time      = 0.0

        self.positions  = np.array([p.position for p in mesh.particles])  # (N, 2)
        self.velocities = np.array([p.velocity for p in mesh.particles])  # (N, 2)
        self.masses     = np.array([p.mass     for p in mesh.particles])  # (N,)
        self.pinned     = np.array([p.pinned   for p in mesh.particles])  # (N,) bool

        self.rest_positions = self.positions.copy()                        # (N, 2)
        self.inv_masses     = np.where(self.pinned, 0.0, 1.0 / self.masses)

        self._cache_springs()
        self.accelerations = self._compute_accelerations()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance simulation by one timestep (self.dt), using substeps."""
        for _ in range(self.substeps):
            self._velocity_verlet_step(self.sub_dt)
        self.time += self.dt

    def sync_particles(self) -> None:
        """
        Copy numpy state back to the Particle objects on mesh.particles.

        Call only when external code needs to iterate over Particle instances.
        Avoid inside the simulation loop — it adds an O(N) Python loop per call.
        """
        for i, p in enumerate(self.mesh.particles):
            p.position = self.positions[i].copy()
            p.velocity = self.velocities[i].copy()

    def get_positions(self) -> np.ndarray:
        """Return current particle positions, shape (N, 2)."""
        return self.positions.copy()

    def get_displacements(self) -> np.ndarray:
        """Return displacement from rest position for each particle, shape (N, 2)."""
        return self.positions - self.rest_positions

    def pin_particle(self, idx: int) -> None:
        self.pinned[idx]     = True
        self.inv_masses[idx] = 0.0

    def apply_impulse(self, idx: int, impulse: np.ndarray) -> None:
        """Apply an instantaneous velocity change to particle idx."""
        self.velocities[idx] += impulse * self.inv_masses[idx]

    # ------------------------------------------------------------------
    # Core integrator
    # ------------------------------------------------------------------

    def _velocity_verlet_step(self, dt: float) -> None:
        """
        One Velocity Verlet substep.

        Mathematics:
          x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
          a(t+dt) = F(x(t+dt)) / m
          v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        """
        a_old = self.accelerations

        self.positions += self.velocities * dt + 0.5 * a_old * dt * dt

        a_new = self._compute_accelerations()

        self.velocities += 0.5 * (a_old + a_new) * dt

        if self.pinned.any():
            self.velocities[self.pinned] = 0.0
            self.positions[self.pinned]  = self.rest_positions[self.pinned]

        self.accelerations = a_new

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def _compute_accelerations(self) -> np.ndarray:
        """
        Compute net acceleration for each particle.

        a = F_total / m = (F_spring + F_gravity + F_wind + F_drag) / m
        """
        forces = np.zeros_like(self.positions)

        self._accumulate_spring_forces(forces)
        self._accumulate_external_forces(forces)

        forces -= self.forces.drag * self.velocities * self.masses[:, None]

        return forces * self.inv_masses[:, None]

    def _accumulate_spring_forces(self, forces: np.ndarray) -> None:
        """
        Vectorised spring force computation.

        For each spring connecting particles i and j:
          delta = x_j - x_i
          dist  = ||delta||
          unit  = delta / dist
          F_spring  = -k * (dist - L₀) * unit
          F_damping = -c * (v_j - v_i) · unit * unit
        """
        p0 = self.spring_p0
        p1 = self.spring_p1

        delta = self.positions[p1] - self.positions[p0]
        dist  = np.linalg.norm(delta, axis=1, keepdims=True)
        dist  = np.maximum(dist, 1e-8)
        unit  = delta / dist

        stretch = dist.squeeze() - self.spring_rest
        f_mag   = self.spring_k * stretch

        rel_vel    = self.velocities[p1] - self.velocities[p0]
        vel_along  = (rel_vel * unit).sum(axis=1)
        f_mag     += self.spring_c * vel_along

        f_total = f_mag[:, None] * unit

        np.add.at(forces, p0,  f_total)
        np.add.at(forces, p1, -f_total)

    def _accumulate_external_forces(self, forces: np.ndarray) -> None:
        """Gravity and wind forces."""
        forces += self.masses[:, None] * self.forces.gravity[None, :]

        if self.forces.wind_noise_scale > 0:
            noise = (self.forces.wind_noise_scale
                     * np.random.randn(2)
                     * np.linalg.norm(self.forces.wind))
        else:
            noise = np.zeros(2)

        forces += (self.forces.wind + noise) * self.masses[:, None]

    # ------------------------------------------------------------------
    # Spring cache
    # ------------------------------------------------------------------

    def _cache_springs(self) -> None:
        """
        Pre-extract spring data into flat numpy arrays for vectorised ops.
        Doing this once at init avoids Python-level looping every step.
        """
        springs          = self.mesh.springs
        self.spring_p0   = np.array([s.p0         for s in springs], dtype=np.int32)
        self.spring_p1   = np.array([s.p1         for s in springs], dtype=np.int32)
        self.spring_rest = np.array([s.rest_length for s in springs], dtype=np.float64)
        self.spring_k    = np.array([s.stiffness   for s in springs], dtype=np.float64)
        self.spring_c    = np.array([s.damping     for s in springs], dtype=np.float64)