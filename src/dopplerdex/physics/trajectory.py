"""Aircraft trajectory representation in a local ENU frame.

A Trajectory is a set of timestamped states (position, velocity) with C1 cubic
Hermite interpolation. Real ADS-B gives both position and velocity, which makes
Hermite interpolation the natural, physically consistent choice; synthetic
trajectories are built from simple kinematic primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Trajectory:
    t: F64  # (N,) seconds, strictly increasing
    pos: F64  # (N, 3) ENU metres
    vel: F64  # (N, 3) ENU m/s

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("Trajectory needs >= 2 samples")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Trajectory times must be strictly increasing")
        if self.pos.shape != (t.size, 3) or self.vel.shape != (t.size, 3):
            raise ValueError("pos/vel must be (N, 3)")

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def contains(self, tq: ArrayLike) -> NDArray[np.bool_]:
        tq = np.asarray(tq, dtype=np.float64)
        return (tq >= self.t[0]) & (tq <= self.t[-1])

    def state(self, tq: ArrayLike) -> tuple[F64, F64]:
        """Cubic-Hermite position and velocity at query times.

        Outside the span the state is extrapolated at the end-point velocity (constant
        velocity), which keeps position and velocity mutually consistent; callers should
        use `contains` to know which samples are backed by data.
        """
        tq_raw = np.asarray(tq, dtype=np.float64)
        tq = np.clip(tq_raw, self.t[0], self.t[-1])
        i = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, self.t.size - 2)
        t0, t1 = self.t[i], self.t[i + 1]
        h = (t1 - t0)[..., None]
        s = ((tq - t0) / (t1 - t0))[..., None]
        p0, p1 = self.pos[i], self.pos[i + 1]
        m0, m1 = self.vel[i] * h, self.vel[i + 1] * h
        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        d00 = 6 * s2 - 6 * s
        d10 = 3 * s2 - 4 * s + 1
        d01 = -6 * s2 + 6 * s
        d11 = 3 * s2 - 2 * s
        vel = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h
        pos = pos + (tq_raw - tq)[..., None] * vel  # linear extrapolation (zero inside span)
        return pos, vel

    # ---- constructors -------------------------------------------------------

    @staticmethod
    def straight(
        p_cpa: ArrayLike,
        velocity: ArrayLike,
        t_cpa: float,
        t_start: float,
        t_end: float,
        dt: float = 1.0,
    ) -> Trajectory:
        """Constant-velocity straight line passing p_cpa at time t_cpa."""
        p_cpa = np.asarray(p_cpa, dtype=np.float64)
        v = np.asarray(velocity, dtype=np.float64)
        t = np.arange(t_start, t_end + 0.5 * dt, dt)
        pos = p_cpa + (t - t_cpa)[:, None] * v
        vel = np.broadcast_to(v, pos.shape).copy()
        return Trajectory(t, pos, vel)

    @staticmethod
    def from_kinematics(
        t: ArrayLike,
        p0: ArrayLike,
        speed: ArrayLike,
        heading_deg: ArrayLike,
        vrate: ArrayLike,
    ) -> Trajectory:
        """Integrate time series of speed / heading / vertical rate into a trajectory.

        Arrays are sampled on `t`; integration uses the trapezoid rule, which is
        accurate for the smooth manoeuvres we synthesise (turns, climbs, speed changes).
        """
        t = np.asarray(t, dtype=np.float64)
        n = t.size
        speed = np.broadcast_to(np.asarray(speed, dtype=np.float64), (n,))
        hdg = np.radians(np.broadcast_to(np.asarray(heading_deg, dtype=np.float64), (n,)))
        vz = np.broadcast_to(np.asarray(vrate, dtype=np.float64), (n,))
        vel = np.stack([speed * np.sin(hdg), speed * np.cos(hdg), vz], axis=-1)
        dt = np.diff(t)[:, None]
        steps = 0.5 * (vel[1:] + vel[:-1]) * dt
        pos = np.asarray(p0, dtype=np.float64) + np.vstack([np.zeros((1, 3)), np.cumsum(steps, axis=0)])
        return Trajectory(t, pos, vel)
