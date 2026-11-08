"""Doppler physics for a moving source and a stationary observer in uniform wind.

Conventions (all SI, local ENU frame):
  - O: observer position (3,), S(t): source position, v_s: source ground velocity.
  - w: uniform wind vector (velocity of the air, e.g. from ADS-B wind / HRRR).
  - c: speed of sound relative to the air.
  - t_r: reception time at the observer, t_e: emission time at the source.

Exact relations for uniform wind (sound rides on the moving air):
  O - S(t_e) = (c * n + w) * (t_r - t_e)          n = wavefront normal (unit)
  =>  tau = t_r - t_e solves (c^2 - |w|^2) tau^2 + 2 (D.w) tau - |D|^2 = 0,  D = O - S(t_e)
  f_r / f_e = dt_e/dt_r = (c + n.w) / (c + n.w - n.v_s)

With w = 0 this reduces to the textbook f = f0 * c / (c + v_r), where v_r = -n.v_s
is the range rate (positive when receding) - the form used in the reference video.
The crucial subtlety (the video's "right vs wrong model"): v_s must be evaluated at
the *emission* time t_e, not the reception time t_r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dopplerdex.physics.trajectory import Trajectory

F64 = NDArray[np.float64]
_ZERO3 = np.zeros(3)


def propagation_delay(d: F64, c: float, wind: ArrayLike = _ZERO3) -> F64:
    """Acoustic travel time for displacement d = O - S (..., 3) in uniform wind."""
    w = np.asarray(wind, dtype=np.float64)
    dw = d @ w
    d2 = np.einsum("...i,...i->...", d, d)
    a = c * c - w @ w
    return (-dw + np.sqrt(dw * dw + a * d2)) / a


@dataclass(frozen=True)
class RetardedSolution:
    t_r: F64  # reception times (M,)
    t_e: F64  # emission times (M,)
    src_pos: F64  # source position at t_e (M, 3)
    src_vel: F64  # source velocity at t_e (M, 3)
    n: F64  # wavefront normal at the observer (M, 3)
    rng: F64  # straight-line distance |O - S(t_e)| (M,)
    doppler: F64  # f_received / f_emitted (M,)
    valid: NDArray[np.bool_]  # t_e lies within the observed trajectory span
    iterations: int
    max_residual: float  # max |t_e + tau(t_e) - t_r| in seconds


def solve_retarded(
    traj: Trajectory,
    observer: ArrayLike,
    t_r: ArrayLike,
    c: float,
    wind: ArrayLike = _ZERO3,
    tol: float = 1e-9,
    max_iter: int = 20,
) -> RetardedSolution:
    """Solve for emission times t_e(t_r) by Newton iteration, then compute the Doppler factor.

    g(t_e) = t_e + tau(O - S(t_e)) - t_r = 0, with g'(t_e) = (c + n.w - n.v) / (c + n.w) > 0
    for subsonic sources, so Newton converges quadratically from the zero-delay guess.
    Emission times outside the trajectory span use constant-velocity extrapolation and are
    flagged invalid in the result.
    """
    o = np.asarray(observer, dtype=np.float64)
    w = np.asarray(wind, dtype=np.float64)
    t_r = np.asarray(t_r, dtype=np.float64)

    pos, _ = traj.state(t_r)
    t_e = t_r - propagation_delay(o - pos, c, w)
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 - `it` is reported after the loop
        pos, vel = traj.state(t_e)
        d = o - pos
        tau = propagation_delay(d, c, w)
        n = (d - tau[:, None] * w) / (c * tau)[:, None]
        nw = n @ w
        nv = np.einsum("ij,ij->i", n, vel)
        g = t_e + tau - t_r
        gp = (c + nw - nv) / (c + nw)
        step = g / gp
        t_e = t_e - step
        if np.max(np.abs(step)) < tol:
            break

    pos, vel = traj.state(t_e)
    d = o - pos
    tau = propagation_delay(d, c, w)
    n = (d - tau[:, None] * w) / (c * tau)[:, None]
    nw = n @ w
    nv = np.einsum("ij,ij->i", n, vel)
    doppler = (c + nw) / (c + nw - nv)
    resid = float(np.max(np.abs(t_e + tau - t_r))) if t_r.size else 0.0
    return RetardedSolution(
        t_r=t_r,
        t_e=t_e,
        src_pos=pos,
        src_vel=vel,
        n=n,
        rng=np.linalg.norm(d, axis=-1),
        doppler=doppler,
        valid=traj.contains(t_e),
        iterations=it,
        max_residual=resid,
    )


def naive_doppler(traj: Trajectory, observer: ArrayLike, t_r: ArrayLike, c: float) -> F64:
    """The *wrong* model: source state taken at the reception time (no retardation).

    Kept deliberately for visualisation and as a negative control in tests.
    """
    o = np.asarray(observer, dtype=np.float64)
    pos, vel = traj.state(np.asarray(t_r, dtype=np.float64))
    d = o - pos
    n = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return c / (c - np.einsum("ij,ij->i", n, vel))


# ---------------------------------------------------------------------------
# Closed form for a straight, constant-velocity pass in still air.
# Used by the fast coarse-search stages and as an analytic test oracle.
# ---------------------------------------------------------------------------


def straight_pass_emission_offset(T: ArrayLike, v: float, r_cpa: float, c: float) -> F64:
    """Emission time relative to source-CPA time, u = t_e - tau_cpa, given T = t_r - tau_cpa.

    Solves c (T - u) = sqrt(r_cpa^2 + v^2 u^2) for the physical (u < T) root.
    """
    T = np.asarray(T, dtype=np.float64)
    a = c * c - v * v
    return (c * c * T - np.sqrt(c * c * v * v * T * T + a * r_cpa * r_cpa)) / a


def straight_pass_frequency(t_r: ArrayLike, f0: float, v: float, r_cpa: float, t_cpa: float, c: float) -> F64:
    """Received frequency for a straight constant-speed pass (still air).

    Args:
        t_r: reception times.
        f0: emitted frequency.
        v: source speed (m/s).
        r_cpa: closest-approach (slant) distance (m).
        t_cpa: time the *source* is at CPA (not when it is heard; that is t_cpa + r_cpa/c).
        c: speed of sound.

    f = f0 / (1 - M^2) * [1 - M v T / sqrt(v^2 T^2 + (1 - M^2) r_cpa^2)],  M = v/c, T = t_r - t_cpa
    """
    T = np.asarray(t_r, dtype=np.float64) - t_cpa
    m = v / c
    one_m2 = 1.0 - m * m
    return f0 / one_m2 * (1.0 - m * v * T / np.sqrt(v * v * T * T + one_m2 * r_cpa * r_cpa))


def straight_pass_params_from_plateaus(f_approach: float, f_recede: float, c: float) -> tuple[float, float]:
    """Model-free initial guess: (f0, v) from the approach / recede asymptotes."""
    v = c * (f_approach - f_recede) / (f_approach + f_recede)
    f0 = 2.0 * f_approach * f_recede / (f_approach + f_recede)
    return f0, v
