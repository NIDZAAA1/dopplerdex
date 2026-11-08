"""Propagation effects between source and microphone.

- Spherical spreading (1/r pressure).
- Ground reflection (Lloyd's mirror): an image path via the ground plane, with a
  plane-wave reflection coefficient (rigid, or Delany-Bazley impedance ground).
- Atmospheric absorption lives in `atmosphere.absorption_db_per_m`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]
C128 = NDArray[np.complex128]


def image_observer(observer: ArrayLike, ground_z: float) -> F64:
    """Mirror the observer through the horizontal ground plane z = ground_z.

    The reflected path from S to O has the same length as the direct path from S to O'.
    """
    o = np.array(observer, dtype=np.float64)
    o[..., 2] = 2.0 * ground_z - o[..., 2]
    return o


def delany_bazley_impedance(freq_hz: ArrayLike, flow_resistivity: float) -> C128:
    """Normalised surface impedance Z/(rho c) of a porous ground (Delany-Bazley 1970).

    flow_resistivity in Pa s / m^2 (kPa s/m^2 * 1e3). Typical values:
      grass ~ 150-300e3, compacted soil ~ 2000e3, asphalt/concrete ~ 2e7+ (near rigid).
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    x = f / (flow_resistivity / 1e3)  # the empirical fit is in f / (kPa s m^-2)
    return 1.0 + 9.08 * x**-0.75 + 1j * 11.9 * x**-0.73


def plane_wave_reflection(freq_hz: ArrayLike, grazing_angle_rad: ArrayLike, flow_resistivity: float | None) -> C128:
    """Plane-wave reflection coefficient R = (Z sin(psi) - 1) / (Z sin(psi) + 1).

    flow_resistivity=None means a rigid ground (R = 1).
    """
    psi = np.asarray(grazing_angle_rad, dtype=np.float64)
    if flow_resistivity is None:
        return np.ones(np.broadcast(np.asarray(freq_hz), psi).shape, dtype=np.complex128)
    z = delany_bazley_impedance(freq_hz, flow_resistivity)
    s = np.sin(psi)
    return (z * s - 1.0) / (z * s + 1.0)


def lloyd_mirror_null_frequencies(mic_height: float, elevation_rad: float, c: float, n_max: int = 5) -> F64:
    """Interference null frequencies over a rigid ground for a distant source.

    Path difference ~ 2 h sin(phi); nulls where it equals odd half-wavelengths.
    """
    delta = 2.0 * mic_height * np.sin(elevation_rad)
    n = np.arange(n_max)
    return (2 * n + 1) * c / (2.0 * delta)
