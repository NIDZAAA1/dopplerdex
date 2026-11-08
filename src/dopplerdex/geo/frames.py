"""Coordinate frames: WGS84 geodetic <-> ECEF <-> local ENU.

All functions are vectorised over leading array dimensions and use float64.
Angles are in degrees at the public API, heights are ellipsoidal (HAE) metres.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

F64 = NDArray[np.float64]


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> F64:
    """Geodetic (deg, deg, m HAE) -> ECEF (..., 3) in metres."""
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    lon_r = np.radians(np.asarray(lon, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)
    sl, cl = np.sin(lat_r), np.cos(lat_r)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    x = (n + h) * cl * np.cos(lon_r)
    y = (n + h) * cl * np.sin(lon_r)
    z = (n * (1.0 - WGS84_E2) + h) * sl
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz: ArrayLike) -> tuple[F64, F64, F64]:
    """ECEF (..., 3) -> (lat deg, lon deg, h m). Bowring's method, ~1e-9 deg accuracy."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * st**3, p - WGS84_E2 * WGS84_A * ct**3)
    sl = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    # Height: stable formulation away from the poles; fall back near poles.
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            np.abs(np.cos(lat)) > 1e-6,
            p / np.cos(lat) - n,
            np.abs(z) / np.abs(sl) - n * (1.0 - WGS84_E2),
        )
    return np.degrees(lat), np.degrees(lon), h


def enu_rotation(lat0: float, lon0: float) -> F64:
    """Rotation matrix R (3x3) such that enu = R @ (ecef - ecef0)."""
    la, lo = np.radians(lat0), np.radians(lon0)
    sla, cla, slo, clo = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array(
        [
            [-slo, clo, 0.0],
            [-sla * clo, -sla * slo, cla],
            [cla * clo, cla * slo, sla],
        ]
    )


class LocalFrame:
    """A local East-North-Up tangent frame anchored at (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0) -> None:
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self.origin_ecef = geodetic_to_ecef(lat0, lon0, h0)
        self.R = enu_rotation(lat0, lon0)

    def to_enu(self, lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> F64:
        ecef = geodetic_to_ecef(lat, lon, h)
        return (ecef - self.origin_ecef) @ self.R.T

    def to_geodetic(self, enu: ArrayLike) -> tuple[F64, F64, F64]:
        enu = np.asarray(enu, dtype=np.float64)
        ecef = enu @ self.R + self.origin_ecef
        return ecef_to_geodetic(ecef)

    def vec_to_enu(self, v_ecef: ArrayLike) -> F64:
        """Rotate a (free) vector from ECEF to ENU."""
        return np.asarray(v_ecef, dtype=np.float64) @ self.R.T


def velocity_enu(gs_ms: ArrayLike, track_deg: ArrayLike, vrate_ms: ArrayLike) -> F64:
    """ADS-B style ground speed / true track / vertical rate -> ENU velocity (..., 3)."""
    tr = np.radians(np.asarray(track_deg, dtype=np.float64))
    gs = np.asarray(gs_ms, dtype=np.float64)
    vz = np.asarray(vrate_ms, dtype=np.float64)
    return np.stack(np.broadcast_arrays(gs * np.sin(tr), gs * np.cos(tr), vz), axis=-1)
