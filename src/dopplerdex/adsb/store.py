"""Query the ingested ADS-B Parquet store and turn tracks into physics Trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from dopplerdex.geo.frames import LocalFrame, velocity_enu
from dopplerdex.physics.trajectory import Trajectory

STORE = Path(__file__).resolve().parents[3] / "data/processed/adsb"


def _day_dir(region: str, date: str) -> Path:
    return STORE / f"region={region}" / f"date={date}"


def load_points(
    t_lo: float,
    t_hi: float,
    lat: float,
    lon: float,
    radius_km: float,
    region: str = "north_america",
    classes: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """All airborne points within radius_km (box prefilter + great-circle-ish refine) and [t_lo, t_hi] (unix s)."""
    frames = []
    for t in np.arange(np.floor(t_lo / 3600) * 3600, t_hi + 1, 3600):
        dt = datetime.fromtimestamp(float(t), UTC)
        f = _day_dir(region, dt.strftime("%Y-%m-%d")) / f"hour={dt.hour:02d}" / "part-0.parquet"
        if f.exists():
            frames.append(f)
    if not frames:
        return pl.DataFrame()
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    q = (
        pl.scan_parquet(frames).filter(
            (pl.col("t") >= t_lo)
            & (pl.col("t") <= t_hi)
            & pl.col("lat").is_between(lat - dlat, lat + dlat)
            & pl.col("lon").is_between(lon - dlon, lon + dlon)
            & ~pl.col("on_ground")
        )  # fmt: skip
    )
    if classes:
        q = q.filter(pl.col("class").is_in(list(classes)))
    return q.sort(["icao", "t"]).collect()


@dataclass
class Flight:
    icao: str
    type: str
    aircraft_class: str
    leg: int
    traj: Trajectory  # ENU in the query frame
    alt_source: str  # "geom" | "baro"
    n_points: int
    max_gap_s: float


def _smooth_track(t: np.ndarray, pos: np.ndarray, gs: np.ndarray, trk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position-consistent velocities.

    Reported ground speed/track and the positions often disagree by several m/s in thinned traces
    (position and velocity messages are timestamped independently). Hermite interpolation with
    inconsistent velocities makes the path wiggle and the Doppler curve zig-zag. We fit a cubic
    smoothing spline per axis through the positions and
    take its derivative as the velocity, so position and velocity always agree. Smoothing level is
    fixed to the expected position noise (fast FITPACK spline; automatic GCV was too slow at scale).
    """
    from scipy.interpolate import UnivariateSpline

    if t.size < 5:
        return pos, velocity_enu(gs, trk, 0.0)
    out_p = np.empty_like(pos)
    out_v = np.empty_like(pos)
    n = t.size
    for k in range(3):
        # fixed smoothing matched to ADS-B position noise (~5 m): FITPACK, O(n), no GCV search
        sigma = 5.0 if k < 2 else 8.0
        try:
            spl = UnivariateSpline(t, pos[:, k], k=3, s=n * sigma**2)
            out_p[:, k] = spl(t)
            out_v[:, k] = spl.derivative()(t)
        except Exception:  # degenerate spacing: fall back to finite differences
            out_p[:, k] = pos[:, k]
            out_v[:, k] = np.gradient(pos[:, k], t)
    return out_p, out_v


def build_flights(
    df: pl.DataFrame, frame: LocalFrame, max_gap_s: float = 90.0, min_points: int = 4, geoid_n: float = 0.0
) -> list[Flight]:
    """Group points by (icao, leg), split at long gaps, convert to ENU trajectories.

    Altitude: GNSS geometric altitude when available (ellipsoidal height per DO-260B), otherwise
    barometric altitude (pressure altitude ~ MSL; QNH correction is applied later by the env module),
    shifted by the geoid undulation `geoid_n` so both are ellipsoidal heights.
    """
    out: list[Flight] = []
    if df.is_empty():
        return out
    for (icao, leg), g in df.group_by(["icao", "leg"], maintain_order=True):
        g = g.unique(subset="t", keep="first").sort("t")
        t = g["t"].to_numpy()
        gaps = np.flatnonzero(np.diff(t) > max_gap_s) + 1
        for seg in np.split(np.arange(t.size), gaps):
            if seg.size < min_points:
                continue
            s = g[seg]
            alt_g = s["alt_geom_m"].to_numpy()
            alt_b = s["alt_baro_m"].to_numpy()
            use_geom = np.mean(np.isfinite(alt_g)) > 0.5
            alt = alt_g if use_geom else alt_b + geoid_n
            ok = np.isfinite(alt) & np.isfinite(s["gs_ms"].to_numpy()) & np.isfinite(s["track_deg"].to_numpy())
            if ok.sum() < min_points:
                continue
            s = s.filter(pl.Series(ok))
            alt = alt[ok]
            tt = s["t"].to_numpy()
            pos = frame.to_enu(s["lat"].to_numpy(), s["lon"].to_numpy(), alt)
            pos, vel = _smooth_track(tt, pos, s["gs_ms"].to_numpy(), s["track_deg"].to_numpy())
            out.append(
                Flight(
                    icao=str(icao),
                    type=str(s["type"][0]),
                    aircraft_class=str(s["class"][0]),
                    leg=int(leg),
                    traj=Trajectory(tt, pos, vel),
                    alt_source="geom" if use_geom else "baro",
                    n_points=int(tt.size),
                    max_gap_s=float(np.max(np.diff(tt))) if tt.size > 1 else 0.0,
                )
            )
    return out
