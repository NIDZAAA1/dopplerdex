"""Historical weather profiles (Open-Meteo historical-forecast API: global, hourly, no key).

Provides, for a site and time:
  - temperature profile T(z) from 2 m and pressure levels 1000-700 hPa  ->  effective sound speed
    over a slant path (harmonic mean of c(z) between ground and source height),
  - wind vector w(z) (ENU, m/s, direction the air moves TOWARD),
  - QNH-style correction for barometric (pressure) altitudes: pressure altitude assumes 1013.25 hPa.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dopplerdex.physics.atmosphere import speed_of_sound

CACHE = Path(__file__).resolve().parents[3] / "data/cache/weather"
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 700)
API = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _fetch(lat: float, lon: float, date: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{lat:.2f}_{lon:.2f}_{date}.json"
    if f.exists():
        return json.loads(f.read_text())
    hourly = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "pressure_msl", "wind_speed_10m",
              "wind_direction_10m"]  # fmt: skip
    for p in LEVELS:
        hourly += [f"temperature_{p}hPa", f"wind_speed_{p}hPa", f"wind_direction_{p}hPa", f"geopotential_height_{p}hPa"]
    url = (f"{API}?latitude={lat:.4f}&longitude={lon:.4f}&start_date={date}&end_date={date}&hourly={','.join(hourly)}"
           "&wind_speed_unit=ms&timezone=UTC")  # fmt: skip
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read())
    f.write_text(json.dumps(d))
    return d


def _wind_enu(speed: float, direction_from_deg: float) -> np.ndarray:
    to = np.radians(direction_from_deg + 180.0)
    return np.array([speed * np.sin(to), speed * np.cos(to), 0.0])


@dataclass(frozen=True)
class Profile:
    z: np.ndarray  # height above MSL (m), ascending
    temp_c: np.ndarray
    wind: np.ndarray  # (n, 3) ENU
    rh: float
    ground_msl: float
    pressure_msl_hpa: float

    def c_at(self, z_msl: np.ndarray | float) -> np.ndarray:
        t = np.interp(z_msl, self.z, self.temp_c)
        return speed_of_sound(t, self.rh)

    def c_path(self, z_top_msl: float, z_bottom_msl: float | None = None, n: int = 32) -> float:
        """Effective sound speed over a path between two heights (harmonic mean: travel-time weighted)."""
        zb = self.ground_msl if z_bottom_msl is None else z_bottom_msl
        zz = np.linspace(zb, max(z_top_msl, zb + 1.0), n)
        return float(1.0 / np.mean(1.0 / self.c_at(zz)))

    def wind_at(self, z_msl: float) -> np.ndarray:
        return np.array([np.interp(z_msl, self.z, self.wind[:, k]) for k in range(3)])

    def wind_path(self, z_top_msl: float) -> np.ndarray:
        zz = np.linspace(self.ground_msl, max(z_top_msl, self.ground_msl + 1.0), 16)
        return np.mean([self.wind_at(z) for z in zz], axis=0)

    def baro_to_msl(self, pressure_alt_m: np.ndarray | float) -> np.ndarray:
        """Pressure altitude (1013.25 hPa datum) -> approx. true MSL altitude using the local MSL pressure."""
        return np.asarray(pressure_alt_m) + 8.43 * (self.pressure_msl_hpa - 1013.25)


def profile(lat: float, lon: float, t_unix: float, ground_msl: float = 0.0) -> Profile:
    dt = datetime.fromtimestamp(t_unix, UTC)
    d = _fetch(round(lat, 2), round(lon, 2), dt.strftime("%Y-%m-%d"))
    h = d["hourly"]
    i = min(int(round(dt.hour + dt.minute / 60)), len(h["time"]) - 1)
    zs, ts, ws = (
        [ground_msl + 2.0],
        [h["temperature_2m"][i]],
        [_wind_enu(h["wind_speed_10m"][i], h["wind_direction_10m"][i])],
    )
    for p in LEVELS:
        z = h.get(f"geopotential_height_{p}hPa", [None])[i]
        if z is None or z <= ground_msl + 10:
            continue  # level below ground
        zs.append(z)
        ts.append(h[f"temperature_{p}hPa"][i])
        ws.append(_wind_enu(h[f"wind_speed_{p}hPa"][i], h[f"wind_direction_{p}hPa"][i]))
    o = np.argsort(zs)
    return Profile(np.array(zs)[o], np.array(ts, dtype=float)[o], np.array(ws)[o], float(h["relative_humidity_2m"][i]),
                   ground_msl, float(h.get("pressure_msl", h["surface_pressure"])[i]))  # fmt: skip
