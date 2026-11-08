"""Terrain elevation (Copernicus GLO-90, global, EGM2008 orthometric heights in metres).

Two levels:
  - CoarseDEM: a region mosaic at 30 arc-seconds (~0.9 km), built once from the COG overviews and
    cached as .npy (used by the coarse search, where 1 km terrain is plenty).
  - FineDEM: full-resolution 3 arc-second (~90 m) tiles fetched on demand and cached on disk
    (used by fine scoring / refinement around a handful of candidates).
Ocean tiles do not exist in the dataset; they read as 0 m.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

CACHE = Path(__file__).resolve().parents[3] / "data/cache/dem"
URL = "https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"
F64 = NDArray[np.float64]


def _tile_name(lat0: int, lon0: int) -> str:
    ns = f"N{lat0:02d}" if lat0 >= 0 else f"S{-lat0:02d}"
    ew = f"E{lon0:03d}" if lon0 >= 0 else f"W{-lon0:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


def _read_tile(lat0: int, lon0: int, size: int | None) -> np.ndarray | None:
    import rasterio
    from rasterio.errors import RasterioIOError

    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(URL.format(n=_tile_name(lat0, lon0))) as ds:
                if size is None:
                    a = ds.read(1)
                else:
                    a = ds.read(1, out_shape=(size, size))
        return np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    except RasterioIOError:
        return None  # ocean / missing tile


class CoarseDEM:
    """Regular lat/lon mosaic; bilinear sampling. Rows go north->south like the source tiles."""

    def __init__(self, lat_min: int, lat_max: int, lon_min: int, lon_max: int, per_deg: int = 120, workers: int = 24):
        from dopplerdex.env import synthetic

        self.flat = synthetic.FLAT["ground"] if synthetic.FLAT else None
        if self.flat is not None:
            return
        self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.per_deg = (
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            per_deg,
        )
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"coarse_{lat_min}_{lat_max}_{lon_min}_{lon_max}_{per_deg}.npy"
        if f.exists():
            self.z = np.load(f)
            return
        nr, nc = (lat_max - lat_min) * per_deg, (lon_max - lon_min) * per_deg
        z = np.zeros((nr, nc), dtype=np.float32)
        jobs = [(la, lo) for la in range(lat_min, lat_max) for lo in range(lon_min, lon_max)]

        def work(job):
            la, lo = job
            return job, _read_tile(la, lo, per_deg)

        with ThreadPoolExecutor(workers) as ex:
            for (la, lo), a in ex.map(work, jobs):
                if a is None:
                    continue
                r0 = (lat_max - la - 1) * per_deg
                c0 = (lo - lon_min) * per_deg
                z[r0 : r0 + per_deg, c0 : c0 + per_deg] = a
        np.save(f, z)
        self.z = z

    def sample(self, lat: ArrayLike, lon: ArrayLike) -> F64:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if self.flat is not None:
            return np.full(np.broadcast(lat, lon).shape, self.flat)
        r = (self.lat_max - lat) * self.per_deg - 0.5
        c = (lon - self.lon_min) * self.per_deg - 0.5
        r0 = np.clip(np.floor(r).astype(int), 0, self.z.shape[0] - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, self.z.shape[1] - 2)
        fr = np.clip(r - r0, 0, 1)
        fc = np.clip(c - c0, 0, 1)
        z = self.z
        return (
            z[r0, c0] * (1 - fr) * (1 - fc) + z[r0, c0 + 1] * (1 - fr) * fc
            + z[r0 + 1, c0] * fr * (1 - fc) + z[r0 + 1, c0 + 1] * fr * fc
        ).astype(np.float64)  # fmt: skip


@lru_cache(maxsize=64)
def _fine_tile(lat0: int, lon0: int) -> np.ndarray | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{_tile_name(lat0, lon0)}.npy"
    if f.exists():
        a = np.load(f)
        return None if a.size == 1 else a
    a = _read_tile(lat0, lon0, None)
    np.save(f, np.zeros(1, np.float32) if a is None else a)
    return a


def fine_elevation(lat: ArrayLike, lon: ArrayLike) -> F64:
    """Bilinear elevation from full-resolution GLO-90 tiles (fetched/cached on demand)."""
    from dopplerdex.env import synthetic

    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    if synthetic.FLAT:
        return np.full(np.broadcast(lat, lon).shape, synthetic.FLAT["ground"])
    out = np.zeros(lat.shape)
    la0 = np.floor(lat).astype(int)
    lo0 = np.floor(lon).astype(int)
    for a, b in set(zip(la0.ravel().tolist(), lo0.ravel().tolist(), strict=True)):
        m = (la0 == a) & (lo0 == b)
        t = _fine_tile(a, b)
        if t is None:
            continue
        n = t.shape[0]
        r = (a + 1 - lat[m]) * n - 0.5
        c = (lon[m] - b) * t.shape[1] - 0.5
        r0 = np.clip(np.floor(r).astype(int), 0, n - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, t.shape[1] - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        out[m] = (t[r0, c0] * (1 - fr) * (1 - fc) + t[r0, c0 + 1] * (1 - fr) * fc
                  + t[r0 + 1, c0] * fr * (1 - fc) + t[r0 + 1, c0 + 1] * fr * fc)  # fmt: skip
    return out
