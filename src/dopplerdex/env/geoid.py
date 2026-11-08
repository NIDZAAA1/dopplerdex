"""Geoid undulation N (EGM96, 15' grid from the PROJ CDN): ellipsoidal h = orthometric H + N."""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

CACHE = Path(__file__).resolve().parents[3] / "data/cache/geoid"
URL = "https://cdn.proj.org/us_nga_egm96_15.tif"


@lru_cache(maxsize=1)
def _grid():
    import rasterio

    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "us_nga_egm96_15.tif"
    if not f.exists():
        urllib.request.urlretrieve(URL, f)
    with rasterio.open(f) as ds:
        return ds.read(1).astype(np.float64), ds.transform


def undulation(lat: ArrayLike, lon: ArrayLike) -> NDArray[np.float64]:
    from dopplerdex.env import synthetic

    if synthetic.FLAT:
        return np.full(np.broadcast(np.asarray(lat), np.asarray(lon)).shape, synthetic.FLAT["geoid"])
    z, tf = _grid()
    lat = np.asarray(lat, dtype=np.float64)
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    c = (lon - tf.c) / tf.a - 0.5
    r = (lat - tf.f) / tf.e - 0.5
    r0 = np.clip(np.floor(r).astype(int), 0, z.shape[0] - 2)
    c0 = np.floor(c).astype(int) % z.shape[1]
    c1 = (c0 + 1) % z.shape[1]
    fr, fc = np.clip(r - r0, 0, 1), c - np.floor(c)
    return (
        z[r0, c0] * (1 - fr) * (1 - fc)
        + z[r0, c1] * (1 - fr) * fc
        + z[r0 + 1, c0] * fr * (1 - fc)
        + z[r0 + 1, c1] * fr * fc
    )
