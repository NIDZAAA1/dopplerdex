"""Single-aircraft mode helpers: candidate selection and the strip-centre estimate."""

import numpy as np
import polars as pl

from dopplerdex.search.locate import _single_candidates, _strip_centre


def _strip(fid: int, lat0: float, n: int, score: float, side: float = 1.0) -> pl.DataFrame:
    # cells along a north-south strip, east of (side=+1) or west of (side=-1) a flight
    return pl.DataFrame({"fid_a": [fid] * n, "lat": lat0 + np.arange(n) * 0.005, "lon": [side * 0.04] * n,
                         "t0_fit": 1000.0 + np.arange(n) * 4.0, "score": [score] * n, "r_a": [3500.0] * n})  # fmt: skip


def test_single_candidates_keeps_best_flights_and_spreads_cells():
    h = pl.concat([_strip(1, 40.0, 100, 15.0), _strip(2, 40.0, 100, 18.0), _strip(3, 40.0, 100, 60.0)])
    out = _single_candidates(h, n_flights=2, cells_per_flight=10)
    assert set(out["fid_a"].unique()) == {1, 2}
    g = out.filter(pl.col("fid_a") == 1).sort("t0_fit")
    assert len(g) == 10
    assert g["t0_fit"].min() == 1000.0 and g["t0_fit"].max() == 1000.0 + 99 * 4.0  # covers the whole strip


def test_strip_centre_averages_one_side_only():
    east, west = _strip(1, 40.0, 9, 15.0, 1.0), _strip(1, 40.0, 9, 15.0, -1.0)
    cand = pl.concat([east, west]).with_columns(pl.lit(1.0).alias("combined"))
    cand = pl.concat([cand.filter(pl.col("lon") > 0).slice(4, 1), cand])  # winner: middle of the east strip
    out = _strip_centre(cand, "combined")
    assert out["lon"][0] > 0  # never pulled towards the mirror strip
    assert abs(out["lat"][0] - (40.0 + 4 * 0.005)) < 0.003  # centre of the east strip within 4 km
