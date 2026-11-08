"""Search orchestration: event constraints + ADS-B day -> coarse hypotheses (-> refinement / scoring)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from dopplerdex.env.dem import CoarseDEM
from dopplerdex.scoring.ridge_match import RidgeSet, score_hypotheses
from dopplerdex.search.coarse import Candidates, EventConstraint, generate, join_pairs
from dopplerdex.search.tracks import TrackSamples, load_chunk


@dataclass(frozen=True)
class SearchRegion:
    lat_min: float = 24.0
    lat_max: float = 50.0
    lon_min: float = -125.0
    lon_max: float = -66.0
    adsb_region: str = "north_america"  # which ingested ADS-B store covers this box

    @staticmethod
    def named(name: str) -> SearchRegion:
        return SEARCH_REGIONS[name]

    @staticmethod
    def bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float, adsb_region: str) -> SearchRegion:
        return SearchRegion(lat_min, lat_max, lon_min, lon_max, adsb_region)


SEARCH_REGIONS: dict[str, SearchRegion] = {
    "conus": SearchRegion(24.0, 50.0, -125.0, -66.0, "north_america"),
    "europe": SearchRegion(35.0, 71.0, -11.0, 34.0, "europe"),
    "serbia": SearchRegion(41.8, 46.3, 18.7, 23.1, "europe"),
}


@dataclass
class CoarseResult:
    hypotheses: pl.DataFrame  # one row per (coarse cell, flight A, flight B): lat, lon, t0, fid_a, fid_b, ...
    flights: pl.DataFrame  # flight table (global ids) with icao/type/class/time span
    stats: dict = field(default_factory=dict)


def coarse_search(
    ev_a: EventConstraint,
    ev_b: EventConstraint,
    t0_lo: float,
    t0_hi: float,
    c: float = 343.0,
    region: SearchRegion = SearchRegion(),
    window_s: float = 7200.0,
    step_m: float = 400.0,
    cell_m: float = 1000.0,
    dem: CoarseDEM | None = None,
    verbose: bool = True,
    ridges_a: RidgeSet | None = None,
    ridges_b: RidgeSet | None = None,
    mic_height: float = 2.0,
    keep_pairs_per_window: int = 3000,
    progress=None,  # optional callable(fraction in [0, 1], message)
) -> CoarseResult:
    """Enumerate (place, T0) hypotheses consistent with both events across the region and T0 range."""
    dem = dem or CoarseDEM(int(np.floor(region.lat_min)), int(np.ceil(region.lat_max)), int(np.floor(region.lon_min)),
                           int(np.ceil(region.lon_max)))  # fmt: skip
    tol = 3.0 * float(np.hypot(ev_a.t_sigma_cond, ev_b.t_sigma_cond))
    bbox = (region.lat_min - 0.2, region.lat_max + 0.2, region.lon_min - 0.3, region.lon_max + 0.3)
    t_min_ev, t_max_ev = min(ev_a.t_heard, ev_b.t_heard), max(ev_a.t_heard, ev_b.t_heard)
    out_rows: list[pl.DataFrame] = []
    flight_tabs: list[pl.DataFrame] = []
    fid_offset = 0
    stats = {"windows": 0, "samples": 0, "cand_a": 0, "cand_b": 0, "matches": 0, "seconds": 0.0, "tol_s": tol}
    t_start = time.time()
    timing = {"load": 0.0, "dem": 0.0, "generate": 0.0, "join": 0.0, "score": 0.0}
    for w0 in np.arange(t0_lo, t0_hi, window_s):
        w1 = min(w0 + window_s, t0_hi)
        _t = time.time()
        ts = load_chunk(w0 + t_min_ev - 1200, w1 + t_max_ev + 300, region.adsb_region, bbox)
        timing["load"] += time.time() - _t
        if len(ts) == 0:
            continue
        _t = time.time()
        elev = dem.sample(ts.lat, ts.lon)
        timing["dem"] += time.time() - _t
        _t = time.time()
        A = generate(ts, elev, ev_a, c, step_m, cell_m, (w0, w1), dedupe_bin_s=tol / 2)
        B = generate(ts, elev, ev_b, c, step_m, cell_m, (w0 - tol, w1 + tol), dedupe_bin_s=tol / 2)
        timing["generate"] += time.time() - _t
        _t = time.time()
        pm = join_pairs(A, B, tol)
        timing["join"] += time.time() - _t
        stats["windows"] += 1
        stats["samples"] += len(ts)
        stats["cand_a"] += len(A)
        stats["cand_b"] += len(B)
        stats["matches"] += len(pm)
        if len(pm):
            df = pl.DataFrame({
                "cell": A.cell[pm.a_idx], "lat": 0.5 * (A.lat[pm.a_idx] + B.lat[pm.b_idx]),
                "lon": 0.5 * (A.lon[pm.a_idx] + B.lon[pm.b_idx]), "t0": 0.5 * (A.t0[pm.a_idx] + B.t0[pm.b_idx]),
                "dt0": pm.dt0, "fid_a": A.fid[pm.a_idx].astype(np.int64) + fid_offset,
                "fid_b": B.fid[pm.b_idx].astype(np.int64) + fid_offset, "r_a": A.r[pm.a_idx], "r_b": B.r[pm.b_idx],
            })  # fmt: skip
            # only places inside the search region (tracks are loaded from a padded box)
            df = df.filter(pl.col("lat").is_between(region.lat_min, region.lat_max)
                           & pl.col("lon").is_between(region.lon_min, region.lon_max))  # fmt: skip
            # one hypothesis per (cell, flight pair): keep the best timing agreement
            df = df.sort(pl.col("dt0").abs()).unique(subset=["cell", "fid_a", "fid_b"], keep="first")
            _t = time.time()
            if ridges_a is not None and ridges_b is not None and len(df):
                # cascade: (1) one representative cell per flight pair, (2) all cells of the best pairs
                reps = df.sort(pl.col("dt0").abs()).unique(subset=["fid_a", "fid_b"], keep="first")
                reps = _score_window(reps, ts, fid_offset, dem, ridges_a, ridges_b, c, mic_height, tol, dt_steps=7)
                top = reps.sort("score").head(keep_pairs_per_window).select(["fid_a", "fid_b"])
                df = df.join(top, on=["fid_a", "fid_b"], how="semi")
                df = _score_window(df, ts, fid_offset, dem, ridges_a, ridges_b, c, mic_height, tol, dt_steps=13)
                stats["pairs_scored"] = stats.get("pairs_scored", 0) + len(reps)
            timing["score"] += time.time() - _t
            out_rows.append(df)
        flight_tabs.append(ts.flights.with_columns((pl.col("id") + fid_offset).alias("id")))
        fid_offset += len(ts.flights)
        if progress is not None:
            progress(min(1.0, (w1 - t0_lo) / max(t0_hi - t0_lo, 1.0)),
                     f"searched start times up to {time.strftime('%H:%M', time.gmtime(w1))} UTC ({stats['matches']:,} matches)")  # fmt: skip
        if verbose:
            print(f"  T0 {time.strftime('%H:%M', time.gmtime(w0))}-{time.strftime('%H:%M', time.gmtime(w1))}: "
                  f"{len(ts) / 1e6:5.2f} M samples, cand A {len(A) / 1e6:6.2f} M, B {len(B) / 1e6:6.2f} M, "
                  f"matches {len(pm):8d}  ({time.time() - t_start:5.0f} s)", flush=True)  # fmt: skip
    stats["seconds"] = time.time() - t_start
    stats["timing"] = {k: round(v, 1) for k, v in timing.items()}
    hyp = pl.concat(out_rows) if out_rows else pl.DataFrame()
    flights = pl.concat(flight_tabs) if flight_tabs else pl.DataFrame()
    stats["hypotheses"] = len(hyp)
    return CoarseResult(hyp, flights, stats)


def coarse_single(
    ev: EventConstraint,
    t0_lo: float,
    t0_hi: float,
    c: float = 343.0,
    region: SearchRegion = SearchRegion(),
    ridges: RidgeSet | None = None,
    window_s: float = 7200.0,
    step_m: float = 400.0,
    cell_m: float = 1000.0,
    dem: CoarseDEM | None = None,
    mic_height: float = 2.0,
    keep_groups_per_window: int = 400,
    verbose: bool = False,
    progress=None,
) -> CoarseResult:
    """Single-event search: every (place, T0) where one flight reproduces the one detected curve.

    Without a second event nothing pins the position along the two candidate strips beside each
    flight except the time window and whole-recording consistency, so this returns many cells per
    flight; the caller decides between them. Cascade: one representative per (flight, 2-min T0 bin)
    is ridge-scored first, then every cell of the best groups.
    """
    dem = dem or CoarseDEM(int(np.floor(region.lat_min)), int(np.ceil(region.lat_max)), int(np.floor(region.lon_min)),
                           int(np.ceil(region.lon_max)))  # fmt: skip
    rs = ridges if ridges is not None else RidgeSet.empty()
    tol = 3.0 * ev.t_sigma_cond
    bbox = (region.lat_min, region.lat_max, region.lon_min, region.lon_max)
    out_rows: list[pl.DataFrame] = []
    flight_tabs: list[pl.DataFrame] = []
    fid_offset = 0
    stats = {"windows": 0, "samples": 0, "cand_a": 0, "seconds": 0.0, "tol_s": tol, "mode": "single"}
    t_start = time.time()
    for w0 in np.arange(t0_lo, t0_hi, window_s):
        w1 = min(w0 + window_s, t0_hi)
        ts = load_chunk(w0 + ev.t_heard - 1200, w1 + ev.t_heard + 300, region.adsb_region, bbox)
        if len(ts) == 0:
            continue
        elev = dem.sample(ts.lat, ts.lon)
        A = generate(ts, elev, ev, c, step_m, cell_m, (w0, w1), dedupe_bin_s=tol)
        stats["windows"] += 1
        stats["samples"] += len(ts)
        stats["cand_a"] += len(A)
        if len(A):
            inside = (
                (A.lat >= region.lat_min)
                & (A.lat <= region.lat_max)
                & (A.lon >= region.lon_min)
                & (A.lon <= region.lon_max)
            )
            idx = np.flatnonzero(inside)
            fid = A.fid[idx].astype(np.int64) + fid_offset
            df = pl.DataFrame({"cell": A.cell[idx], "lat": A.lat[idx], "lon": A.lon[idx], "t0": A.t0[idx], "dt0": np.zeros(idx.size),
                               "fid_a": fid, "fid_b": fid, "r_a": A.r[idx]})  # fmt: skip
            df = df.with_columns((pl.col("t0") // 120).cast(pl.Int64).alias("tbin"))
            if len(df) and rs.t.size:
                reps = df.unique(subset=["fid_a", "tbin"], keep="first")
                reps = _score_window(reps, ts, fid_offset, dem, rs, RidgeSet.empty(), c, mic_height, tol, dt_steps=7)
                top = reps.sort("score").head(keep_groups_per_window).select(["fid_a", "tbin"])
                df = df.join(top, on=["fid_a", "tbin"], how="semi")
                df = _score_window(df, ts, fid_offset, dem, rs, RidgeSet.empty(), c, mic_height, tol, dt_steps=13)
                stats["groups_scored"] = stats.get("groups_scored", 0) + len(reps)
            out_rows.append(df.drop("tbin"))
        flight_tabs.append(ts.flights.with_columns((pl.col("id") + fid_offset).alias("id")))
        fid_offset += len(ts.flights)
        if progress is not None:
            progress(min(1.0, (w1 - t0_lo) / max(t0_hi - t0_lo, 1.0)),
                     f"searched start times up to {time.strftime('%H:%M', time.gmtime(w1))} UTC")  # fmt: skip
        if verbose:
            print(f"  single T0 {time.strftime('%H:%M', time.gmtime(w0))}: {len(ts) / 1e6:5.2f} M samples, "
                  f"cand {len(A) / 1e6:6.2f} M ({time.time() - t_start:5.0f} s)", flush=True)  # fmt: skip
    stats["seconds"] = time.time() - t_start
    hyp = pl.concat(out_rows) if out_rows else pl.DataFrame()
    flights = pl.concat(flight_tabs) if flight_tabs else pl.DataFrame()
    stats["hypotheses"] = len(hyp)
    return CoarseResult(hyp, flights, stats)


def _track_arrays(ts: TrackSamples):
    """Contiguous per-flight index ranges and vertical rates for the scoring kernel."""
    fid = ts.fid
    n_f = int(fid.max()) + 1 if fid.size else 0
    starts = np.searchsorted(fid, np.arange(n_f), side="left")
    ends = np.searchsorted(fid, np.arange(n_f), side="right")
    vu = np.zeros_like(ts.alt_msl)
    same = fid[1:] == fid[:-1]
    d = np.diff(ts.alt_msl) / np.maximum(np.diff(ts.t), 1e-3)
    vu[1:][same] = d[same]
    return starts.astype(np.int64), ends.astype(np.int64), vu


def _score_window(
    df, ts, fid_offset, dem, ra: RidgeSet, rb: RidgeSet, c, mic_height, tol, dt_steps: int = 9
) -> pl.DataFrame:
    starts, ends, vu = _track_arrays(ts)
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    z_mic = dem.sample(lat, lon) + mic_height
    dts = np.linspace(-tol / 2, tol / 2, dt_steps)
    rms_a, rms_b, dt = score_hypotheses(
        lat, lon, z_mic, df["t0"].to_numpy(), (df["fid_a"].to_numpy() - fid_offset).astype(np.int64),
        (df["fid_b"].to_numpy() - fid_offset).astype(np.int64), ts.t, ts.lat, ts.lon, ts.alt_msl, ts.gs, ts.trk, vu,
        starts, ends, ra.t, ra.lf, ra.w, ra.seg, ra.n_seg, rb.t, rb.lf, rb.w, rb.seg, rb.n_seg, c, dts, 0.03, 0.6,
    )  # fmt: skip
    na, nb = ra.t.size, rb.t.size
    score = np.sqrt((rms_a**2 * na + rms_b**2 * nb) / max(na + nb, 1))  # combined RMS (cents): lower is better
    return df.with_columns(pl.Series("rms_a", rms_a), pl.Series("rms_b", rms_b), pl.Series("dt", dt),
                           pl.Series("score", score), (pl.col("t0") + pl.Series(dt)).alias("t0_fit"))  # fmt: skip


def candidates_near(cand: Candidates, lat: float, lon: float, radius_m: float) -> np.ndarray:
    d = np.hypot((cand.lat - lat) * 111320.0, (cand.lon - lon) * 111320.0 * np.cos(np.radians(lat)))
    return np.flatnonzero(d < radius_m)
