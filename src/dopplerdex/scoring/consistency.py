"""Whole-recording consistency of a (place, T0) hypothesis with ALL traffic, not just the matched events.

At the true place and time every aircraft's predicted loudness must agree with the recording's
loudness envelope (louder when aircraft are close, quiet when none are). This is the soft version of
the reference video's "silence filter", and it also rewards the loud passes the tonal detector
ignores (e.g. jets on approach).
"""

from __future__ import annotations

import numpy as np

from dopplerdex.adsb.store import build_flights, load_points
from dopplerdex.env.dem import fine_elevation
from dopplerdex.env.geoid import undulation
from dopplerdex.geo.frames import LocalFrame
from dopplerdex.search.loudness import _excess, predicted_level
from dopplerdex.search.oracle import predict_passes


def predicted_passes_at(lat: float, lon: float, t0: float, dur: float, c: float = 343.0, mic_height: float = 2.0,
                        radius_km: float = 15.0, region: str = "north_america"):  # fmt: skip
    """All predicted passes (any class) at a hypothesised place over the recording."""
    n_geo = float(undulation(lat, lon))
    ground = float(fine_elevation(lat, lon)[0])
    frame = LocalFrame(lat, lon, 0.0)
    df = load_points(t0 - 900, t0 + dur + 600, lat, lon, radius_km, region)
    flights = build_flights(df, frame, geoid_n=n_geo)
    mic = np.array([0.0, 0.0, ground + n_geo + mic_height])
    return predict_passes(flights, mic, c, max_range=12_000)


def loudness_from_passes(t_rec, level_rec, passes, t0: float, max_shift_s: float = 6.0) -> float:
    dur = float(t_rec[-1] - t_rec[0])
    if not passes:
        return 0.0
    obs = _excess(np.interp(np.arange(0, dur, 1.0), t_rec - t_rec[0], level_rec), 601)
    obs = (obs - obs.mean()) / (obs.std() + 1e-9)
    best = -1.0
    for s in np.arange(-max_shift_s, max_shift_s + 0.5, 1.0):
        pred = _excess(predicted_level(passes, t0 + s + np.arange(0, dur, 1.0)), 601)
        if pred.std() < 1e-6:
            continue
        pz = (pred - pred.mean()) / pred.std()
        best = max(best, float(np.mean(pz * obs)))
    return best


def loudness_consistency(
    t_rec: np.ndarray,
    level_rec: np.ndarray,
    lat: float,
    lon: float,
    t0: float,
    c: float = 343.0,
    mic_height: float = 2.0,
    max_shift_s: float = 6.0,
    radius_km: float = 15.0,
    region: str = "north_america",
) -> tuple[float, int]:
    """Max normalised correlation (over a small clock shift) between observed and predicted loudness
    excess. Returns (corr, number of passes within 12 km)."""
    dur = float(t_rec[-1] - t_rec[0])
    n_geo = float(undulation(lat, lon))
    ground = float(fine_elevation(lat, lon)[0])
    frame = LocalFrame(lat, lon, 0.0)
    df = load_points(t0 - 900, t0 + dur + 600, lat, lon, radius_km, region)
    flights = build_flights(df, frame, geoid_n=n_geo)
    mic = np.array([0.0, 0.0, ground + n_geo + mic_height])
    passes = predict_passes(flights, mic, c, max_range=12_000)
    if not passes:
        return 0.0, 0
    obs = _excess(np.interp(np.arange(0, dur, 1.0), t_rec - t_rec[0], level_rec), 601)
    obs = (obs - obs.mean()) / (obs.std() + 1e-9)
    best = -1.0
    for s in np.arange(-max_shift_s, max_shift_s + 0.5, 1.0):
        pred = _excess(predicted_level(passes, t0 + s + np.arange(0, dur, 1.0)), 601)
        if pred.std() < 1e-6:
            continue
        pz = (pred - pred.mean()) / pred.std()
        best = max(best, float(np.mean(pz * obs)))
    return best, len(passes)


def explained_events(passes, events, t0: float, max_shift_s: float = 6.0, thr_cents: float = 30.0,
                     win_s: float = 40.0, mp_events=(), thr_mp: float = 0.06):  # fmt: skip
    """Observed -> predicted check: how many detected aircraft events does this (place, T0) explain?

    Tonal event: some predicted pass (any flight) peaks within win_s of the event's heard CPA and its
    Doppler curve reproduces the ridges to within thr_cents RMS (per-ridge constant removed).
    Ground-reflection (multipath) event: the predicted 2 h sin(elevation) / c curve, with the unknown
    mic height h fitted, reproduces the delay track to within thr_mp (RMS / peak delay).
    One clock shift (+-max_shift_s) is shared by all events. Returns (count, [(kind, index, icao,
    misfit), ...]) with kind "t" (tonal, misfit in cents) or "m" (multipath, relative misfit).
    """
    from dopplerdex.audio.multipath import predicted_ratio_rms
    from dopplerdex.scoring.refine import _rms
    from dopplerdex.scoring.ridge_match import CENTS, RidgeSet

    items = [("t", i) for i, e in enumerate(events) if e.tier != "G"] + [("m", j) for j in range(len(mp_events))]
    if not items:
        return 0, []
    shifts = np.arange(-max_shift_s, max_shift_s + 0.5, 1.0)
    best = np.full((len(items), shifts.size), np.inf)  # misfit / threshold
    who: dict[tuple[int, int], str] = {}
    for a, (kind, i) in enumerate(items):
        if kind == "t":
            e = events[i]
            rs = RidgeSet.from_event(e)
            tc, t_rel, thr = t0 + e.fit.t_cpa_heard, rs.t, thr_cents
        else:
            e = mp_events[i]
            tc, t_rel, thr = t0 + e.t_cpa, e.t, thr_mp
        for p in passes:
            if not np.isfinite(p.t_cpa_heard) or abs(p.t_cpa_heard - tc) > win_s:
                continue
            m = p.valid
            if m.sum() < 4:
                continue
            tr = p.t_r[m]
            if kind == "t":
                lp = np.log(p.doppler[m])
            for b, s in enumerate(shifts):
                tt = t0 + s + t_rel
                if kind == "t":
                    ok = (tt >= tr[0]) & (tt <= tr[-1])
                    if ok.mean() < 0.6:
                        continue
                    r, _ = _rms(rs, np.interp(tt, tr, lp), ok)
                    mis = r * CENTS
                else:
                    if p.sin_el is None:
                        continue
                    mis = predicted_ratio_rms(e, tt, p.t_r, p.sin_el, p.valid)
                if mis / thr < best[a, b]:
                    best[a, b] = mis / thr
                    who[(a, b)] = p.flight.icao
    hits = best < 1.0
    b = int(np.argmax(hits.sum(axis=0) - 1e-6 * np.nanmin(np.where(hits, best, 1e3), axis=0)))
    out = [(items[a][0], items[a][1], who[(a, b)], float(best[a, b] * (thr_cents if items[a][0] == "t" else thr_mp)))
           for a in range(len(items)) if hits[a, b]]  # fmt: skip
    return len(out), out
