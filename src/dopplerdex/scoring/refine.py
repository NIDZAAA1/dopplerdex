"""Continuous refinement of top hypotheses with the full physics model.

Compared with the coarse kernel this uses: raw ADS-B tracks (cubic Hermite with reported velocities),
full-resolution terrain, the weather profile (path-averaged sound speed, wind on the ray, barometric
altitude correction), and a continuous optimisation of (east, north, clock offset).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dopplerdex.adsb.store import Flight, build_flights, load_points
from dopplerdex.audio.events import DetectedEvent
from dopplerdex.env.dem import fine_elevation
from dopplerdex.env.geoid import undulation
from dopplerdex.env.weather import Profile, profile
from dopplerdex.geo.frames import LocalFrame
from dopplerdex.physics.doppler import solve_retarded
from dopplerdex.scoring.ridge_match import CENTS, RidgeSet


@dataclass
class Refined:
    lat: float
    lon: float
    t0: float
    rms_a: float  # cents
    rms_b: float
    score: float  # combined RMS (cents)
    icao_a: str
    icao_b: str
    success: bool


def _pick_flight(flights: list[Flight], icao: str, t_mid: float) -> Flight | None:
    cand = [f for f in flights if f.icao == icao and f.traj.t_start - 120 <= t_mid <= f.traj.t_end + 120]
    return max(cand, key=lambda f: f.n_points) if cand else None


def _rms(rs: RidgeSet, logd: np.ndarray, ok: np.ndarray, clip: float = 0.03) -> tuple[float, int]:
    tot, tw, n = 0.0, 0.0, 0
    for s in range(rs.n_seg):
        m = (rs.seg == s) & ok
        if m.sum() < 3:
            continue
        d = rs.lf[m] - logd[m]
        w = rs.w[m]
        mu = np.sum(w * d) / np.sum(w)
        mu += np.average(np.clip(d - mu, -3 * clip, 3 * clip), weights=w)
        r = np.clip(d - mu, -clip, clip)
        tot += np.sum(w * r * r)
        tw += np.sum(w)
        n += int(m.sum())
    return (np.sqrt(tot / tw) if tw > 0 else 1.0), n


def refine_hypothesis(
    lat: float,
    lon: float,
    t0: float,
    icao_a: str | None,
    icao_b: str | None,
    ev_a: DetectedEvent | None,
    ev_b: DetectedEvent | None,
    mic_height: float = 2.0,
    use_weather: bool = True,
    region: str = "north_america",
    dt_sigma: float | None = None,
    extra: list[tuple[str, DetectedEvent]] | None = None,
    mp_extra: list | None = None,
) -> Refined | None:
    """Refine (east, north, clock offset) against every matched curve: the search pair (ev_a, ev_b) plus
    any further explained events (extra = [(icao, event), ...]), all sharing one clock.

    ev_b=None and no extra: single-event refinement. One curve does not pin the position along the
    flight (moving along the track is equivalent to shifting the clock), so the clock offset gets a
    Gaussian penalty (dt_sigma, default 2 s) and the position stays where the consistency put it.

    mp_extra = [(icao, MultipathEvent), ...]: ground-reflection delay tracks; each contributes its
    scale-fitted 2 h sin(elevation) / c misfit, weighted as ~3x looser than a tonal curve."""
    matches = ([(icao_a, ev_a)] if ev_a is not None else []) + ([(icao_b, ev_b)] if ev_b is not None else [])
    matches += list(extra or [])
    mp_extra = list(mp_extra or [])
    if len(matches) + len(mp_extra) == 0:
        return None
    if len(matches) + len(mp_extra) == 1:
        dt_sigma = 2.0 if dt_sigma is None else dt_sigma
    sets = [RidgeSet.from_event(ev) for _, ev in matches]
    spans = [(rs.t.min(), rs.t.max(), rs.t.mean()) for rs in sets] + [
        (m.t[0], m.t[-1], m.t.mean()) for _, m in mp_extra
    ]
    t_lo = t0 + min(a for a, _, _ in spans) - 900
    t_hi = t0 + max(b for _, b, _ in spans) + 600
    n_geo = float(undulation(lat, lon))
    prof: Profile | None = None
    baro_off = 0.0
    ground0 = float(fine_elevation(lat, lon)[0])
    if use_weather:
        try:
            prof = profile(lat, lon, t0 + float(np.mean([m for _, _, m in spans])), ground0)
            baro_off = float(prof.baro_to_msl(0.0))
        except Exception:  # network/cache failure: fall back to standard atmosphere
            prof = None
    df = load_points(t_lo, t_hi, lat, lon, 30.0, region)
    if df.is_empty():
        return None
    import polars as pl

    df = df.filter(pl.col("icao").is_in(list({icao for icao, _ in matches} | {icao for icao, _ in mp_extra})))
    frame = LocalFrame(lat, lon, 0.0)
    flights = build_flights(df, frame, geoid_n=n_geo + baro_off)
    legs = []
    for (icao, _), rs in zip(matches, sets, strict=True):
        fl = _pick_flight(flights, icao, t0 + rs.t.mean())
        if fl is None:
            if icao in (icao_a, icao_b) and icao is not None:
                return None  # the search pair itself must be reconstructable
            continue
        if prof is None:
            c_, w_ = 343.0, np.zeros(3)
        else:
            pnt, _ = fl.traj.state(np.array([t0 + rs.t.mean()]))
            z_msl = pnt[0, 2] - n_geo
            c_, w_ = prof.c_path(z_msl), prof.wind_path(z_msl)
        legs.append((fl, rs, c_, w_))
    mp_legs = []
    for icao, mev in mp_extra:
        fl = _pick_flight(flights, icao, t0 + float(mev.t.mean()))
        if fl is not None:
            mp_legs.append((fl, mev))
    coslat = np.cos(np.radians(lat))

    def mic_at(e: float, n: float) -> np.ndarray:
        la, lo = lat + n / 111320.0, lon + e / (111320.0 * coslat)
        g = float(fine_elevation(la, lo)[0])
        return frame.to_enu(la, lo, g + n_geo + mic_height)

    C_T, C_M = 0.012, 0.04  # robust scales: ~20 cents (tonal), 4 % of the peak delay (ground reflection)

    def robust(r: float, c_: float) -> float:
        """Cauchy loss: ~r^2 for good fits, bounded growth for a mismatched aircraft."""
        return c_**2 * np.log1p((r / c_) ** 2)

    def objective(x: np.ndarray, details: bool = False):
        e, n, dt = x
        mic = mic_at(e, n)
        tot, cnt, per = 0.0, 0, []
        for fl, rs, c_, w_ in legs:
            sol = solve_retarded(fl.traj, mic, t0 + dt + rs.t, c_, w_)
            r, k = _rms(rs, np.log(sol.doppler), sol.valid)
            per.append(r)
            if k < 0.5 * rs.t.size:
                return (1.0, per) if details else 1.0
            tot += robust(r, C_T) * k
            cnt += k
        # ground-reflection passes: one microphone, so ONE height (delay scale k = 2 h / c) for all of them
        mp = []
        for fl, mev in mp_legs:
            sol = solve_retarded(fl.traj, mic, t0 + dt + mev.t, 343.0, np.zeros(3))
            sel = sol.valid
            if sel.mean() < 0.6:
                continue
            mp.append(((sol.src_pos[sel, 2] - mic[2]) / np.maximum(sol.rng[sel], 1.0), mev.tau[sel], mev.w[sel]))
        if mp:
            k_ = sum(float(np.sum(w * y * s_)) for s_, y, w in mp) / max(
                sum(float(np.sum(w * s_**2)) for s_, y, w in mp), 1e-30
            )
            for s_, y, w in mp:
                pred = k_ * s_
                r = float(np.sqrt(np.average(((y - pred) / max(float(pred.max()), 1e-6)) ** 2, weights=w)))
                tot += 0.09 * robust(r, C_M) * y.size  # 0.3^2: a 2 % delay misfit ~ a 10-cent tonal residual
                cnt += y.size
        val = tot / max(cnt, 1)
        if dt_sigma:  # 3e-5 ~ (10 cents)^2: one sigma of clock shift costs about a 10-cent residual
            val += 3e-5 * (dt / dt_sigma) ** 2
        return (val, per) if details else val

    best = None
    for x0 in ([0.0, 0.0, 0.0], [300.0, 0.0, 0.0], [0.0, 300.0, 0.0], [-300.0, -300.0, 0.0]):
        res = minimize(objective, np.array(x0), method="Nelder-Mead",
                       options={"initial_simplex": np.array(x0) + np.array([[0, 0, 0], [400, 0, 0], [0, 400, 0], [0, 0, 4]]),
                                "xatol": 5.0, "fatol": 1e-10, "maxiter": 400})  # fmt: skip
        if best is None or res.fun < best.fun:
            best = res
    e, n, dt = best.x
    val, per = objective(best.x, details=True)
    r_a = per[0] if per else float(np.sqrt(val))
    r_b = per[1] if len(per) > 1 else r_a
    return Refined(lat + n / 111320.0, lon + e / (111320.0 * coslat), t0 + dt, r_a * CENTS, r_b * CENTS,
                   float(np.sqrt(val)) * CENTS, icao_a or "", icao_b or icao_a or "", bool(best.success))  # fmt: skip
