"""Step 2 of the web flow: run the blind localisation pipeline and package results for the map."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from dopplerdex.adsb.store import load_points
from dopplerdex.audio.events import DetectConfig, detect_events
from dopplerdex.audio.spectrogram import load_audio, spectrogram, whiten
from dopplerdex.physics.atmosphere import speed_of_sound
from dopplerdex.search.locate import locate
from dopplerdex.web.analysis import _jpg
from dopplerdex.web.data import status as data_status
from dopplerdex.web.regions import search_region


def parse_tz(spec: str | None) -> tzinfo:
    """'UTC+01:00' / 'UTC-03:30' / 'UTC' fixed offsets (as chosen in the web form) or an IANA zone name."""
    spec = (spec or "UTC").strip()
    m = re.fullmatch(r"UTC(?:([+-])(\d{1,2})(?::?(\d{2}))?)?", spec)
    if m:
        if not m.group(1):
            return UTC
        sign = 1 if m.group(1) == "+" else -1
        return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)))
    return ZoneInfo(spec)


def time_window(when: dict, duration_s: float) -> tuple[float, float, list[str]]:
    """Recording-start window (unix s) and the UTC dates of flight data it needs."""
    tz = parse_tz(when.get("tz"))
    date = when["date"]
    if when.get("time"):
        start = datetime.fromisoformat(f"{date} {when['time']}").replace(tzinfo=tz).timestamp()
        unc = 60.0 * float(when.get("uncertainty_min") or 30.0)
        lo, hi = start - unc, start + unc
    else:  # the whole local day
        lo = datetime.fromisoformat(date).replace(tzinfo=tz).timestamp()
        hi = lo + 86400
    days = set()
    t = lo - 3600
    while t <= hi + duration_s + 3600:
        days.add(datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d"))
        t += 3600
    return lo, hi, sorted(days)


def needed_data(when: dict, where: dict, duration_s: float) -> dict:
    region, info = search_region(where)
    lo, hi, days = time_window(when, duration_s)
    per_day = {d: data_status(d, info["adsb_store"]) for d in days}
    worst = "ready"
    for st in per_day.values():
        if ["ready", "raw", "remote", "missing"].index(st["state"]) > ["ready", "raw", "remote", "missing"].index(
            worst
        ):
            worst = st["state"]
    return {"region": info, "window": {"lo": lo, "hi": hi, "hours": (hi - lo) / 3600}, "days": per_day, "state": worst}


def _overlay(x, fs, ev, t0, lat, lon, icao, c) -> str | None:
    from dopplerdex.scoring.consistency import predicted_passes_at
    from dopplerdex.viz.plots import dark_fig, plot_spectrogram

    passes = {p.flight.icao: p for p in predicted_passes_at(lat, lon, t0, len(x) / fs, c)}
    p = passes.get(icao)
    if p is None:
        return None
    tc = ev.fit.t_cpa_heard
    sp = whiten(spectrogram(x, fs, nperseg=8192, hop=1024)).time_slice(tc - 100, tc + 100).band(40, 700)
    fig, ax = dark_fig(1, 2, figsize=(16, 4.4))
    for a in ax:
        plot_spectrogram(a, sp, vmin=0, vmax=14)
    tt = p.t_r - t0
    m = p.valid & (tt > tc - 100) & (tt < tc + 100)
    for r in [ev.ridge, *ev.companions]:
        ratio = np.median(r.f / np.interp(r.t, tt, p.doppler))
        ax[1].plot(tt[m], ratio * p.doppler[m], color="#4cc9f0", lw=1.0, ls="--")
    ax[0].set_title(f"your recording around {tc:.0f} s", loc="left", fontsize=10)
    ax[1].set_title(f"prediction at the found spot: {icao} {p.flight.type} (closest {p.r_cpa / 1000:.1f} km, "
                    f"{p.speed_cpa:.0f} m/s)", loc="left", fontsize=10)  # fmt: skip
    return _jpg(fig)


def run_search(path: str, meta: dict, progress=None) -> dict:
    when, where = meta["when"], meta["where"]
    temp_c = float(meta.get("temp_c") or 15.0)
    c = float(speed_of_sound(temp_c, 70.0))
    x, fs = load_audio(path, 8000)
    duration = len(x) / fs
    region, info = search_region(where)
    lo, hi, _ = time_window(when, duration)
    events, _ = detect_events(x, fs, DetectConfig(c=c))
    res = locate(x, fs, lo, hi, c, region, events=events, progress=progress)
    out: dict = {"region": info, "window": {"lo": lo, "hi": hi}, "stats": {k: v for k, v in res.stats.items() if k != "timing"},
                 "pairs": res.pairs_tried, "candidates": [], "best": None, "tracks": [], "overlays": []}  # fmt: skip
    r = res.ranked
    if r.is_empty():
        out["message"] = "No hypothesis: not enough usable aircraft events (see the audio check)."
        return out
    rows = []
    for row in r.head(25).iter_rows(named=True):
        la = row.get("lat_ref") if row.get("lat_ref") is not None else row["lat"]
        lo_ = row.get("lon_ref") if row.get("lon_ref") is not None else row["lon"]
        rows.append({"rank": row["rank"], "lat": la, "lon": lo_, "t0": row["t0_fit"],
                     "t0_utc": datetime.fromtimestamp(row["t0_fit"], UTC).strftime("%Y-%m-%d %H:%M:%S"),
                     "combined": row.get("combined"), "loud": row.get("loud_corr"), "spectro": row.get("spectro"),
                     "ridge_cents": row.get("score"), "flight_a": f"{row.get('icao')} {row.get('type') or ''}".strip(),
                     "flight_b": f"{row.get('icao_b')} {row.get('type_b') or ''}".strip() if row.get("icao_b") else "", "pair_kind": row.get("pair_kind"),
                     "icao_a": row.get("icao"), "icao_b": row.get("icao_b"), "event_a": row.get("event_a"),
                     "event_b": row.get("event_b")})  # fmt: skip
    out["candidates"] = rows
    best = rows[0]
    out["best"] = best
    m = res.stats.get("loud_margin")
    out["confidence_margin"] = m
    # loudness-correlation gap between #1 and #2 (benchmark: true answers typically win by > 0.1)
    out["confidence"] = (
        "n/a"
        if m is None
        else "clear winner"
        if m > 0.1
        else "moderate"
        if m > 0.04
        else "weak: alternatives are close"
    )
    if res.stats.get("mode") == "single":
        out["confidence"] += " (single aircraft: the spot can slide along its flight path or mirror to the other side)"

    if progress:
        progress(0.97, "drawing results")
    # matched aircraft tracks around the best hypothesis
    for icao, ev_i in ((best["icao_a"], best["event_a"]), (best["icao_b"], best["event_b"])):
        if not icao:
            continue
        t_c = best["t0"] + (
            events[ev_i].fit.t_cpa_heard if best["pair_kind"] == "tonal" or icao == best["icao_a"] else 0
        )
        df = load_points(t_c - 900, t_c + 900, best["lat"], best["lon"], 60, region.adsb_region)
        df = df.filter(pl.col("icao") == icao).sort("t")
        if len(df):
            out["tracks"].append({"icao": icao, "points": np.round(df.select(["lat", "lon"]).to_numpy(), 5).tolist()})
    for icao, ev_i in (
        (best["icao_a"], best["event_a"]),
        (best["icao_b"], best["event_b"] if best["pair_kind"] == "tonal" else None),
    ):
        if icao and ev_i is not None:
            img = _overlay(x, fs, events[ev_i], best["t0"], best["lat"], best["lon"], icao, c)
            if img:
                out["overlays"].append(img)
    known = meta.get("known_location")
    if known and known.get("lat") is not None:
        kl, ko = float(known["lat"]), float(known["lon"])
        d = [
            float(np.hypot((c_["lat"] - kl) * 111320, (c_["lon"] - ko) * 111320 * np.cos(np.radians(kl))))
            for c_ in rows
        ]
        hit = [i for i, dd in enumerate(d) if dd < 1500]
        out["evaluation"] = {"best_error_m": round(d[0]), "rank_within_1500m": (hit[0] + 1) if hit else None,
                             "note": "known location used only for this comparison, after the search"}  # fmt: skip
    return out


def utc_now_date() -> str:
    return (datetime.now(UTC) - timedelta(hours=0)).strftime("%Y-%m-%d")
