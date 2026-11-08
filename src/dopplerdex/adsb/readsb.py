"""Parse readsb `trace_full_*.json` files (adsb.lol / ADS-B Exchange globe_history format).

Trace point layout (readsb README-json):
  [0] seconds after file timestamp  [1] lat  [2] lon  [3] baro alt (ft) or "ground"
  [4] ground speed (kt)  [5] track (deg)  [6] flags  [7] vertical rate (ft/min)
  [8] details dict or null  [9] source type  [10] geometric alt (ft)  [11] geom rate (ft/min)
  [12] IAS (kt)  [13] roll (deg)
Flags: 1 stale, 2 new leg, 4 vertical rate is geometric, 8 altitude is geometric.
"""

from __future__ import annotations

import gzip

import numpy as np
import orjson

FT = 0.3048
KT = 0.514444
FPM = FT / 60.0

COLUMNS = (
    "t", "lat", "lon", "alt_baro_m", "alt_geom_m", "on_ground", "gs_ms", "track_deg", "baro_rate_ms",
    "geom_rate_ms", "flags", "src", "leg", "wind_dir", "wind_ms", "oat_c",
)  # fmt: skip

SOURCES = {"adsb_icao": 0, "adsb_icao_nt": 1, "adsr_icao": 2, "tisb_icao": 3, "adsc": 4, "mlat": 5, "other": 6,
           "mode_s": 7, "adsb_other": 8, "adsr_other": 9, "tisb_trackfile": 10, "tisb_other": 11, "mode_ac": 12,
           "unknown": 13}  # fmt: skip


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else np.nan


def parse_trace(raw: bytes) -> tuple[dict, dict[str, np.ndarray]] | None:
    """Return (header, columns) for one trace file, or None if empty/corrupt."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    try:
        d = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None
    tr = d.get("trace") or []
    if not tr:
        return None
    n = len(tr)
    t0 = float(d.get("timestamp", 0.0))
    t = np.empty(n)
    lat = np.empty(n)
    lon = np.empty(n)
    alt_b = np.full(n, np.nan)
    alt_g = np.full(n, np.nan)
    gnd = np.zeros(n, dtype=bool)
    gs = np.full(n, np.nan)
    trk = np.full(n, np.nan)
    br = np.full(n, np.nan)
    gr = np.full(n, np.nan)
    flags = np.zeros(n, dtype=np.int16)
    src = np.full(n, 13, dtype=np.int8)
    wd = np.full(n, np.nan)
    ws = np.full(n, np.nan)
    oat = np.full(n, np.nan)
    for i, p in enumerate(tr):
        t[i] = t0 + p[0]
        lat[i] = p[1]
        lon[i] = p[2]
        a = p[3]
        if a == "ground":
            gnd[i] = True
        else:
            alt_b[i] = _num(a) * FT
        gs[i] = _num(p[4]) * KT
        trk[i] = _num(p[5])
        flags[i] = p[6] or 0
        br[i] = _num(p[7]) * FPM
        det = p[8]
        if len(p) > 9:
            src[i] = SOURCES.get(p[9] or "unknown", 6)
            alt_g[i] = _num(p[10]) * FT
            gr[i] = _num(p[11]) * FPM
        if det:
            wd[i] = _num(det.get("wd"))
            ws[i] = _num(det.get("ws")) * KT
            oat[i] = _num(det.get("oat"))
    leg = np.cumsum((flags & 2) > 0).astype(np.int16)
    cols = dict(
        t=t, lat=lat, lon=lon, alt_baro_m=alt_b, alt_geom_m=alt_g, on_ground=gnd, gs_ms=gs, track_deg=trk,
        baro_rate_ms=br, geom_rate_ms=gr, flags=flags, src=src, leg=leg, wind_dir=wd, wind_ms=ws, oat_c=oat,
    )  # fmt: skip
    hdr = {k: d.get(k) for k in ("icao", "r", "t", "desc", "dbFlags", "ownOp", "year")}
    return hdr, cols
