"""Blind localisation of ANY recording (same pipeline for every input).

uv run python scripts/locate_recording.py <audio> --date 2026-11-07 --region serbia \
    [--start "2026-11-07 17:05" --tz Europe/Belgrade --window-min 30] [--temp-c 12] [--out DIR]

--start/--window-min: approximate recording start (local) and +- uncertainty; omit to search the whole UTC day.
Region: a preset name (conus, europe, serbia) or "lat_min,lat_max,lon_min,lon_max,adsb_store".
The ADS-B day archive must be ingested for the region's store (scripts/ingest_adsb.py DATE --region STORE).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dopplerdex.audio.spectrogram import load_audio
from dopplerdex.physics.atmosphere import speed_of_sound
from dopplerdex.search.locate import locate
from dopplerdex.search.pipeline import SEARCH_REGIONS, SearchRegion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--date", required=True, help="UTC date of the recording (YYYY-MM-DD)")
    ap.add_argument("--region", default="conus")
    ap.add_argument("--start", default=None, help='approximate local start, e.g. "2026-11-07 17:05"')
    ap.add_argument("--tz", default="UTC")
    ap.add_argument("--window-min", type=float, default=30.0, help="+- uncertainty of --start (minutes)")
    ap.add_argument("--temp-c", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.region in SEARCH_REGIONS:
        region = SEARCH_REGIONS[args.region]
    else:
        a, b, c_, d, store = args.region.split(",")
        region = SearchRegion.bbox(float(a), float(b), float(c_), float(d), store)
    if args.start:
        t0 = datetime.fromisoformat(args.start).replace(tzinfo=ZoneInfo(args.tz)).timestamp()
        t0_lo, t0_hi = t0 - 60 * args.window_min, t0 + 60 * args.window_min
    else:
        day = datetime.fromisoformat(args.date).replace(tzinfo=UTC).timestamp()
        t0_lo, t0_hi = day - 3600, day + 86400
    c = float(speed_of_sound(args.temp_c, 70.0))
    x, fs = load_audio(args.audio, 8000)
    res = locate(x, fs, t0_lo, t0_hi, c, region, verbose=True)
    out = Path(args.out or f"data/analysis/locate_{Path(args.audio).stem[:30]}")
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(res.events)} events: {[(round(e.fit.t_cpa_heard), e.tier, round(e.fit.v), round(e.fit.r_cpa)) for e in res.events]}"
    )
    print(f"pairs tried: {res.pairs_tried}; stats: {res.stats}")
    if len(res.ranked):
        cols = [k for k in ("rank", "lat_ref", "lon_ref", "lat", "lon", "t0_fit", "score", "loud_corr", "spectro", "combined",
                            "icao", "type", "icao_b", "type_b", "pair_kind") if k in res.ranked.columns]  # fmt: skip
        top = res.ranked.select(cols).head(15)
        top.write_csv(out / "ranked.csv")
        print(top)
        b = res.best
        when = datetime.fromtimestamp(b["t0_fit"], UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"BEST: {b.get('lat_ref') or b['lat']:.5f}, {b.get('lon_ref') or b['lon']:.5f}  recording start {when}  "
              f"(confidence margin {res.stats.get('loud_margin')})")  # fmt: skip
    else:
        print("No hypothesis: not enough usable aircraft events in this recording.")
    (out / "summary.json").write_text(json.dumps({"stats": res.stats, "pairs": res.pairs_tried}, default=str, indent=1))


if __name__ == "__main__":
    main()
