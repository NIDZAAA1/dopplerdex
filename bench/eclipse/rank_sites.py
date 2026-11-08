"""Rank Eclipse Soundscapes site-hours on 2024-04-08 by close ADS-B passes (selection only, before any audio).

uv run python bench/eclipse/rank_sites.py -> data/raw/eclipse/site_hours.json
A pass counts when the aircraft comes within 3 km horizontally and below 1,500 m above the site, i.e. the
geometry our detectors can use. Truth coordinates are used here only to choose WHICH audio to fetch.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from dopplerdex.adsb.store import load_points
from dopplerdex.env.dem import fine_elevation

ROOT = Path(__file__).resolve().parents[2]
DAY0 = 1712534400.0  # 2024-04-08T00:00:00Z


def main() -> None:
    sites = json.loads((ROOT / "data/raw/eclipse/sites.json").read_text())
    rows = []
    for s in sites:
        if not (24 <= s["lat"] <= 50 and -125 <= s["lon"] <= -66) or not s.get("zip"):
            continue
        df = load_points(DAY0, DAY0 + 86400, s["lat"], s["lon"], 5.0, "north_america")
        if df.is_empty():
            continue
        g = float(fine_elevation(s["lat"], s["lon"])[0])
        d = np.hypot(
            (df["lat"].to_numpy() - s["lat"]) * 111320,
            (df["lon"].to_numpy() - s["lon"]) * 111320 * np.cos(np.radians(s["lat"])),
        )
        alt = (
            np.where(np.isfinite(df["alt_geom_m"].to_numpy()), df["alt_geom_m"].to_numpy(), df["alt_baro_m"].to_numpy())
            - g
        )
        df = df.with_columns(pl.Series("d", d), pl.Series("agl", alt)).filter(
            (pl.col("d") < 3000) & (pl.col("agl") < 1500)
        )
        if df.is_empty():
            continue
        passes = df.group_by("icao").agg(
            pl.col("t").sort_by("d").first().alias("t_cpa"), pl.first("class"), pl.min("d").alias("dmin")
        )
        passes = passes.with_columns(((pl.col("t_cpa") - DAY0) // 3600).cast(pl.Int32).alias("hour"))
        for (hour,), g_ in passes.group_by("hour"):
            rows.append({"esid": s["esid"], "lat": s["lat"], "lon": s["lon"], "hour_utc": int(hour), "n_close": len(g_),
                         "n_prop": int(g_.filter(pl.col("class").is_in(["piston", "turboprop", "helicopter"])).height),
                         "n_jet": int(g_.filter(pl.col("class") == "jet").height)})  # fmt: skip
    rows.sort(key=lambda r: -(r["n_close"] + r["n_prop"]))
    (ROOT / "data/raw/eclipse/site_hours.json").write_text(json.dumps(rows, indent=1))
    print(f"{len({r['esid'] for r in rows})} sites with close passes, {len(rows)} site-hours")
    for r in rows[:25]:
        print(r)


if __name__ == "__main__":
    main()
