"""DOPPLERGEIST four-flyover synthetic fixture through our full pipeline (blind; truth read only to score).

uv run python bench/external/dg_four.py data/cache/dg_bench/four
Inputs used: inputs/recording.mp3, inputs/request.json (clock), inputs/flights.csv (the entire toy flight
universe, ingested as ADS-B store region "dg_synth"), and the fixture's documented operator terrain
(constant 122 m ground, zero geoid). Search area: CONUS, as the request's coverage states.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from dopplerdex.adsb.aircraft_types import classify_type
from dopplerdex.adsb.store import STORE
from dopplerdex.audio.spectrogram import load_audio
from dopplerdex.env.synthetic import use_flat_world
from dopplerdex.physics.atmosphere import speed_of_sound
from dopplerdex.search.locate import locate
from dopplerdex.search.pipeline import SearchRegion

REGION = "dg_synth"


def ingest(flights_csv: Path) -> None:
    df = pl.read_csv(flights_csv).sort(["flight_id", "time_s"])
    out = []
    for g in df.partition_by("flight_id"):
        t, lat, lon = (g[c].to_numpy() for c in ("time_s", "lat", "lon"))
        e = np.gradient(lon, t) * 111320.0 * np.cos(np.radians(lat))
        n = np.gradient(lat, t) * 111320.0
        out.append(g.with_columns(pl.Series("gs_ms", np.hypot(e, n)), pl.Series("track_deg", np.degrees(np.arctan2(e, n)) % 360),
                                  pl.Series("geom_rate_ms", np.gradient(g["alt_m"].to_numpy(), t))))  # fmt: skip
    df = pl.concat(out)
    n = len(df)
    df = df.select(
        pl.col("flight_id").alias("icao"), pl.col("typecode").alias("type"),
        pl.col("typecode").map_elements(classify_type, return_dtype=pl.Utf8).alias("class"),
        pl.col("time_s").alias("t"), "lat", "lon", pl.col("alt_m").alias("alt_baro_m"), pl.col("alt_m").alias("alt_geom_m"),
        pl.lit(False).alias("on_ground"), "gs_ms", "track_deg", pl.col("geom_rate_ms").alias("baro_rate_ms"), "geom_rate_ms",
        pl.lit(0, pl.Int16).alias("flags"), pl.lit(0, pl.Int8).alias("src"), pl.lit(0, pl.Int16).alias("leg"),
        pl.lit(None, pl.Float64).alias("wind_dir"), pl.lit(None, pl.Float64).alias("wind_ms"), pl.lit(None, pl.Float64).alias("oat_c"),
    )  # fmt: skip
    df = df.with_columns(pl.from_epoch(pl.col("t"), time_unit="s").alias("_dt"))
    for (day, hour), g in df.group_by(
        pl.col("_dt").dt.strftime("%Y-%m-%d").alias("_day"), pl.col("_dt").dt.hour().alias("_hour")
    ):
        d = STORE / f"region={REGION}" / f"date={day}" / f"hour={hour:02d}"
        d.mkdir(parents=True, exist_ok=True)
        g.drop("_dt").write_parquet(d / "part-0.parquet")
        ac = g.group_by("icao").agg(pl.first("type"), pl.first("class"), pl.len().alias("n_points"),
                                    pl.min("t").alias("t_min"), pl.max("t").alias("t_max"), pl.min("lat").alias("lat_min"),
                                    pl.max("lat").alias("lat_max"), pl.min("lon").alias("lon_min"), pl.max("lon").alias("lon_max"))  # fmt: skip
        ac.with_columns(pl.lit(None, pl.Utf8).alias("reg"), pl.lit(None, pl.Utf8).alias("desc")).write_parquet(
            d.parent / "aircraft.parquet"
        )
        (d.parent / "_ingest_stats.json").write_text(json.dumps({"points": n, "source": str(flights_csv)}))


def main() -> None:
    root = Path(sys.argv[1])
    req = json.loads((root / "inputs/request.json").read_text())
    ingest(root / "inputs/flights.csv")
    use_flat_world(122.0, 0.0)
    t_epoch = datetime.fromisoformat(req["time_window"]["start_utc"].replace("Z", "+00:00")).timestamp()
    x, fs = load_audio(str(root / "inputs" / req["audio_path"]), 8000)
    region = SearchRegion(24.0, 50.0, -125.0, -66.0, REGION)  # "coverage: conus"
    t = time.time()
    res = locate(x, fs, t_epoch - 2.0, t_epoch + 2.0, float(speed_of_sound(15.0, 70.0)), region, verbose=False)
    out = {"seconds": round(time.time() - t, 1), "n_events": len(res.events), "mode": res.stats.get("mode"),
           "tiers": [e.tier for e in res.events], "t_cpa": [round(e.fit.t_cpa_heard, 1) for e in res.events]}  # fmt: skip
    if not res.ranked.is_empty():
        r = res.ranked.row(0, named=True)
        lat = r["lat_ref"] if r.get("lat_ref") is not None else r["lat"]
        lon = r["lon_ref"] if r.get("lon_ref") is not None else r["lon"]
        out.update(lat=lat, lon=lon, flights=[r.get("icao"), r.get("icao_b")], margin=res.stats.get("loud_margin"),
                   t0_offset_s=round(r["t0_fit"] - t_epoch, 2))  # fmt: skip
        top = res.ranked.head(20)
        out["top"] = [[float(a if a is not None else b), float(c if c is not None else d)]
                      for a, b, c, d in zip(top["lat_ref"], top["lat"], top["lon_ref"], top["lon"], strict=True)]  # fmt: skip
    # ---- scoring (truth read only now)
    lab = json.loads((root / "truth/labels.json").read_text())
    lab = lab.get("case") or lab.get(next(iter(lab))) if "location" not in lab else lab
    tl = lab.get("location") or [lab.get("lat"), lab.get("lon")]
    if "lat" in out:
        d = [
            math.hypot((a - tl[0]) * 111320, (b - tl[1]) * 111320 * math.cos(math.radians(tl[0])))
            for a, b in out["top"]
        ]
        out["error_m"] = round(d[0], 1)
        out["rank_within_1000m"] = next((i + 1 for i, v in enumerate(d) if v < 1000), None)
    out["truth_flights"] = lab.get("flight_ids")
    out.pop("top", None)
    print(json.dumps(out, indent=1))
    (Path(__file__).parent / "dg_four_result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
