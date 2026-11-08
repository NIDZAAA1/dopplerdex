"""End-to-end blind localisation benchmark on synthetic recordings rendered from REAL ADS-B traffic.

uv run python bench/locate_bench.py --n 6 --date 2026-09-08
Each case: random CONUS spot near real GA traffic + real time -> render 25 min of audio from all real
flights within 20 km -> blind pipeline (detect, pair search over the whole US, T0 window +-3 h) ->
rank and error of the true location.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from dopplerdex.audio.spectrogram import resample
from dopplerdex.eval.real_track_scenarios import build_real_scenario
from dopplerdex.search.locate import locate
from dopplerdex.search.tracks import hour_files
from dopplerdex.sim.scenario import simulate

ROOT = Path(__file__).resolve().parent


def pick_site(date: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """A random moment on a random GA/turboprop track over CONUS land; mic 0.5-4 km to the side."""
    day = datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()
    hour = int(rng.integers(13, 23))
    f = hour_files(day + hour * 3600, day + hour * 3600 + 1, "north_america")[0]
    df = (pl.scan_parquet(f).filter(pl.col("class").is_in(["piston", "turboprop"]) & ~pl.col("on_ground")
                                   & pl.col("lat").is_between(26, 48) & pl.col("lon").is_between(-122, -70)
                                   & (pl.col("alt_baro_m") < 3000)).select(["t", "lat", "lon", "track_deg"]).collect())  # fmt: skip
    row = df.row(int(rng.integers(0, len(df))), named=True)
    off = rng.uniform(500, 4000) * rng.choice([-1, 1])
    tr = np.radians(row["track_deg"] or 0.0)
    lat = row["lat"] + off * (-np.sin(tr)) / 111320
    lon = row["lon"] + off * np.cos(tr) / (111320 * np.cos(np.radians(row["lat"])))
    t0 = row["t"] - rng.uniform(300, 1200)  # the GA pass happens inside the recording
    return float(lat), float(lon), float(t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--date", default="2026-09-08")
    ap.add_argument("--duration", type=float, default=1500.0)
    ap.add_argument("--t0-window-h", type=float, default=3.0)
    args = ap.parse_args()
    rows = []
    for k in range(args.n):
        seed = args.seed0 + k
        rng = np.random.default_rng(seed)
        lat, lon, t0 = pick_site(args.date, rng)
        t_case = time.time()
        rs = build_real_scenario(lat, lon, t0, args.duration, seed)
        res = simulate(rs.scenario)
        x = resample(res.audio, int(rs.scenario.fs), 8000)
        c = rs.scenario.atmosphere.c
        w = args.t0_window_h * 3600
        out = locate(x, 8000, t0 - w, t0 + w, c)
        _ = sorted((p.label, round(float(np.min(np.linalg.norm(p.traj.pos[:, :2], axis=1)))))
                            for p in rs.scenario.aircraft if "jet" not in p.label.lower())[:8]  # fmt: skip
        print(f"case {seed}: {len(rs.scenario.aircraft)} flights rendered; events "
              f"{[(round(e.fit.t_cpa_heard), e.tier, round(e.fit.v), round(e.fit.r_cpa)) for e in out.events]}; "
              f"pairs {out.pairs_tried}", flush=True)  # fmt: skip
        row = {"seed": seed, "lat": lat, "lon": lon, "t0": t0, "n_flights": len(rs.scenario.aircraft),
               "n_events": out.stats["n_events"], "pairs": len(out.pairs_tried), "rank": None, "err_m": None,
               "best_err_m": None, "dt0_s": None, "seconds": time.time() - t_case}  # fmt: skip
        r = out.ranked
        if len(r):
            la = r["lat_ref"].fill_null(r["lat"]).to_numpy() if "lat_ref" in r.columns else r["lat"].to_numpy()
            lo = r["lon_ref"].fill_null(r["lon"]).to_numpy() if "lon_ref" in r.columns else r["lon"].to_numpy()
            d = np.hypot((la - lat) * 111320, (lo - lon) * 111320 * np.cos(np.radians(lat)))
            ok = np.flatnonzero((d < 1500) & (np.abs(r["t0_fit"].to_numpy() - t0) < 30))
            row["best_err_m"] = float(d[0])
            row["dt0_s"] = float(r["t0_fit"][0] - t0)
            row["loud_margin"] = out.stats.get("loud_margin")
            # rank of the truth under each ranking scheme, on the SAME candidate set (ablation)
            truth_ok = (np.hypot((r["lat"].to_numpy() - lat) * 111320, (r["lon"].to_numpy() - lon) * 111320
                                 * np.cos(np.radians(lat))) < 1500) & (np.abs(r["t0_fit"].to_numpy() - t0) < 30)  # fmt: skip
            schemes = {"ridge": r["score"].to_numpy(), "loud": -r["loud_corr"].to_numpy(),
                       "spectro": -r["spectro"].to_numpy(), "combined": -r["combined"].to_numpy()}  # fmt: skip
            for name, key in schemes.items():
                order = np.argsort(key, kind="stable")
                hit = np.flatnonzero(truth_ok[order])
                row[f"rank_{name}"] = int(hit[0]) + 1 if hit.size else None
            row["n_candidates"] = int(len(r))
            if ok.size:
                row["rank"] = int(ok[0]) + 1
                row["err_m"] = float(d[ok[0]])
        rows.append(row)
        print(json.dumps(row), flush=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (ROOT / "results" / f"locate_{stamp}.json").write_text(json.dumps(rows, indent=1))
    for name in ("ridge", "loud", "spectro", "combined"):
        rk = [r.get(f"rank_{name}") for r in rows]
        print(f"{name:9s} top-1 {sum(1 for x in rk if x == 1)}/{len(rows)}  top-5 {sum(1 for x in rk if x and x <= 5)}/{len(rows)}"
              f"  ranks {rk}")  # fmt: skip
    ranks = [r["rank"] for r in rows]
    print(f"top-1: {sum(1 for x in ranks if x == 1)}/{len(rows)}   top-10: {sum(1 for x in ranks if x and x <= 10)}/{len(rows)}"
          f"   found at all: {sum(1 for x in ranks if x)}/{len(rows)}")  # fmt: skip


if __name__ == "__main__":
    main()
