"""Head-to-head comparison: dopplerdex vs hermes (dopplergeo) vs DOPPLERGEIST on the same real recordings.

Protocol "local" (hermes' published evaluation protocol): search a 15 km radius around a centre placed 5 km
from the truth at a bearing derived from the case id (so no grid is aligned on the answer); recording start
known to +-30 s. Every system gets identical inputs; truth is used only for scoring afterwards.

uv run python bench/compare/run_compare.py [--cases test_data/external/manifest.json] [--only id1,id2] [--systems ours,hermes,dg]
Results append to bench/compare/results.jsonl (one line per case x system); resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference"
OUT = ROOT / "bench/compare"
RUNS = OUT / "runs"
RESULTS = OUT / "results.jsonl"
HERMES_PY = REF / "_venv_hermes/bin/python"
HERMES_CLI = REF / "_venv_hermes/bin/dopplergeo"
DG_CLI = REF / "_venv_dg/bin/dopplergeist"
DG_CACHE = ROOT / "data/cache/dopplergeist"
HERMES_CACHE = ROOT / "data/cache/hermes"
RADIUS_KM, OFFSET_KM, TOL_S = 15.0, 5.0, 30.0
TOL_APPROX_S = 1800.0  # recordist gave only an approximate time ("around 7 pm"): same wider window for every system


def time_tolerance(case: dict) -> float:
    if case.get("time_tol_s"):
        return float(case["time_tol_s"])
    text = (str(case.get("time_source", "")) + " " + str(case.get("notes", ""))).lower()
    return TOL_APPROX_S if "approx" in text or "around" in text else TOL_S


# coarse country boxes for DOPPLERGEIST's required --country (W, S, E, N); only for cases we have
COUNTRIES = {"US": (-125, 24, -66, 50), "CA": (-141, 41.7, -52, 83), "GB": (-8.7, 49.8, 1.9, 60.9), "DE": (5.5, 47, 15.5, 55.2),
             "NL": (3.3, 50.7, 7.3, 53.6), "BE": (2.5, 49.5, 6.4, 51.5), "FR": (-5.5, 41.2, 10, 51.2), "RS": (18.8, 41.8, 23.1, 46.2),
             "CZ": (12.0, 48.5, 18.9, 51.1), "AT": (9.5, 46.3, 17.2, 49.1), "PS": (34.2, 31.2, 35.6, 32.6), "IL": (34.2, 29.4, 35.9, 33.4),
             "ES": (-9.4, 35.9, 3.4, 43.9), "IT": (6.6, 35.4, 18.6, 47.1), "IE": (-10.7, 51.4, -5.9, 55.4)}  # fmt: skip


def country_of(lat: float, lon: float) -> str | None:
    hits = [
        (k, (b[2] - b[0]) * (b[3] - b[1])) for k, b in COUNTRIES.items() if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]
    ]
    return min(hits, key=lambda h: h[1])[0] if hits else None


def offset_centre(case_id: str, lat: float, lon: float) -> tuple[float, float]:
    """Exactly hermes' eval_batch.region_center, so every system searches the identical area."""
    b = math.radians(int(hashlib.md5(case_id.encode()).hexdigest()[:6], 16) % 360)
    return (lat + OFFSET_KM * math.cos(b) / 111.2,
            lon + OFFSET_KM * math.sin(b) / (111.2 * math.cos(math.radians(lat))))  # fmt: skip


def dist_m(a_lat, a_lon, b_lat, b_lon) -> float:
    return float(np.hypot((a_lat - b_lat) * 111320, (a_lon - b_lon) * 111320 * np.cos(np.radians(b_lat))))


def utc_days(t_lo: float, t_hi: float) -> list[str]:
    days, t = set(), t_lo - 3600
    while t <= t_hi + 3600:
        days.add(datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d"))
        t += 3600
    return sorted(days)


def load_cases(paths: list[Path]) -> list[dict]:
    cases = []
    for p in paths:
        for c in json.loads(p.read_text()):
            audio = Path(c["audio"]) if c.get("audio") else (ROOT / "test_data/external" / c["file"] if c.get("file") else None)
            usable = bool(c.get("start_utc")) and c.get("lat") is not None and audio is not None and audio.exists()
            if usable:
                c["audio"] = str(audio)
                cases.append(c)
    if any("eclipse" in str(p) for p in paths):
        return cases  # dedicated set: no built-in cases
    # optional private case: the colsto challenge (answer kept out of the public repo; see docs/BENCHMARKS.md)
    priv = ROOT / "test_data/colsto_challenge/truth.local.json"
    if priv.exists():
        c = json.loads(priv.read_text())
        c["audio"] = str(ROOT / "test_data/colsto_challenge" / c["file"])
        if Path(c["audio"]).exists():
            cases.append(c)
    return cases


# ------------------------------------------------------------------------------- systems


def run_ours(case: dict, centre, t0: float, workdir: Path, tol: float = TOL_S) -> dict:
    from dopplerdex.audio.spectrogram import load_audio
    from dopplerdex.physics.atmosphere import speed_of_sound
    from dopplerdex.search.locate import locate
    from dopplerdex.web.data import prepare, status
    from dopplerdex.web.regions import search_region

    clat, clon = centre
    dlat, dlon = RADIUS_KM / 111.32, RADIUS_KM / (111.32 * math.cos(math.radians(clat)))
    region, info = search_region({"mode": "bbox", "bbox": [clat - dlat, clat + dlat, clon - dlon, clon + dlon]})
    x, fs = load_audio(case["audio"], 8000)
    for day in utc_days(t0 - tol, t0 + tol + len(x) / fs):
        if status(day, info["adsb_store"])["state"] != "ready":
            prepare(day, info["adsb_store"])
    res = locate(x, fs, t0 - tol, t0 + tol, float(speed_of_sound(15.0, 70.0)), region)
    if res.ranked.is_empty():
        return {"answered": False, "n_events": res.stats.get("n_events")}
    r = res.ranked
    lat = r["lat_ref"].fill_null(r["lat"]).to_numpy() if "lat_ref" in r.columns else r["lat"].to_numpy()
    lon = r["lon_ref"].fill_null(r["lon"]).to_numpy() if "lon_ref" in r.columns else r["lon"].to_numpy()
    return {"answered": True, "lat": float(lat[0]), "lon": float(lon[0]), "top": [[float(a), float(b)] for a, b in zip(lat[:20], lon[:20], strict=True)],
            "confidence": res.stats.get("loud_margin"), "n_events": res.stats.get("n_events"), "mode": res.stats.get("mode")}  # fmt: skip


def _raw_parts(day: str) -> list[str]:
    return [str(p) for p in sorted((ROOT / "data/raw/adsblol" / day).glob("*.tar.a?"))]


def run_hermes(case: dict, centre, t0: float, workdir: Path, tol: float = TOL_S) -> dict:
    from dopplerdex.web.data import prepare, raw_parts

    clat, clon = centre
    env = {"DOPPLER_GEO_CACHE": str(HERMES_CACHE), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", case["audio"]],
                               capture_output=True, text=True).stdout.strip() or 0)  # fmt: skip
    # pre-warm hermes' crop cache from the SAME local adsb.lol archives (its fetch_tracks accepts tar_parts)
    for day in utc_days(t0 - tol - 1800, t0 + tol + dur + 1800):
        if not raw_parts(day):
            prepare(day, "europe" if clon > -30 else "north_america")  # downloads the archive (ingest is harmless)
        code = ("import datetime as dt, sys\nfrom dopplergeo.adsb import fetch_tracks\n"
                f"d=dt.date.fromisoformat('{day}'); s=dt.datetime(d.year,d.month,d.day,tzinfo=dt.UTC).timestamp()\n"
                f"fetch_tracks(d,({clat - 0.6},{clon - 0.9},{clat + 0.6},{clon + 0.9}),s,s+86400,tar_parts={_raw_parts(day)!r})\n")  # fmt: skip
        subprocess.run([str(HERMES_PY), "-c", code], env=env, check=True, capture_output=True, timeout=7200)
    out = workdir / "hermes"
    iso = datetime.fromtimestamp(t0, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = [str(HERMES_CLI), "locate", case["audio"], "--date", iso, "--region", f"{clat:.6f},{clon:.6f},{RADIUS_KM}",
           "--time-tolerance", str(int(tol)), "--offline", "--broadband", "--no-plots", "--out", str(out)]  # fmt: skip
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    rep_f = out / "report.json"
    if not rep_f.exists():
        return {"answered": False, "error": (p.stderr or p.stdout)[-400:]}
    rep = json.loads(rep_f.read_text())
    b = rep.get("best")
    if not rep.get("ok") or not b:
        return {"answered": False, "error": rep.get("error"), "confidence": rep.get("confidence")}
    top = [[float(t["lat"]), float(t["lon"])] for t in (rep.get("top") or []) if "lat" in t][:20]
    return {"answered": True, "lat": float(b["lat"]), "lon": float(b["lon"]), "top": top or [[float(b["lat"]), float(b["lon"])]],
            "confidence": rep.get("confidence") or rep.get("flag")}  # fmt: skip


def run_dg(case: dict, centre, t0: float, workdir: Path, tol: float = TOL_S) -> dict:
    clat, clon = centre
    cc = country_of(case["lat"], case["lon"])
    if cc is None:
        return {"answered": False, "error": "no DOPPLERGEIST country profile mapped for this site"}
    dlat, dlon = RADIUS_KM / 111.32, RADIUS_KM / (111.32 * math.cos(math.radians(clat)))
    cfg = workdir / "dg_config.json"
    DG_CACHE.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"cache_root": str(DG_CACHE), "offline": False}))
    jobs = workdir / "dg_jobs"
    s = datetime.fromtimestamp(t0 - tol, UTC).isoformat()
    e = datetime.fromtimestamp(t0 + tol, UTC).isoformat()
    cmd = [str(DG_CLI), "analyze", case["audio"], "--start-utc", s, "--end-utc", e, "--time-source", "recordist metadata",
           "--country", cc, "--bounds", f"{clon - dlon}", f"{clat - dlat}", f"{clon + dlon}", f"{clat + dlat}",
           "--jobs", str(jobs), "--config", str(cfg), "--online", "--max-runtime", "5400", "--max-upload-mb", "900"]  # fmt: skip
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=9000)
    results = sorted(jobs.rglob("result.json"), key=lambda f: f.stat().st_mtime) if jobs.exists() else []
    if not results:
        return {"answered": False, "error": (p.stderr or p.stdout)[-400:]}
    res = json.loads(results[-1].read_text())
    cands = _find_candidates(res)
    if not cands:
        return {"answered": False, "error": str(res.get("status") or res.get("outcome") or "no candidates")[:300]}
    return {
        "answered": True,
        "lat": cands[0][0],
        "lon": cands[0][1],
        "top": cands[:20],
        "confidence": res.get("confidence"),
    }


def _find_candidates(obj) -> list[list[float]]:
    """Ranked (lat, lon) candidates anywhere in DOPPLERGEIST's result document (first ranked list found)."""
    if isinstance(obj, dict):
        for k in ("candidates", "ranked_candidates", "locations", "ranking"):
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict) and "lat" in v[0]:
                return [[float(c["lat"]), float(c["lon"])] for c in v]
        for v in obj.values():
            r = _find_candidates(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_candidates(v)
            if r:
                return r
    return []


SYSTEMS = {"ours": run_ours, "hermes": run_hermes, "dg": run_dg}


def done_keys() -> set[tuple[str, str]]:
    if not RESULTS.exists():
        return set()
    return {
        (json.loads(line)["case"], json.loads(line)["system"])
        for line in RESULTS.read_text().splitlines()
        if line.strip()
    }


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "test_data/external/manifest.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--systems", default="ours,hermes,dg")
    args = ap.parse_args()
    manifests = [Path(args.cases)] if Path(args.cases).exists() else []
    cases = load_cases(manifests)
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    RUNS.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    print(f"{len(cases)} cases with known location + start time; free disk {free_gb():.0f} GB", flush=True)
    for case in cases:
        t0 = datetime.fromisoformat(case["start_utc"].replace("Z", "+00:00")).timestamp()
        centre = offset_centre(case["id"], case["lat"], case["lon"])
        for sysname in args.systems.split(","):
            if (case["id"], sysname) in done:
                continue
            if free_gb() < 25:
                print("low disk space (<25 GB): stopping", flush=True)
                return
            workdir = RUNS / case["id"]
            workdir.mkdir(parents=True, exist_ok=True)
            t = time.time()
            try:
                r = SYSTEMS[sysname](case, centre, t0, workdir, time_tolerance(case))
            except Exception as exc:  # recorded, not fatal
                r = {"answered": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
            r.update(case=case["id"], system=sysname, seconds=round(time.time() - t, 1), truth=[case["lat"], case["lon"]],
                     truth_precision_m=case.get("loc_precision_m"), centre=list(centre), time_tol_s=time_tolerance(case))  # fmt: skip
            if r.get("answered"):
                r["error_m"] = round(dist_m(r["lat"], r["lon"], case["lat"], case["lon"]))
                hits = [i for i, (a, b) in enumerate(r.get("top", [])) if dist_m(a, b, case["lat"], case["lon"]) < 1500]
                r["rank_within_1500m"] = hits[0] + 1 if hits else None
            with RESULTS.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"{case['id']:>22s} {sysname:7s} answered={r.get('answered')} error_m={r.get('error_m')} "
                  f"({r['seconds']:.0f} s) {r.get('error', '')[:120] if not r.get('answered') else ''}", flush=True)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
