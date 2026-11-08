"""ADS-B data availability and preparation (download adsb.lol day archives + ingest a region)."""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dopplerdex.adsb.ingest import REGIONS, ingest_day

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/adsblol"
STORE = ROOT / "data/processed/adsb"
API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/v{dot}-planes-readsb-prod-0"
UA = {"User-Agent": "dopplerdex/0.1 (local research tool)"}


def ingested() -> list[dict]:
    out = []
    for reg in sorted(STORE.glob("region=*")):
        for day in sorted(reg.glob("date=*")):
            st = day / "_ingest_stats.json"
            if day.name.endswith(".partial") or not st.exists():
                continue  # incomplete or interrupted preparation
            info = json.loads(st.read_text())
            out.append({"region": reg.name.split("=", 1)[1], "date": day.name.split("=", 1)[1],
                        "aircraft": info.get("aircraft_kept"), "points": info.get("points")})  # fmt: skip
    return out


def raw_parts(date: str) -> list[Path]:
    return sorted((RAW / date).glob("*.tar.a?"))


def remote_release(date: str) -> dict | None:
    """Release metadata (asset urls + sizes) or None if not (yet) published."""
    d = datetime.fromisoformat(date)
    url = API.format(year=d.year, dot=d.strftime("%Y.%m.%d"))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
            rel = json.loads(r.read())
    except Exception:
        return None
    assets = [{"name": a["name"], "url": a["browser_download_url"], "size": a["size"]} for a in rel.get("assets", [])
              if ".tar.a" in a["name"]]  # fmt: skip
    return (
        {"assets": assets, "size": sum(a["size"] for a in assets), "published": rel.get("published_at")}
        if assets
        else None
    )


def status(date: str, store: str) -> dict:
    if (STORE / f"region={store}" / f"date={date}" / "_ingest_stats.json").exists():
        return {"state": "ready", "text": f"Flight data for {date} ({store}) is prepared."}
    if raw_parts(date):
        return {
            "state": "raw",
            "text": f"Archive for {date} is downloaded; it needs preparing for '{store}' (~1-3 min).",
        }
    rel = remote_release(date)
    if rel:
        return {"state": "remote", "size": rel["size"],
                "text": f"Archive for {date} is available to download ({rel['size'] / 1e9:.1f} GB, ~2-6 min), then prepare (~1-3 min)."}  # fmt: skip
    nxt = (datetime.fromisoformat(date) + timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "state": "missing",
        "text": f"No archive for {date} yet. adsb.lol publishes each day around 03:30 UTC on {nxt}.",
    }


def prepare(date: str, store: str, progress=None) -> dict:
    """Download (if needed) and ingest one day for one store region."""

    def rep(f: float, m: str) -> None:
        if progress:
            progress(f, m)

    parts = raw_parts(date)
    if not parts:
        rel = remote_release(date)
        if not rel:
            raise RuntimeError(f"no adsb.lol archive published for {date} yet")
        (RAW / date).mkdir(parents=True, exist_ok=True)
        total, done = rel["size"], 0
        for a in rel["assets"]:
            dst = RAW / date / a["name"]
            tmp = dst.with_suffix(dst.suffix + ".part")
            with (
                urllib.request.urlopen(urllib.request.Request(a["url"], headers=UA), timeout=60) as r,
                open(tmp, "wb") as fh,
            ):
                while chunk := r.read(1 << 22):
                    fh.write(chunk)
                    done += len(chunk)
                    rep(0.7 * done / total, f"downloading {done / 1e9:.2f} / {total / 1e9:.2f} GB")
            tmp.rename(dst)
        parts = raw_parts(date)
    rep(0.7, f"preparing {store} (reading the day's traces)")
    t = time.time()
    out = STORE / f"region={store}" / f"date={date}"
    stats = ingest_day(parts, out, REGIONS[store], workers=6)  # atomic, writes its own completion marker
    rep(1.0, f"prepared {stats['aircraft_kept']:,} aircraft in {time.time() - t:.0f} s")
    return stats


def utc_day_span(date: str) -> tuple[float, float]:
    d = datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()
    return d, d + 86400
