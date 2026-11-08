"""Ingest an adsb.lol / ADSBx globe_history day archive into region-filtered, hourly Parquet.

Output layout (Hive-style, one directory per day):
  data/processed/adsb/date=YYYY-MM-DD/hour=HH/part-*.parquet   points, sorted by (icao, t)
  data/processed/adsb/date=YYYY-MM-DD/aircraft.parquet         one row per aircraft (type, class, bbox)

The archive is streamed (split .tar.aa/.tar.ab parts are concatenated on the fly), traces are parsed
in a process pool, and only aircraft with at least one point inside the region box are kept.
"""

from __future__ import annotations

import io
import tarfile
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dopplerdex.adsb.aircraft_types import classify_type
from dopplerdex.adsb.readsb import parse_trace


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


REGIONS = {
    "conus": Region("conus", 23.0, 51.0, -127.0, -64.0),
    "north_america": Region("north_america", 15.0, 60.0, -135.0, -55.0),
    "europe": Region("europe", 34.0, 72.0, -12.0, 35.0),
    "world": Region("world", -90.0, 90.0, -180.0, 180.0),
}


class _MultiFile(io.RawIOBase):
    """Read several files back-to-back as one stream (split tar parts)."""

    def __init__(self, paths: list[Path]):
        self._files = [open(p, "rb") for p in paths]  # noqa: SIM115
        self._i = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while self._i < len(self._files):
            n = self._files[self._i].readinto(b)
            if n:
                return n
            self._files[self._i].close()
            self._i += 1
        return 0


def iter_trace_members(parts: list[Path]) -> Iterator[bytes]:
    stream = io.BufferedReader(_MultiFile(parts), buffer_size=1 << 22)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if m.isfile() and "/traces/" in f"/{m.name.lstrip('./')}" and "trace_full" in m.name:
                f = tf.extractfile(m)
                if f is not None:
                    yield f.read()


def _process(args: tuple[bytes, Region]) -> tuple[dict, dict] | None:
    raw, reg = args
    out = parse_trace(raw)
    if out is None:
        return None
    hdr, c = out
    inside = (
        (c["lat"] >= reg.lat_min) & (c["lat"] <= reg.lat_max) & (c["lon"] >= reg.lon_min) & (c["lon"] <= reg.lon_max)
    )
    if not inside.any():
        return None
    # keep points inside a slightly padded box (flights crossing the border keep their approach)
    pad = 1.0
    keep = (
        (c["lat"] >= reg.lat_min - pad) & (c["lat"] <= reg.lat_max + pad)
        & (c["lon"] >= reg.lon_min - pad) & (c["lon"] <= reg.lon_max + pad)
    )  # fmt: skip
    return hdr, {k: v[keep] for k, v in c.items()}


def ingest_day(parts: list[Path], out_dir: Path, region: Region, workers: int = 8, batch: int = 4000) -> dict:
    """Atomic: everything is written to `<out_dir>.partial` and renamed to out_dir only on success, with a
    `_ingest_stats.json` completion marker, so an interrupted run can never look like usable data."""
    import json
    import shutil

    final_dir = out_dir
    out_dir = final_dir.with_name(final_dir.name + ".partial")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    writers: dict[int, pq.ParquetWriter] = {}
    hour_buf: dict[int, list[pa.Table]] = {}
    aircraft_rows: list[dict] = []
    n_files = n_kept = n_points = 0

    def flush(final: bool = False) -> None:
        for h, tabs in list(hour_buf.items()):
            if not tabs:
                continue
            tab = pa.concat_tables(tabs)
            if h not in writers:
                d = out_dir / f"hour={h:02d}"
                d.mkdir(exist_ok=True)
                writers[h] = pq.ParquetWriter(d / "part-0.parquet", tab.schema, compression="zstd")
            writers[h].write_table(tab, row_group_size=200_000)
            hour_buf[h] = []

    def consume(res) -> None:
        nonlocal n_kept, n_points
        if res is None:
            return
        hdr, c = res
        n = c["t"].size
        if n == 0:
            return
        n_kept += 1
        n_points += n
        icao = str(hdr.get("icao") or "")
        type_code = hdr.get("t") or ""
        cls = classify_type(type_code)
        aircraft_rows.append({
            "icao": icao, "reg": hdr.get("r") or "", "type": type_code, "desc": hdr.get("desc") or "",
            "class": cls, "n_points": n, "t_min": float(c["t"].min()), "t_max": float(c["t"].max()),
            "lat_min": float(np.nanmin(c["lat"])), "lat_max": float(np.nanmax(c["lat"])),
            "lon_min": float(np.nanmin(c["lon"])), "lon_max": float(np.nanmax(c["lon"])),
        })  # fmt: skip
        hours = ((c["t"] % 86400) // 3600).astype(int)
        cols = {"icao": np.full(n, icao), "type": np.full(n, type_code), "class": np.full(n, cls), **c}
        tab = pa.table(cols)
        for h in np.unique(hours):
            hour_buf.setdefault(int(h), []).append(tab.filter(pa.array(hours == h)))

    with ProcessPoolExecutor(workers) as ex:
        pending = []
        for raw in iter_trace_members(parts):
            n_files += 1
            pending.append(raw)
            if len(pending) >= batch:
                for res in ex.map(_process, [(r, region) for r in pending], chunksize=64):
                    consume(res)
                pending = []
                flush()
                print(f"  {n_files:6d} traces read, {n_kept:6d} kept, {n_points / 1e6:6.1f} M points, "
                      f"{time.time() - t_start:5.0f} s", flush=True)  # fmt: skip
        for res in ex.map(_process, [(r, region) for r in pending], chunksize=64):
            consume(res)
    flush(final=True)
    for w in writers.values():
        w.close()
    pq.write_table(pa.Table.from_pylist(aircraft_rows), out_dir / "aircraft.parquet")
    stats = {"traces": n_files, "aircraft_kept": n_kept, "points": n_points, "seconds": time.time() - t_start,
             "region": region.name}  # fmt: skip
    (out_dir / "_ingest_stats.json").write_text(json.dumps(stats, indent=1))
    shutil.rmtree(final_dir, ignore_errors=True)
    out_dir.rename(final_dir)
    return stats
