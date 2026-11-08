"""Extract one UTC hour of audio from an Eclipse Soundscapes site zip on Zenodo without downloading it all.

uv run python bench/eclipse/fetch_clip.py ESID HOUR_UTC   (2024-04-08)
Reads the zip's central directory with HTTP range requests, picks the WAV whose filename start time
(UTC, AudioMoth time-chime clock) precedes the hour, streams and inflates that member only up to the end
of the hour, and writes data/raw/eclipse/clips/es{ESID}_{HH}.wav (16 kHz mono) + a JSON sidecar.
"""

from __future__ import annotations

import json
import struct
import sys
import time
import urllib.request
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/eclipse/clips"
DAY0 = datetime(2024, 4, 8, tzinfo=UTC).timestamp()


def rng(url: str, a: int, b: int | None = None, tries: int = 6) -> bytes:
    h = {"Range": f"bytes={a}-" + ("" if b is None else str(b))}
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001  (429 rate limit, transient network errors)
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def members(url: str, size: int) -> list[dict]:
    tail = rng(url, size - 65536, size - 1)
    j = tail.rfind(b"PK\x06\x06")
    cd_size, cd_off = struct.unpack("<QQ", tail[j + 40 : j + 56])
    cd = rng(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while cd[p : p + 4] == b"PK\x01\x02":
        (method,) = struct.unpack("<H", cd[p + 10 : p + 12])
        csz, usz = struct.unpack("<II", cd[p + 20 : p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28 : p + 34])
        (off,) = struct.unpack("<I", cd[p + 42 : p + 46])
        name = cd[p + 46 : p + 46 + nl].decode()
        extra = cd[p + 46 + nl : p + 46 + nl + el]
        q = 0
        while q < len(extra):  # zip64 extended information
            hid, hsz = struct.unpack("<HH", extra[q : q + 4])
            if hid == 1:
                vals = list(struct.unpack(f"<{hsz // 8}Q", extra[q + 4 : q + 4 + hsz]))
                if usz == 0xFFFFFFFF:
                    usz = vals.pop(0)
                if csz == 0xFFFFFFFF:
                    csz = vals.pop(0)
                if off == 0xFFFFFFFF:
                    off = vals.pop(0)
            q += 4 + hsz
        out.append({"name": name, "method": method, "csize": csz, "usize": usz, "offset": off})
        p += 46 + nl + el + cl
    return out


def zip_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def extract(url: str, m: dict, a_s: float, b_s: float) -> tuple[np.ndarray, int]:
    """Samples [a_s, b_s) seconds into one deflated WAV member, inflating only up to b_s."""
    lh = rng(url, m["offset"], m["offset"] + 29)
    nl, el = struct.unpack("<HH", lh[26:30])
    data_off = m["offset"] + 30 + nl + el
    req = urllib.request.Request(url, headers={"Range": f"bytes={data_off}-{data_off + m['csize'] - 1}"})
    d = zlib.decompressobj(-15)
    buf, need, sr, ch, done = bytearray(), None, 0, 1, False
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                while True:
                    chunk = r.read(4 << 20)
                    if not chunk:
                        break
                    buf += d.decompress(chunk)
                    if need is None and len(buf) > 4096:
                        i = buf.find(b"fmt ")
                        ch, sr = struct.unpack("<HI", buf[i + 10 : i + 16])
                        hdr = buf.find(b"data") + 8
                        bps = ch * 2
                        need = [hdr + int(a_s * sr) * bps, hdr + int(b_s * sr) * bps]
                    if need is not None:
                        if len(buf) >= need[1]:
                            done = True
                            break
                        if need[0] > (64 << 20) and len(buf) > need[0]:
                            drop = need[0]
                            del buf[:drop]
                            need = [0, need[1] - drop]
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"  stream retry {attempt + 1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
            d, buf, need = zlib.decompressobj(-15), bytearray(), None
    if need is None:
        return np.zeros(0), 16000
    got = bytes(buf[need[0] : need[1]] if done else buf[need[0] :])
    x = np.frombuffer(got[: len(got) // (ch * 2) * ch * 2], dtype="<i2").reshape(-1, ch).mean(axis=1) / 32768.0
    return x, sr


def main() -> None:
    esid, hour = sys.argv[1], int(sys.argv[2])
    site = {s["esid"]: s for s in json.loads((ROOT / "data/raw/eclipse/sites.json").read_text())}[esid]
    url = site["zip"]
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"es{esid}_{hour:02d}.wav"
    if dst.exists():
        print("exists", dst)
        return
    ms = [m for m in members(url, zip_size(url)) if m["name"].upper().endswith(".WAV")]
    t_lo, t_hi = DAY0 + hour * 3600, DAY0 + (hour + 1) * 3600
    starts = sorted(((datetime.strptime(Path(m["name"]).stem, "%Y%m%d_%H%M%S").replace(tzinfo=UTC).timestamp(), m) for m in ms),
                    key=lambda x: x[0])  # fmt: skip
    use = [(t, m, starts[k + 1][0] if k + 1 < len(starts) else t + 86400) for k, (t, m) in enumerate(starts)]
    use = [(t, m, t_next) for t, m, t_next in use if t < t_hi and t_next > t_lo]
    if not use:
        raise SystemExit("no WAV covers the requested hour")
    sr_out = 16000
    y = np.zeros(int(3600 * sr_out), np.float32)
    covered = 0.0
    for t_file, m, _ in use:
        seg, sr = extract(url, m, max(0.0, t_lo - t_file), t_hi - t_file)
        if seg.size == 0:
            continue
        z = resample_poly(seg, sr_out, sr) if sr != sr_out else seg
        i0 = int(round((max(t_file, t_lo) - t_lo) * sr_out))
        n = min(z.size, y.size - i0)
        y[i0 : i0 + n] = z[:n]
        covered += n / sr_out
        print(f"  {m['name']} -> +{n / sr_out:.0f} s", flush=True)
    sf.write(dst, y, sr_out, subtype="PCM_16")
    m = use[0][1]
    meta = {"esid": esid, "lat": site["lat"], "lon": site["lon"], "start_utc": datetime.fromtimestamp(t_lo, UTC).isoformat().replace("+00:00", "Z"),
            "duration_s": len(y) / 16000, "covered_s": covered, "source_member": m["name"], "n_members": len(use), "record": site["record"], "license": site["license"],
            "clock": site["clock"]}  # fmt: skip
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {dst.name}: {covered / 60:.1f} of 60 min covered", flush=True)


if __name__ == "__main__":
    main()
