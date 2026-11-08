"""Single-aircraft verification: crop the colsto challenge to stretches with one aircraft curve and run
our system under the comparison protocol (15 km radius around a 5 km-offset centre, start +-tol).

uv run python bench/single/crop_test.py [--tol 30]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/compare"))
from run_compare import offset_centre, run_ours  # noqa: E402

# The challenge answer is private: test_data/colsto_challenge/truth.local.json (gitignored) holds
# {"file": ..., "lat": ..., "lon": ..., "start_utc": ...}; see docs/BENCHMARKS.md.
_T = json.loads((ROOT / "test_data/colsto_challenge/truth.local.json").read_text())
SRC = ROOT / "test_data/colsto_challenge" / _T["file"]
T_FILE0 = datetime.fromisoformat(_T["start_utc"].replace("Z", "+00:00")).timestamp()
LAT, LON = _T["lat"], _T["lon"]
# (name, crop start s, crop length s): each holds one of the two known prop passes (PC12 at 2153.6 s, B350 at 3076.9 s)
CROPS = [("pc12_only", 1700, 900), ("b350_only", 2700, 900), ("pc12_short", 1950, 420), ("b350_short", 2880, 420)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=30.0)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    work = ROOT / "data/cache/single_crops"
    work.mkdir(parents=True, exist_ok=True)
    out = ROOT / "bench/single/results.jsonl"
    for name, s0, n in CROPS:
        if args.only and name != args.only:
            continue
        wav = work / f"{name}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s0), "-t", str(n), "-i", str(SRC), "-ac", "1", "-ar", "8000", str(wav)], check=True)
        case = {"id": f"colsto_{name}", "audio": str(wav), "lat": LAT, "lon": LON}
        centre = offset_centre(case["id"], LAT, LON)
        r = run_ours(case, centre, T_FILE0 + s0, work, tol=args.tol)
        if r.get("answered"):
            d = [math.hypot((a - LAT) * 111320, (b - LON) * 111320 * math.cos(math.radians(LAT))) for a, b in r["top"]]
            r["error_m"] = round(d[0])
            r["rank_within_1500m"] = next((i + 1 for i, x in enumerate(d) if x < 1500), None)
        r.update({"case": case["id"], "tol_s": args.tol})
        print(json.dumps({k: v for k, v in r.items() if k != "top"}), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
