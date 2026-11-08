"""Detection comparison on the 40 real Livermore smartphone recordings (Zenodo 23424278, CC BY 4.0).

uv run python bench/external/livermore_detect.py
Both detectors run blind (no bands, windows or times) and emit DOPPLERGEIST JobResult events; both are
scored by the same rule afterwards. The recordings have no established sample-zero clock, so an event
counts as the annotated aircraft when its ridge frequencies overlap the annotated frequency band (from
frequency_vs_time_used.csv, +-3 %) for at least half of that band, in any harmonic ratio 1/3..3.
Other events are reported as unmatched (not necessarily false: other sources may be audible).
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/livermore_zenodo_23424278/x"
WORK = ROOT / "data/cache/livermore_runs"
DG = ROOT / "reference/DOPPLERGEIST"
DG_PY = ROOT / "reference/_venv_dg/bin/python"
RATIOS = (1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0)


def run(system: str, wav: Path, out: Path) -> tuple[list[dict], float]:
    out.mkdir(parents=True, exist_ok=True)
    req = out / "request.json"
    req.write_text(json.dumps({"schema_version": 1, "audio_path": str(wav)}))
    t = time.time()
    if system == "ours":
        cmd = [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "python",
            str(ROOT / "bench/external/dg_run_one.py"),
            str(req),
            str(out),
        ]
        env = None
    else:
        cmd = [str(DG_PY), str(DG / "scripts/discover_benchmark_case.py"), str(req), str(out)]
        env = {"PYTHONPATH": str(DG / "src"), "PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    dt = time.time() - t
    f = out / "result.json"
    if p.returncode != 0 or not f.exists():
        return [{"error": (p.stderr or p.stdout)[-300:]}], dt
    return json.loads(f.read_text()).get("events", []), dt


def overlaps(ev: dict, lo: float, hi: float) -> bool:
    f = np.array([pt["frequency_hz"] for pt in ev.get("ridge", [])])
    if f.size == 0:
        return False
    a, b = float(np.percentile(f, 5)), float(np.percentile(f, 95))
    for k in RATIOS:
        x0, x1 = max(a * k, lo * 0.97), min(b * k, hi * 1.03)
        if x1 - x0 >= 0.5 * (hi - lo):
            return True
    return False


def main() -> None:
    rows = []
    for wav in sorted(p for p in DATA.rglob("audio.wav") if "__MACOSX" not in p.parts):
        case = wav.parent.name
        with (wav.parent / "frequency_vs_time_used.csv").open() as fh:
            fr = [float(r["frequency_hz"]) for r in csv.DictReader(fh)]
        lo, hi = float(np.percentile(fr, 2)), float(np.percentile(fr, 98))
        row = {"case": case, "band": [round(lo, 1), round(hi, 1)]}
        for system in ("ours", "dg"):
            evs, dt = run(system, wav, WORK / system / case)
            if evs and "error" in evs[0]:
                row[system] = {"error": evs[0]["error"], "seconds": round(dt, 1)}
                continue
            hit = [overlaps(e, lo, hi) for e in evs]
            row[system] = {
                "detected": any(hit),
                "n_events": len(evs),
                "unmatched": hit.count(False),
                "seconds": round(dt, 1),
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "bench/external/livermore_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for system in ("ours", "dg"):
        ok = [r[system] for r in rows if "error" not in r[system]]
        print(system, f"detected {sum(r['detected'] for r in ok)}/{len(rows)}",
              f"events {sum(r['n_events'] for r in ok)} unmatched {sum(r['unmatched'] for r in ok)}",
              f"clips with any event {sum(r['n_events'] > 0 for r in ok)}",
              f"errors {len(rows) - len(ok)} median s {np.median([r['seconds'] for r in ok]):.1f}")  # fmt: skip


if __name__ == "__main__":
    main()
