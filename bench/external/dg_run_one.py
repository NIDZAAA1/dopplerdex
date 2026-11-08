"""Adapter: our detector behind DOPPLERGEIST's label-free benchmark runner (discovery only).

Invoked by reference/DOPPLERGEIST/scripts/run_benchmark.py as
    uv run python bench/external/dg_run_one.py {request} {output}
Writes a DOPPLERGEIST JobResult (status inconclusive, events only) to {output}/result.json.
Road traffic (tier G) is not reported: it is not an aircraft event and is never used by our search.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dopplerdex.audio.events import DetectConfig, detect_events
from dopplerdex.audio.spectrogram import load_audio


def main() -> None:
    req_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    req = json.loads(req_path.read_text())
    audio = Path(req["audio_path"])
    if not audio.is_absolute():
        audio = req_path.parent / audio
    x, fs = load_audio(str(audio), 8000)
    events, _ = detect_events(x, fs, DetectConfig())
    out_events = []
    for k, e in enumerate(ev for ev in events if ev.tier != "G"):
        ridges = [e.ridge, *e.companions]
        t0 = float(min(r.t[0] for r in ridges))
        t1 = float(max(r.t[-1] for r in ridges))
        r = e.ridge
        t, idx = np.unique(r.t, return_index=True)
        f = r.f[idx]
        out_events.append({
            "event_id": f"e{k}", "start_s": t0, "end_s": t1,
            "band_hz": [float(min(rr.f.min() for rr in ridges)) * 0.99, float(max(rr.f.max() for rr in ridges)) * 1.01],
            "cpa_receive_s": max(0.0, float(e.fit.t_cpa_heard)),
            "ridge": [{"time_s": float(a), "frequency_hz": float(b)} for a, b in zip(t, f, strict=True)],
            "quality": {"tier_a": float(e.tier == "A"), "evidence": float(e.evidence)},
        })  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "schema_version": 1, "job_id": out.name, "status": "inconclusive", "stage": "discovery", "events": out_events,
        "reason_code": "discovery_only", "message": "dopplerdex detector only; no flight matching for this control.",
    }, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
