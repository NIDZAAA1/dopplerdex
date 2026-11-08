"""Manifest of fetched Eclipse Soundscapes clips for bench/compare/run_compare.py --cases and hermes' eval_batch."""

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data/raw/eclipse/clips"


def main() -> None:
    ours, hermes = [], []
    for j in sorted(CLIPS.glob("*.json")):
        m = json.loads(j.read_text())
        wav = j.with_suffix(".wav")
        if not wav.exists() or m.get("covered_s", m["duration_s"]) < 1800:
            continue
        cid = wav.stem
        ours.append({"id": cid, "audio": str(wav), "lat": m["lat"], "lon": m["lon"], "loc_precision_m": 50,
                     "start_utc": m["start_utc"], "time_source": "AudioMoth filename (time chime)", "time_tol_s": 90,
                     "source": f"Eclipse Soundscapes ESID#{m['esid']} (zenodo {m['record']}, {m['license']})"})  # fmt: skip
        hermes.append({"id": cid, "audio": str(wav), "t0_unix": datetime.fromisoformat(m["start_utc"].replace("Z", "+00:00")).timestamp(),
                       "truth_lat": m["lat"], "truth_lon": m["lon"], "truth_unc_m": 50, "split": "eclipse"})  # fmt: skip
    (ROOT / "bench/eclipse/manifest.json").write_text(json.dumps(ours, indent=1))
    (ROOT / "bench/eclipse/manifest_hermes.json").write_text(json.dumps(hermes, indent=1))
    print(len(ours), "clips")


if __name__ == "__main__":
    main()
