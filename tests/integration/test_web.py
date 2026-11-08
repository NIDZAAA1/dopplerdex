"""Web API: upload -> audio check (job) -> verdict; region resolution. (Search needs flight data; not run here.)"""

import io
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from dopplerdex.sim.scenario import Scenario, simulate, straight_pass
from dopplerdex.web.app import app
from dopplerdex.web.regions import adsb_store_for, search_region


def _wav_bytes() -> bytes:
    scn = Scenario([straight_pass("C172", 60.0, 1200.0, 600.0, 55.0, 70.0), straight_pass("BE20", 240.0, 2500.0, 1200.0, 95.0, 200.0)],
                   duration=320.0, seed=2)  # fmt: skip
    res = simulate(scn)
    buf = io.BytesIO()
    sf.write(buf, res.audio.astype(np.float32), int(scn.fs), format="WAV")
    return buf.getvalue()


def test_region_resolution():
    assert adsb_store_for((41.8, 46.3, 18.7, 23.1)) == "europe"
    assert adsb_store_for((30.0, 40.0, -100.0, -90.0)) == "north_america"
    assert adsb_store_for((-40.0, -30.0, 140.0, 150.0)) == "world"
    reg, info = search_region({"mode": "continent", "continent": "Europe"})
    assert info["adsb_store"] == "europe" and reg.lat_min == 35.0
    _, info2 = search_region({"mode": "place", "bbox": [44.7, 44.9, 20.3, 20.6], "label": "Belgrade", "pad_km": 30})
    assert info2["bbox"][0] < 44.7 and info2["area_km2"] > 1000


def test_upload_and_audio_check():
    client = TestClient(app)
    r = client.post("/api/upload", files={"file": ("flyover.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    up = r.json()["upload_id"]
    meta = '{"where": {"mode": "continent", "continent": "Continental US"}, "when": {"date": "2026-09-08", "time": "20:15", "tz": "UTC", "uncertainty_min": 30}}'
    job = client.post("/api/analyze", data={"upload_id": up, "meta": meta}).json()
    for _ in range(300):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert res["verdict"]["n_air"] >= 2 and res["verdict"]["searchable"]
    assert res["spectrogram_jpg"] and res["duration_s"] > 300
    assert "estimate" in res and res["estimate"]["high"] > res["estimate"]["low"] > 0
    assert res["data"]["region"]["adsb_store"] == "north_america"


def test_bad_upload_id_rejected():
    client = TestClient(app)
    assert client.post("/api/analyze", data={"upload_id": "../../etc", "meta": "{}"}).status_code == 400


def test_time_zone_offsets_and_window():
    from datetime import UTC, datetime

    from dopplerdex.web.search import parse_tz, time_window

    assert parse_tz("UTC+01:00").utcoffset(None).total_seconds() == 3600
    assert parse_tz("UTC-03:30").utcoffset(None).total_seconds() == -12600
    assert parse_tz("UTC").utcoffset(None).total_seconds() == 0
    lo, hi, days = time_window({"date": "2026-11-07", "time": "17:05", "tz": "UTC+01:00", "uncertainty_min": 15}, 1800)
    assert datetime.fromtimestamp(lo, UTC).strftime("%H:%M") == "15:50"
    assert datetime.fromtimestamp(hi, UTC).strftime("%H:%M") == "16:20"
    assert days == ["2026-11-07"]
