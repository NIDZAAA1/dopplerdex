from pathlib import Path

import numpy as np
import pytest

from dopplerdex.adsb.aircraft_types import _table, classify_description, classify_type
from dopplerdex.adsb.readsb import KT, parse_trace

SAMPLES = Path(__file__).resolve().parents[2] / "materials/data_samples/adsblol_2026-11-05/traces"


@pytest.mark.skipif(not _table(), reason="aircraft type table unavailable (offline)")
def test_classify():
    assert classify_type("C172") == "piston"
    assert classify_type("BE20") == "turboprop"
    assert classify_type("A320") == "jet"
    assert classify_type("R44") == "helicopter"
    assert classify_type("ZZZZ") == "unknown"
    assert classify_description("H1T") == "helicopter"


def test_parse_synthetic_trace():
    import orjson

    raw = orjson.dumps(
        {
            "icao": "a1b2c3",
            "t": "C172",
            "timestamp": 1.0e9,
            "trace": [
                [10.0, 40.0, -73.0, "ground", 0.0, None, 2, None, None, "adsb_icao", None, None, None, None],
                [
                    20.0,
                    40.01,
                    -73.0,
                    1000,
                    100.0,
                    90.0,
                    0,
                    500,
                    {"wd": 270, "ws": 10, "oat": 15},
                    "adsb_icao",
                    1100,
                    512,
                    95,
                    0.0,
                ],
                [30.0, 40.02, -73.0, 1200, 101.0, 91.0, 0, 0, None, "mlat", None, None, None, None],
            ],
        }
    )
    hdr, c = parse_trace(raw)
    assert hdr["icao"] == "a1b2c3" and hdr["t"] == "C172"
    np.testing.assert_allclose(c["t"], [1e9 + 10, 1e9 + 20, 1e9 + 30])
    assert c["on_ground"].tolist() == [True, False, False]
    assert np.isnan(c["alt_baro_m"][0]) and c["alt_baro_m"][1] == pytest.approx(304.8)
    assert c["gs_ms"][1] == pytest.approx(100 * KT)
    assert c["wind_ms"][1] == pytest.approx(10 * KT) and c["oat_c"][1] == 15
    assert c["leg"].tolist() == [1, 1, 1]
    assert c["src"].tolist() == [0, 0, 5]


@pytest.mark.skipif(not SAMPLES.exists(), reason="sample traces not downloaded")
def test_parse_real_samples():
    files = sorted(SAMPLES.glob("*/trace_full_*.json"))[:50]
    assert files
    for f in files:
        out = parse_trace(f.read_bytes())
        assert out is not None
        _, c = out
        assert np.all(np.diff(c["t"]) >= 0)
        assert np.all(np.abs(c["lat"]) <= 90) and np.all(np.abs(c["lon"]) <= 180)
