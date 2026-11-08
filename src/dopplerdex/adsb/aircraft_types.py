"""Aircraft class from the ICAO Doc 8643 type designator.

Doc 8643 description code, e.g. 'L1P' = Landplane, 1 engine, Piston; 'H1T' = Helicopter, turboshaft;
'L2J' = jet. We map to acoustic classes: piston | turboprop | helicopter | jet | other | unknown.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CACHE = Path(__file__).resolve().parents[3] / "data/cache/aircraft_types2.json"
# ICAO Doc 8643 designator -> [name, description code, wake category], as published (gzip) by the
# tar1090-db project. Downloaded on first use and cached; not redistributed with this repository.
TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js"


@lru_cache(maxsize=1)
def _table() -> dict[str, list]:
    if not CACHE.exists():
        try:
            import gzip
            import urllib.request

            req = urllib.request.Request(TYPES_URL, headers={"User-Agent": "dopplerdex (aircraft type table)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(data))
        except Exception:  # offline: every type classifies as "unknown"
            return {}
    return json.loads(CACHE.read_text())


def classify_description(desc: str | None) -> str:
    if not desc or len(desc) < 3:
        return "unknown"
    kind, eng = desc[0], desc[2]
    if kind in ("H", "G"):  # helicopter, gyrocopter
        return "helicopter"
    if eng == "J":
        return "jet"
    if eng == "P":
        return "piston"
    if eng == "T":
        return "turboprop"
    if eng == "E":
        return "other"  # electric
    return "unknown"


def classify_type(type_code: str | None) -> str:
    if not type_code:
        return "unknown"
    row = _table().get(type_code.upper())
    return classify_description(row[1]) if row else "unknown"
