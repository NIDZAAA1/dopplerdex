"""Turn 'where' hints (continent, country, city, custom box) into a search bounding box."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from dopplerdex.adsb.ingest import REGIONS as ADSB_STORES
from dopplerdex.search.pipeline import SearchRegion

CACHE = Path(__file__).resolve().parents[3] / "data/cache/geocode"

# (lat_min, lat_max, lon_min, lon_max)
CONTINENTS: dict[str, tuple[float, float, float, float]] = {
    "Continental US": (24.0, 50.0, -125.0, -66.0),
    "North America": (15.0, 60.0, -135.0, -55.0),
    "Europe": (35.0, 71.0, -11.0, 34.0),
    "South America": (-56.0, 13.0, -82.0, -34.0),
    "Africa": (-35.0, 37.5, -18.0, 52.0),
    "Asia": (-11.0, 56.0, 60.0, 150.0),
    "Oceania": (-48.0, 0.0, 110.0, 180.0),
}


def adsb_store_for(bbox: tuple[float, float, float, float]) -> str:
    """Smallest ingested-store region that fully contains the box (falls back to 'world')."""
    la0, la1, lo0, lo1 = bbox
    for name in ("conus", "north_america", "europe"):
        r = ADSB_STORES[name]
        if r.lat_min <= la0 and la1 <= r.lat_max and r.lon_min <= lo0 and lo1 <= r.lon_max:
            return "north_america" if name == "conus" else name
    return "world"


@lru_cache(maxsize=256)
def geocode(query: str) -> dict | None:
    """Bounding box of a place name via OpenStreetMap Nominatim (cached on disk). Sends only the query."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = "".join(ch if ch.isalnum() else "_" for ch in query.lower())[:80]
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()) or None
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "dopplerdex/0.1 (local research tool)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        hits = json.loads(r.read())
    out = None
    if hits:
        h = hits[0]
        s, n, w, e = (float(v) for v in h["boundingbox"])
        out = {"name": h.get("display_name", query), "bbox": [s, n, w, e], "lat": float(h["lat"]), "lon": float(h["lon"]),
               "type": h.get("type", "")}  # fmt: skip
    f.write_text(json.dumps(out))
    return out


def search_region(where: dict) -> tuple[SearchRegion, dict]:
    """where = {"mode": "unknown"|"continent"|"place"|"bbox", "continent": str, "bbox": [s, n, w, e], "pad_km": float}.

    'place' expects a bbox already resolved by geocode() on the client side (shown to the user first).
    Returns the region and a description for the UI.
    """
    mode = where.get("mode", "unknown")
    if mode == "continent":
        bbox = CONTINENTS[where["continent"]]
        label = where["continent"]
    elif mode in ("place", "bbox"):
        s, n, w, e = (float(v) for v in where["bbox"])
        pad = float(where.get("pad_km", 0.0)) / 111.0
        bbox = (s - pad, n + pad, w - pad * 1.4, e + pad * 1.4)
        label = where.get("label") or "custom area"
    else:
        bbox = CONTINENTS["Continental US"] if where.get("default") == "conus" else CONTINENTS["Europe"]
        label = "unknown (defaulting to " + ("Continental US" if where.get("default") == "conus" else "Europe") + ")"
    store = adsb_store_for(bbox)
    region = SearchRegion(bbox[0], bbox[1], bbox[2], bbox[3], store)
    coslat = max(0.2, abs(math.cos(math.radians(0.5 * (bbox[0] + bbox[1])))))
    area_km2 = (bbox[1] - bbox[0]) * 111.0 * (bbox[3] - bbox[2]) * 111.0 * coslat
    return region, {"label": label, "bbox": list(bbox), "adsb_store": store, "area_km2": area_km2}
