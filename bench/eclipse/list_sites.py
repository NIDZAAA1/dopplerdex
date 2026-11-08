"""List Eclipse Soundscapes 2024-04-08 site records on Zenodo (CC BY 4.0) with their published coordinates.

python3 bench/eclipse/list_sites.py  ->  data/raw/eclipse/sites.json
"""

import csv
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data/raw/eclipse"


def get(url: str, tries: int = 4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs, page = {}, 1
    while True:
        q = urllib.parse.urlencode({"q": '"Total Solar Eclipse ESID"', "size": 25, "page": page})
        raw = get("https://zenodo.org/api/communities/eclipsesoundscapes/records?" + q)
        if raw is None:
            break
        d = json.loads(raw)
        hits = d["hits"]["hits"]
        for h in hits:
            t = h["metadata"]["title"]
            if "ESID" in t and "2024-04-08" in t:
                recs[h["id"]] = {"id": h["id"], "title": t, "license": (h["metadata"].get("license") or {}).get("id"),
                                 "files": {f["key"]: f["links"]["self"] for f in h.get("files", [])}}  # fmt: skip
        print(f"page {page}: {len(hits)} hits, {len(recs)} records (total {d['hits']['total']})", flush=True)
        if len(hits) < 25:
            break
        page += 1
        time.sleep(0.5)
    sites = []
    for r in recs.values():
        url = r["files"].get("total_eclipse_data.csv")
        raw = get(url) if url else None
        if not raw:
            continue
        row = next(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))), None)
        if not row:
            continue
        try:
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
        except (KeyError, ValueError):
            continue
        day = [k for k in r["files"] if k.endswith("_2024_04_08.zip")]
        sites.append({"esid": row["ESID"], "record": r["id"], "lat": lat, "lon": lon, "license": r["license"],
                      "clock": row.get("WAV Files Time & Date Settings"), "zip": r["files"].get(day[0]) if day else None,
                      "file_list": r["files"].get("file_list.csv")})  # fmt: skip
    (OUT / "sites.json").write_text(json.dumps(sites, indent=1))
    print(f"{len(sites)} sites with coordinates", flush=True)


if __name__ == "__main__":
    main()
