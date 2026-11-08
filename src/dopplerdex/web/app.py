"""Local web app: upload -> audio check -> search -> map results, plus Learn and Data pages.

Run:  uv run dopplerdex-web   (or: uv run python -m dopplerdex.web.app)   -> http://127.0.0.1:8765
Binds to localhost only: recordings and results never leave this machine (except: a place-name lookup
sends just the typed name to OpenStreetMap Nominatim; flight data and weather are downloaded).
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dopplerdex.web import data as data_mod
from dopplerdex.web.analysis import analyse, estimate_seconds
from dopplerdex.web.jobs import JobManager
from dopplerdex.web.regions import CONTINENTS, geocode, search_region
from dopplerdex.web.search import needed_data, run_search

ROOT = Path(__file__).resolve().parents[3]
UPLOADS = ROOT / "data/uploads"
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 1_000_000_000

app = FastAPI(title="Dopplerdex", docs_url="/api/docs")
jobs = JobManager()


def _upload_dir(upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", upload_id):
        raise HTTPException(400, "bad upload id")
    d = UPLOADS / upload_id
    if not d.exists():
        raise HTTPException(404, "unknown upload")
    return d


def _audio_path(d: Path) -> Path:
    files = [p for p in d.iterdir() if p.name != "meta.json"]
    if not files:
        raise HTTPException(404, "upload has no audio")
    return files[0]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {"continents": list(CONTINENTS), "ingested": data_mod.ingested()}


@app.get("/api/geocode")
def api_geocode(q: str) -> dict:
    if not q.strip():
        raise HTTPException(400, "empty query")
    hit = geocode(q.strip())
    if not hit:
        raise HTTPException(404, f"no place found for '{q}'")
    return hit


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    upload_id = uuid.uuid4().hex[:12]
    d = UPLOADS / upload_id
    d.mkdir(parents=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "audio")[:80] or "audio"
    size = 0
    with open(d / safe, "wb") as fh:
        while chunk := await file.read(1 << 22):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "file too large (max 1 GB)")
            fh.write(chunk)
    return {"upload_id": upload_id, "name": safe, "bytes": size}


@app.post("/api/analyze")
def api_analyze(upload_id: str = Form(...), meta: str = Form("{}")) -> dict:
    d = _upload_dir(upload_id)
    m = json.loads(meta)
    (d / "meta.json").write_text(json.dumps(m, indent=1))
    path = _audio_path(d)

    def task(progress=None) -> dict:
        res = analyse(str(path), float(m.get("temp_c") or 15.0), progress=progress)
        try:
            need = needed_data(m["when"], m["where"], res["duration_s"])
        except Exception as exc:  # incomplete form: still return the audio check
            need = {"error": str(exc)}
        res["data"] = need
        if "region" in need:
            v = res["verdict"]
            n_tonal = min(3, max(0, v["n_air"] - 1))
            n_loud = min(3 - n_tonal, v["n_loud"]) if v["n_air"] >= 1 else 0
            res["estimate"] = estimate_seconds(
                need["region"]["area_km2"],
                need["window"]["hours"],
                n_tonal,
                n_loud,
                need["state"],
                n_single=1 if v["n_air"] == 1 and n_loud == 0 else 0,
            )
        return res

    return jobs.submit("analyze", task).public()


@app.post("/api/search")
def api_search(upload_id: str = Form(...), meta: str = Form("{}")) -> dict:
    d = _upload_dir(upload_id)
    m = json.loads(meta)
    (d / "meta.json").write_text(json.dumps(m, indent=1))
    path = _audio_path(d)
    return jobs.submit("search", lambda progress=None: run_search(str(path), m, progress=progress), heavy=True).public()


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j.public(with_result=j.status == "done")


@app.get("/api/data/status")
def api_data_status(date: str, store: str) -> dict:
    return data_mod.status(date, store)


@app.post("/api/data/prepare")
def api_prepare(date: str = Form(...), store: str = Form(...)) -> dict:
    if store not in data_mod.REGIONS:
        raise HTTPException(400, f"unknown store '{store}'")
    return jobs.submit("prepare", lambda progress=None: data_mod.prepare(date, store, progress), heavy=True).public()


@app.get("/api/region")
def api_region(where: str) -> dict:
    _, info = search_region(json.loads(where))
    return info


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    print("Dopplerdex running at http://127.0.0.1:8765  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
