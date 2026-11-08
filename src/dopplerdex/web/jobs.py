"""Minimal in-process job manager: long tasks run in worker threads, the UI polls their progress."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def public(self, with_result: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 3),
             "message": self.message, "error": self.error, "elapsed_s": round((self.finished or time.time()) - (self.started or self.created), 1)}  # fmt: skip
        if with_result:
            d["result"] = self.result
        return d


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._fast = ThreadPoolExecutor(2, thread_name_prefix="fast")  # audio analysis
        self._heavy = ThreadPoolExecutor(1, thread_name_prefix="heavy")  # searches, data preparation (CPU/RAM heavy)

    def submit(self, kind: str, fn, *args, heavy: bool = False, **kw) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind)
        with self._lock:
            self._jobs[job.id] = job

        def progress(frac: float, msg: str) -> None:
            job.progress, job.message = max(job.progress, min(frac, 1.0)), msg

        def run() -> None:
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(*args, progress=progress, **kw)
                job.status, job.progress = "done", 1.0
            except Exception as exc:  # surfaced to the UI
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3)
            finally:
                job.finished = time.time()

        (self._heavy if heavy else self._fast).submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)
