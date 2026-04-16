from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    workflow: str
    status: str
    created_at: str
    updated_at: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    cache_key: str | None
    progress: float
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, workflow: str, payload: dict[str, Any], cache_key: str | None = None) -> JobRecord:
        now = utc_now_iso()
        job = JobRecord(
            job_id=str(uuid.uuid4()),
            workflow=workflow,
            status="queued",
            created_at=now,
            updated_at=now,
            payload=payload,
            result=None,
            error=None,
            cache_key=cache_key,
            progress=0.0,
            detail="Queued",
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update_status(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        progress: float | None = None,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = JobRecord(
                job_id=current.job_id,
                workflow=current.workflow,
                status=status,
                created_at=current.created_at,
                updated_at=utc_now_iso(),
                payload=current.payload,
                result=result,
                error=error,
                cache_key=current.cache_key,
                progress=current.progress if progress is None else progress,
                detail=current.detail if detail is None else detail,
            )

    def update_progress(self, job_id: str, progress: float, detail: str | None = None) -> None:
        self.update_status(job_id, "running", progress=progress, detail=detail)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._cache.get(cache_key)
        return None if cached is None else dict(cached)

    def put_cached_result(self, cache_key: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._cache[cache_key] = dict(result)
