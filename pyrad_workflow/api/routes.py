from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from ..application import ClinicalRadiomicsPlatformService, WORKFLOW_NAMES
from ..infrastructure.settings import AppSettings


def install_routes(
    app: FastAPI,
    service: ClinicalRadiomicsPlatformService,
    settings: AppSettings,
    static_dir: Path,
) -> None:
    def serve_page(filename: str) -> FileResponse:
        page_path = static_dir / filename
        if not page_path.is_file():
            raise HTTPException(status_code=404, detail=f"Page not found: {filename}")
        return FileResponse(page_path)

    async def parse_payload(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return payload

    def error_response(message: str, status: int = 400) -> JSONResponse:
        return JSONResponse({"ok": False, "error": message}, status_code=status)

    def remember_output_dir(result: dict[str, Any]) -> None:
        output_dir = result.get("output_dir")
        if output_dir:
            app.state.download_roots.add(Path(str(output_dir)).resolve())

    def run_sync_endpoint(workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = service.run_workflow_sync(workflow, payload)
        remember_output_dir(result)
        return result

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return serve_page("index.html")

    @app.get("/validate", response_class=FileResponse)
    def validate_page() -> FileResponse:
        return serve_page("validate.html")

    @app.get("/extract", response_class=FileResponse)
    def extract_page() -> FileResponse:
        return serve_page("extract.html")

    @app.get("/select", response_class=FileResponse)
    def select_page() -> FileResponse:
        return serve_page("select.html")

    @app.get("/train", response_class=FileResponse)
    def train_page() -> FileResponse:
        return serve_page("train.html")

    @app.get("/full", response_class=FileResponse)
    def full_page() -> FileResponse:
        return serve_page("full.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "status": "healthy",
            "mode": "local",
            "workspace": str(settings.workspace),
        }

    @app.get("/api/config")
    @app.get("/api/v1/config")
    def api_config() -> dict[str, Any]:
        return {"defaults": service.default_paths()}

    @app.post("/api/inspect/data")
    @app.post("/api/v1/inspect/data")
    async def api_inspect_data(request: Request):
        try:
            return service.inspect_data(await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/inspect/features")
    @app.post("/api/v1/inspect/features")
    async def api_inspect_features(request: Request):
        try:
            return service.inspect_features(await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/validate")
    async def api_validate(request: Request):
        try:
            return run_sync_endpoint("validate", await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/extract")
    async def api_extract(request: Request):
        try:
            return run_sync_endpoint("extract", await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/select")
    async def api_select(request: Request):
        try:
            return run_sync_endpoint("select", await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/train")
    async def api_train(request: Request):
        try:
            return run_sync_endpoint("train", await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/full")
    async def api_full(request: Request):
        try:
            return run_sync_endpoint("full", await parse_payload(request))
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.post("/api/v1/workflows/{workflow_name}")
    async def api_workflow(workflow_name: str, request: Request):
        if workflow_name not in WORKFLOW_NAMES:
            raise HTTPException(status_code=404, detail=f"Unknown workflow: {workflow_name}")

        payload = await parse_payload(request)
        run_async = bool(payload.pop("run_async", True))
        try:
            if run_async:
                job = service.submit_workflow(workflow_name, payload)
                if job.result is not None:
                    remember_output_dir(job.result)
                return JSONResponse({"ok": True, "job": job.to_dict()}, status_code=202)

            return {"ok": True, "result": run_sync_endpoint(workflow_name, payload)}
        except HTTPException:
            raise
        except Exception as exc:
            return error_response(str(exc))

    @app.get("/api/v1/jobs/{job_id}")
    def api_job(job_id: str):
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        if job.result is not None:
            remember_output_dir(job.result)
        return {"ok": True, "job": job.to_dict()}

    @app.get("/download")
    @app.get("/api/v1/download")
    def download(path: str):
        candidate = Path(path).resolve()
        roots = app.state.download_roots
        if not any(root == candidate or root in candidate.parents for root in roots):
            raise HTTPException(status_code=403, detail="File is outside allowed download roots.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(candidate, filename=candidate.name)
