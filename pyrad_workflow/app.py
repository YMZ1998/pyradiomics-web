from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import install_routes
from .application import ClinicalRadiomicsPlatformService
from .infrastructure import InMemoryJobStore, build_settings


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("pyrad_workflow")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def create_app(workspace: Path | None = None) -> FastAPI:
    logger = configure_logging()
    settings = build_settings(workspace)
    executor = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="pyrad-platform")
    service = ClinicalRadiomicsPlatformService(settings=settings, job_store=InMemoryJobStore(), executor=executor)
    static_dir = Path(__file__).resolve().parent / "static"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Clinical Radiomics Prediction Platform",
        summary="Layered FastAPI backend for radiomics validation, extraction, selection, training, and pipeline execution.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.workspace = settings.workspace
    app.state.download_roots = {settings.workspace.resolve()}
    logger.info("Application initialized: workspace=%s max_workers=%s", settings.workspace, settings.max_workers)

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    install_routes(app=app, service=service, settings=settings, static_dir=static_dir)
    return app


app = create_app()
