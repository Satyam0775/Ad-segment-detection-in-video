"""FastAPI application entrypoint. Run with: uvicorn src.main:app --reload"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config import PROJECT_ROOT, ensure_dirs, get_settings
from src.logging_setup import configure_logging

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
ensure_dirs()

app = FastAPI(
    title="Ad Segment Detector",
    description="Option B: Ad segment detection in video. Given a video URL, "
    "returns a machine-readable timeline of advertising segments.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

_viewer_dir = PROJECT_ROOT / "viewer"
if _viewer_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(_viewer_dir), html=True), name="viewer")

_downloads_dir = settings.DATA_DIR / "downloads"
if _downloads_dir.exists():
    app.mount(
        "/downloads",
        StaticFiles(directory=str(_downloads_dir)),
        name="downloads",
    )
