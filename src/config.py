"""
Central configuration for the Ad Segment Detector.

Every tunable used anywhere in the pipeline lives here and is overridable
via environment variables or a local .env file. No real secret should ever
be committed here — see .env.example for the documented list of variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General ---
    APP_NAME: str = "ad-segment-detector"
    LOG_LEVEL: str = "INFO"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

    # --- Acquisition ---
    VIDEO_MAX_HEIGHT: int = Field(default=720, description="Max download height, e.g. 720 for 720p")
    DOWNLOAD_TIMEOUT: int = Field(default=600, description="Seconds before a download is aborted")
    FFMPEG_PATH: str = Field(default="ffmpeg", description="Path to ffmpeg binary, or 'ffmpeg' if on PATH")
    FFPROBE_PATH: str = Field(default="ffprobe", description="Path to ffprobe binary, or 'ffprobe' if on PATH")
    YTDLP_COOKIES_FILE: Optional[str] = Field(default=None, description="Optional cookies.txt for gated content")

    # --- Frame sampling ---
    FRAME_SAMPLE_INTERVAL: float = Field(default=2.0, description="Base seconds between sampled frames")
    FRAME_DENSE_SAMPLE_INTERVAL: float = Field(
        default=0.5, description="Denser seconds-between-frames used around candidate ad regions / scene cuts"
    )
    SHORT_VIDEO_DURATION_THRESHOLD_S: float = Field(
        default=120.0, description="Videos shorter than this (Shorts/Reels) use dense sampling throughout"
    )
    BUMPER_GUARD_WINDOW_S: float = Field(
        default=3.0,
        description="Seconds of forced dense sampling injected around every detected scene cut, "
        "so a bumper shorter than the base interval is not skipped between two samples",
    )

    # --- ASR ---
    WHISPER_MODEL: str = Field(default="small", description="faster-whisper size: tiny|base|small|medium|large-v3")
    WHISPER_DEVICE: str = Field(default="cpu", description="cpu or cuda")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", description="int8|float16|float32")
    ASR_LANGUAGE: Optional[str] = Field(default=None, description="Force language e.g. 'en'; None = auto-detect")

    # --- OCR ---
    OCR_PROVIDER: str = Field(default="easyocr", description="easyocr is the only wired-in provider by default")
    OCR_LANGUAGES: str = Field(default="en", description="Comma separated language codes for OCR")

    # --- Vision / VLM ---
    VLM_PROVIDER: str = Field(default="mock", description="mock|gemini|openrouter|local")
    VLM_API_KEY: Optional[str] = Field(default=None, description="API key for the selected hosted VLM provider")
    VLM_MODEL: str = Field(default="gemini-2.0-flash", description="Model name for the selected provider")
    VLM_MAX_CALLS_PER_VIDEO: int = Field(default=40, description="Hard cap on VLM calls per video to bound cost")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    VLM_TIMEOUT_S: float = Field(default=30.0, description="Per-call timeout for hosted VLM providers")

    # --- Detection thresholds (configurable, never hard-coded in logic) ---
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(default=0.55, description="Min fused confidence to emit a segment")
    TIMESTAMP_MERGE_TOLERANCE_S: float = Field(
        default=8.0, description="Tolerance (s) within which ASR, OCR, and VLM signals are merged during fusion"
    )
    SEGMENT_MERGE_GAP_S: float = Field(
        default=15.0, description="Candidate detections within this gap (s) are merged into one segment"
    )
    MIN_SEGMENT_DURATION_S: float = Field(default=1.0, description="Segments shorter than this are flagged, not dropped")
    BACK_TO_BACK_SPLIT_GAP_S: float = Field(
        default=1.5,
        description="If two ad candidates differ in brand/type and the gap between them is below this, "
        "they are still split into two segments rather than merged (see ambiguity item 8)",
    )

    # --- Live stream (Tier 2) ---
    LIVE_WINDOW_SECONDS: float = Field(default=30.0, description="Rolling window size for live-stream processing")
    LIVE_WINDOW_OVERLAP_S: float = Field(
        default=5.0, description="Overlap between consecutive live windows so boundary-straddling ads aren't lost"
    )

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    @property
    def ocr_language_list(self) -> List[str]:
        return [x.strip() for x in self.OCR_LANGUAGES.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_dirs() -> None:
    s = get_settings()
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)
    s.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (s.DATA_DIR / "downloads").mkdir(parents=True, exist_ok=True)
    (s.DATA_DIR / "frames").mkdir(parents=True, exist_ok=True)
