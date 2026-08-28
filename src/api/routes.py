"""
FastAPI routes.

This is OUR required HTTP API (assignment Section 19). Gemini/OpenRouter are
optional internal AI providers this API may call out to internally — they
are configured via VLM_PROVIDER and are not exposed as separate endpoints.
Swagger UI (/docs) and curl/Postman are just clients for testing this API.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.acquisition.downloader import AcquisitionError, FFmpegNotFoundError
from src.acquisition.validators import InvalidURLError, UnsupportedPlatformError
from src.asr.transcriber import ASRError
from src.config import get_settings
from src.logging_setup import get_logger
from src.media.frames import FrameExtractionError
from src.ocr.extractor import OCRError
from src.pipeline import PipelineError, run_pipeline
from src.schemas.response import DetectRequest, DetectionResponse, HealthResponse
from src.vision.base import VLMError

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", app_name=settings.APP_NAME)


@router.post("/api/v1/detect", response_model=DetectionResponse)
def detect(request: DetectRequest) -> DetectionResponse:
    logger.info("Received /api/v1/detect request for url=%s", request.url)
    try:
        result = run_pipeline(request.url, kind_hint=request.kind_hint, live_max_seconds=request.live_max_seconds)
        return result.response
    except (InvalidURLError, UnsupportedPlatformError) as e:
        logger.warning("Invalid request: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FFmpegNotFoundError as e:
        logger.error("Missing dependency: %s", e)
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: {e}") from e
    except AcquisitionError as e:
        logger.error("Acquisition failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Could not acquire video: {e}") from e
    except (ASRError, OCRError, VLMError, FrameExtractionError) as e:
        logger.error("Processing stage failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}") from e
    except PipelineError as e:
        logger.warning("Pipeline rejected request: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - last resort, never leak raw traceback to client
        logger.exception("Unexpected error processing %s", request.url)
        raise HTTPException(status_code=500, detail="Internal server error. See server logs for details.") from e
