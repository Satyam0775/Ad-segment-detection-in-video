"""
Pipeline orchestrator: URL/path -> DetectionResponse.

This is the single place that wires acquisition -> audio -> ASR -> frames ->
OCR -> VLM -> fusion -> segmentation -> stats, so both the FastAPI route and
scripts/run_pipeline.py call the exact same code path (no drift between
"what the API does" and "what the CLI script does").
"""
from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.acquisition.downloader import AcquiredMedia, download_youtube, ingest_local_file
from src.acquisition.validators import ParsedSource, validate_and_classify
from src.asr.transcriber import TranscriptSegment, Transcriber, query_transcript_window
from src.config import ensure_dirs, get_settings
from src.detection.fusion import fuse_votes, merge_vote_dicts
from src.detection.rules import mark_scene_cuts, score_ocr_results, score_transcript_segments, score_vlm_results
from src.detection.segmenter import build_segments
from src.logging_setup import get_logger
from src.media.audio import extract_audio
from src.media.frames import SampledFrame, extract_frames
from src.media.scenes import detect_scene_cuts
from src.ocr.extractor import OCRResult, run_ocr_on_frames
from src.schemas.response import DetectionResponse, Kind, Platform, SourceInfo, Stats
from src.vision.analyzer import analyze_selected_frames, get_vision_provider, select_candidate_frames

logger = get_logger(__name__)


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineResult:
    response: DetectionResponse
    work_dir: Path


def _run_detection_from_signals(
    source_url: str,
    media: AcquiredMedia,
    transcript_segments: List[TranscriptSegment],
    frames: List[SampledFrame],
    scene_cuts: List[float],
    ocr_results: List[OCRResult],
    wall_clock_start: float,
) -> DetectionResponse:
    provider = get_vision_provider()
    candidate_timestamps = list(merge_vote_dicts(score_transcript_segments(transcript_segments), score_ocr_results(ocr_results)).keys()) + scene_cuts

    # Recompute merged votes in the same order as the existing pipeline.
    asr_votes = score_transcript_segments(transcript_segments)
    ocr_votes = score_ocr_results(ocr_results)
    merged_votes = merge_vote_dicts(asr_votes, ocr_votes)
    mark_scene_cuts(merged_votes, scene_cuts)

    selected_frames = select_candidate_frames(
        frames, candidate_timestamps, max_calls=get_settings().VLM_MAX_CALLS_PER_VIDEO,
    )
    # Combine localized ASR transcript window and OCR text overlay outputs to provide rich multimodal context to the VLM
    context_by_ts = {}
    for f in selected_frames:
        ts = f.timestamp_s
        asr_part = query_transcript_window(transcript_segments, ts - 2.0, ts + 2.0)
        ocr_part = " ".join([r.text for r in ocr_results if abs(r.timestamp_s - ts) <= 2.0])
        context_by_ts[ts] = f"{asr_part} {ocr_part}".strip()

    vlm_results = analyze_selected_frames(selected_frames, context_by_ts, provider=provider)
    model_calls = len(vlm_results)
    estimated_cost = model_calls * provider.cost_per_call_usd

    vlm_votes = score_vlm_results(vlm_results)
    all_votes = merge_vote_dicts(merged_votes, vlm_votes)
    mark_scene_cuts(all_votes, scene_cuts)

    fused_points = fuse_votes(all_votes)
    segments = build_segments(fused_points, media.duration_s, len(frames))

    wall_clock = time.time() - wall_clock_start
    response = DetectionResponse(
        source=SourceInfo(
            url=source_url, platform=media.platform, kind=media.kind, duration_s=round(media.duration_s, 2),
        ),
        segments=segments,
        stats=Stats(
            wall_clock_s=round(wall_clock, 2),
            estimated_cost_usd=round(estimated_cost, 4),
            frames_sampled=len(frames),
            model_calls=model_calls,
        ),
    )
    return response


def run_pipeline_from_precomputed(
    source_url: str,
    media: AcquiredMedia,
    transcript_segments: List[TranscriptSegment],
    frames: List[SampledFrame],
    scene_cuts: List[float],
    ocr_results: List[OCRResult],
) -> DetectionResponse:
    """Run VLM + fusion + segmentation using precomputed ASR/OCR/frames/scene cuts.

    This reuses the same downstream logic as run_pipeline without redoing
    acquisition, frame extraction, scene detection, or OCR.
    """
    t0 = time.time()
    return _run_detection_from_signals(
        source_url=source_url,
        media=media,
        transcript_segments=transcript_segments,
        frames=frames,
        scene_cuts=scene_cuts,
        ocr_results=ocr_results,
        wall_clock_start=t0,
    )


def run_pipeline(url: str, kind_hint: Optional[Kind] = None, live_max_seconds: Optional[float] = None) -> PipelineResult:
    settings = get_settings()
    ensure_dirs()
    t0 = time.time()

    parsed: ParsedSource = validate_and_classify(url, kind_hint=kind_hint)
    logger.info("Processing url=%s platform=%s kind=%s", parsed.url, parsed.platform, parsed.kind)

    media: AcquiredMedia
    if parsed.platform == Platform.youtube:
        media = download_youtube(parsed.url, kind_hint=kind_hint)
    elif parsed.platform in (Platform.instagram, Platform.file):
        # Instagram Reels: assignment requires manual/local acquisition, never
        # automated scraping. `url` here is expected to be a local file path
        # for Instagram in practice; if a bare instagram.com URL reaches this
        # branch we fail clearly rather than attempting to scrape it.
        local_path = parsed.url[len("file://"):] if parsed.url.startswith("file://") else parsed.url
        if parsed.platform == Platform.instagram and (local_path.startswith("http")):
            raise PipelineError(
                "Automated Instagram acquisition is out of scope per the assignment brief. "
                "Capture the Reel manually (e.g. screen recording) and pass the local file path instead."
            )
        media = ingest_local_file(local_path, platform=parsed.platform, kind_hint=kind_hint)
    else:
        raise PipelineError(f"Unsupported platform for acquisition: {parsed.platform}")

    work_dir = settings.DATA_DIR / "downloads" / f"work_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"

    try:
        # --- Audio + ASR ---
        try:
            audio_path = extract_audio(media.local_path, out_path=work_dir / "audio.wav")
            transcript_segments = Transcriber().transcribe(audio_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("ASR stage failed, continuing with empty transcript: %s", e)
            transcript_segments = []

        # --- Frames + scene cuts ---
        frames = extract_frames(media.local_path, media.duration_s, frames_dir)
        try:
            scene_cuts = detect_scene_cuts(media.local_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Scene detection failed: %s", e)
            scene_cuts = []

        # --- OCR ---
        try:
            ocr_results = run_ocr_on_frames(frames)
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR stage failed, continuing without OCR signal: %s", e)
            ocr_results = []
        response = _run_detection_from_signals(
            source_url=parsed.url,
            media=media,
            transcript_segments=transcript_segments,
            frames=frames,
            scene_cuts=scene_cuts,
            ocr_results=ocr_results,
            wall_clock_start=t0,
        )

    finally:
        # Keep frames/audio for debugging by default; caller may clean up work_dir.
        pass

    logger.info(
        "Pipeline complete for %s: %d segments, %.1fs wall clock, $%.4f estimated",
        parsed.url, len(response.segments), response.stats.wall_clock_s, response.stats.estimated_cost_usd,
    )
    return PipelineResult(response=response, work_dir=work_dir)


def cleanup_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
