"""Audio extraction from video files using ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(video_path: Path, out_path: Path | None = None, sample_rate: int = 16000) -> Path:
    """
    Extract mono PCM WAV audio at `sample_rate` Hz — the format faster-whisper
    expects. Raises AudioExtractionError with the ffmpeg stderr tail on failure.
    """
    settings = get_settings()
    out_path = out_path or video_path.with_suffix(".wav")
    cmd = [
        settings.FFMPEG_PATH, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
        "-loglevel", "error",
        str(out_path),
    ]
    logger.info("Extracting audio: %s -> %s", video_path, out_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise AudioExtractionError(f"Audio extraction timed out for {video_path}") from e
    if result.returncode != 0 or not out_path.exists():
        tail = "\n".join((result.stderr or "").strip().splitlines()[-8:])
        raise AudioExtractionError(f"ffmpeg audio extraction failed for {video_path}: {tail}")
    return out_path
