"""
Video acquisition: yt-dlp for YouTube, local-file passthrough for
Instagram Reels (per assignment: acquisition is manual/local, not automated)
and any other pre-downloaded media.

Design choices (see DESIGN.md for the full write-up):
  - We never request more than VIDEO_MAX_HEIGHT to keep cost/latency bounded.
  - yt-dlp failures, missing ffmpeg, timeouts, and dead URLs all raise a
    typed AcquisitionError with a human-readable reason instead of leaking
    raw subprocess tracebacks up to the API layer.
  - We probe metadata (duration, is_live) via yt-dlp -J before committing to
    a full download, which lets us correctly classify vod vs live even when
    the URL alone was ambiguous.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.logging_setup import get_logger
from src.schemas.response import Kind, Platform

logger = get_logger(__name__)


class AcquisitionError(RuntimeError):
    """Raised for any acquisition failure: missing tool, dead URL, network, timeout."""


class FFmpegNotFoundError(AcquisitionError):
    pass


@dataclass
class AcquiredMedia:
    local_path: Path
    duration_s: float
    platform: Platform
    kind: Kind
    title: Optional[str] = None
    is_live: bool = False


def _which_or_raise(binary: str, hint: str) -> str:
    resolved = shutil.which(binary) or (binary if Path(binary).exists() else None)
    if not resolved:
        raise FFmpegNotFoundError(
            f"Required binary '{binary}' was not found on PATH. {hint}"
        )
    return resolved


def check_ffmpeg_available() -> bool:
    settings = get_settings()
    return shutil.which(settings.FFMPEG_PATH) is not None or Path(settings.FFMPEG_PATH).exists()


def check_ytdlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def probe_youtube_metadata(url: str, timeout: int) -> dict:
    """Run `yt-dlp -J` (dump JSON, no download) to get duration/is_live/title cheaply."""
    if not check_ytdlp_available():
        raise AcquisitionError("yt-dlp is not installed or not on PATH. Run: pip install yt-dlp")
    cmd = ["yt-dlp", "-J", "--no-warnings", "--skip-download", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise AcquisitionError(f"Metadata probe timed out after {timeout}s for URL: {url}") from e
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        # yt-dlp is chatty; surface only the last few lines to keep errors readable.
        tail = "\n".join(stderr.splitlines()[-5:])
        raise AcquisitionError(f"yt-dlp could not read this URL (it may be dead, private, or region-locked): {tail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AcquisitionError(f"yt-dlp returned non-JSON metadata for URL: {url}") from e


def download_youtube(url: str, kind_hint: Optional[Kind] = None) -> AcquiredMedia:
    settings = get_settings()
    from src.config import ensure_dirs

    ensure_dirs()

    if not check_ytdlp_available():
        raise AcquisitionError("yt-dlp is not installed. Install with: pip install yt-dlp")
    ffmpeg_bin = _which_or_raise(settings.FFMPEG_PATH, "Install FFmpeg and ensure it is on PATH, or set FFMPEG_PATH in .env")

    meta = probe_youtube_metadata(url, timeout=min(60, settings.DOWNLOAD_TIMEOUT))
    is_live = bool(meta.get("is_live") or meta.get("live_status") in ("is_live", "post_live"))
    duration = float(meta.get("duration") or 0.0)
    title = meta.get("title")

    if kind_hint is not None:
        kind = kind_hint
    elif is_live:
        kind = Kind.live
    elif duration and duration <= 90:
        kind = Kind.short
    else:
        kind = Kind.vod

    out_dir = settings.DATA_DIR / "downloads"
    out_template = str(out_dir / "%(id)s.%(ext)s")
    fmt = f"bestvideo[height<={settings.VIDEO_MAX_HEIGHT}]+bestaudio/best[height<={settings.VIDEO_MAX_HEIGHT}]"

    cmd = [
        "yt-dlp",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-warnings",
        "--no-playlist",
        url,
    ]

    # Only pass --ffmpeg-location if settings.FFMPEG_PATH is a specific custom path,
    # or if we are not utilizing the standard system PATH.
    if settings.FFMPEG_PATH != "ffmpeg" or not shutil.which("ffmpeg"):
        cmd.extend(["--ffmpeg-location", str(Path(ffmpeg_bin).parent)])

    if settings.YTDLP_COOKIES_FILE:
        cmd.extend(["--cookies", settings.YTDLP_COOKIES_FILE])
    if is_live:
        # For a truly live URL we still let yt-dlp capture from "now"; Tier-2 live handling
        # (windowed processing) operates on the growing file. A max duration keeps test runs bounded.
        cmd.extend(["--live-from-start"]) if meta.get("live_status") == "is_live" else None

    logger.info("Starting yt-dlp download: %s (kind=%s, target_height<=%s)", url, kind, settings.VIDEO_MAX_HEIGHT)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=settings.DOWNLOAD_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise AcquisitionError(
            f"Download timed out after {settings.DOWNLOAD_TIMEOUT}s. "
            f"For live streams, consider a shorter test window (live_max_seconds)."
        ) from e
    elapsed = time.time() - t0

    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-8:])
        raise AcquisitionError(f"yt-dlp download failed for {url} after {elapsed:.1f}s: {tail}")

    video_id = meta.get("id") or "unknown"
    expected_mp4 = out_dir / f"{video_id}.mp4"
    if expected_mp4.exists():
        local_path = expected_mp4
    else:
        # Avoid picking temporary unmerged stream files like {video_id}.f398.mp4 or {video_id}.f251.webm
        candidates = list(out_dir.glob(f"{video_id}.*"))
        exact_stem_matches = [p for p in candidates if p.stem == video_id]
        if exact_stem_matches:
            mp4s = [p for p in exact_stem_matches if p.suffix == ".mp4"]
            local_path = mp4s[0] if mp4s else exact_stem_matches[0]
        else:
            mp4s = [p for p in candidates if p.suffix == ".mp4" and not p.stem.startswith(f"{video_id}.f")]
            local_path = mp4s[0] if mp4s else (candidates[0] if candidates else None)

    if not local_path or not local_path.exists():
        raise AcquisitionError(f"yt-dlp reported success but no output file was found for id={video_id}")

    logger.info("Downloaded %s -> %s in %.1fs", url, local_path, elapsed)

    if not has_audio_stream(local_path):
        raise AcquisitionError(
            f"The downloaded media file '{local_path.name}' does not contain an audio stream, "
            f"which is required for ASR processing. This indicates stream-merging failed."
        )

    resolved_duration = duration or ffprobe_duration(local_path)
    return AcquiredMedia(
        local_path=local_path, duration_s=resolved_duration, platform=Platform.youtube,
        kind=kind, title=title, is_live=is_live,
    )


def ingest_local_file(path: str, platform: Platform, kind_hint: Optional[Kind] = None) -> AcquiredMedia:
    """
    Ingest a local media file — used for manually-captured Instagram Reels
    (per the assignment's Reels rule) and for a recorded fallback when a
    live stream is offline at run time.
    """
    p = Path(path)
    if not p.exists():
        raise AcquisitionError(f"Local file does not exist: {p}")
    duration = ffprobe_duration(p)
    if not has_audio_stream(p):
        raise AcquisitionError(
            f"The ingested local file '{p.name}' does not contain an audio stream, "
            f"which is required for processing."
        )
    kind = kind_hint or (Kind.short if duration <= 90 else Kind.vod)
    return AcquiredMedia(local_path=p, duration_s=duration, platform=platform, kind=kind, title=p.stem)


def has_audio_stream(path: Path) -> bool:
    """Check if the media file contains at least one audio stream."""
    settings = get_settings()
    ffprobe = _which_or_raise(settings.FFPROBE_PATH, "Install FFmpeg (includes ffprobe) and ensure it is on PATH")
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        str(path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise AcquisitionError(f"ffprobe timed out verifying audio for {path}") from e
    if result.returncode != 0:
        raise AcquisitionError(f"ffprobe failed to inspect streams for {path}: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise AcquisitionError(f"Could not parse streams from ffprobe output for {path}") from e


def ffprobe_duration(path: Path) -> float:
    settings = get_settings()
    ffprobe = _which_or_raise(settings.FFPROBE_PATH, "Install FFmpeg (includes ffprobe) and ensure it is on PATH")
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise AcquisitionError(f"ffprobe timed out inspecting {path}") from e
    if result.returncode != 0:
        raise AcquisitionError(f"ffprobe failed to read duration for {path}: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise AcquisitionError(f"Could not parse duration from ffprobe output for {path}") from e
