"""
Configurable frame sampling.

Design (see DESIGN.md Section "Frame sampling" for the full reasoning):

- Uniform 1fps-style sampling is a *decision*, not a default — and on its
  own it will miss short bumpers (ambiguity item 3: a 1.4s "sponsored by"
  card can fall entirely between two samples 2s apart).
- We therefore sample at FRAME_SAMPLE_INTERVAL everywhere, AND inject dense
  sampling (FRAME_DENSE_SAMPLE_INTERVAL) inside a BUMPER_GUARD_WINDOW_S
  window around every detected scene cut. A scene cut is cheap to compute
  (see scenes.py) and is a strong prior for "something changed here that's
  worth looking at closely," which is exactly where short bumpers live.
- Short-form video (Shorts/Reels, duration < SHORT_VIDEO_DURATION_THRESHOLD_S)
  uses dense sampling throughout, because these formats are short enough
  that the cost of dense sampling is bounded and the ad content is often
  the entire clip.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.config import get_settings
from src.logging_setup import get_logger
from src.media.scenes import detect_scene_cuts

logger = get_logger(__name__)


class FrameExtractionError(RuntimeError):
    pass


@dataclass
class SampledFrame:
    timestamp_s: float
    path: Path


_FRAME_RE = re.compile(r"^frame_([0-9]+(?:\.[0-9]+)?)$")


def build_sample_timestamps(duration_s: float, scene_cuts: List[float]) -> List[float]:
    """Pure function: decide which timestamps to sample. Kept separate from
    disk I/O so it is trivially unit-testable."""
    settings = get_settings()
    base_interval = settings.FRAME_SAMPLE_INTERVAL
    dense_interval = settings.FRAME_DENSE_SAMPLE_INTERVAL
    guard = settings.BUMPER_GUARD_WINDOW_S

    if duration_s <= settings.SHORT_VIDEO_DURATION_THRESHOLD_S:
        base_interval = dense_interval  # short-form: dense throughout

    timestamps = set()
    # Prevent sampling exactly at or past decodable end of the video, which triggers errors on fractional durations
    max_t = max(0.0, duration_s - 0.05) if duration_s > 0.1 else duration_s
    t = 0.0
    while t <= max_t:
        timestamps.add(round(t, 2))
        t += base_interval

    for cut in scene_cuts:
        lo, hi = max(0.0, cut - guard / 2), min(max_t, cut + guard / 2)
        t = lo
        while t <= hi:
            timestamps.add(round(t, 2))
            t += dense_interval

    return sorted(timestamps)


def extract_frames(video_path: Path, duration_s: float, out_dir: Path) -> List[SampledFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    try:
        scene_cuts = detect_scene_cuts(video_path)
    except Exception as e:  # noqa: BLE001 - scene detection is best-effort
        logger.warning("Scene detection failed (%s); continuing with uniform sampling only", e)
        scene_cuts = []

    timestamps = build_sample_timestamps(duration_s, scene_cuts)
    logger.info("Sampling %d frames from %s (duration=%.1fs)", len(timestamps), video_path, duration_s)

    frames: List[SampledFrame] = []
    for ts in timestamps:
        out_path = out_dir / f"frame_{ts:.2f}.jpg"
        cmd = [
            settings.FFMPEG_PATH, "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", "-loglevel", "error", str(out_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Frame extraction timed out at t=%.2fs, skipping", ts)
            continue
        if result.returncode != 0 or not out_path.exists():
            logger.warning("Frame extraction failed at t=%.2fs: %s", ts, (result.stderr or "").strip()[-200:])
            continue
        frames.append(SampledFrame(timestamp_s=ts, path=out_path))

    if not frames:
        raise FrameExtractionError(f"No frames could be extracted from {video_path}")
    return frames


def load_sampled_frames_from_dir(frames_dir: Path) -> List[SampledFrame]:
    if not frames_dir.exists():
        raise FrameExtractionError(f"Frames directory not found: {frames_dir}")

    frames: List[SampledFrame] = []
    for p in sorted(frames_dir.glob("frame_*.jpg"), key=lambda x: x.name):
        m = _FRAME_RE.match(p.stem)
        if not m:
            continue
        frames.append(SampledFrame(timestamp_s=float(m.group(1)), path=p))

    frames.sort(key=lambda f: f.timestamp_s)
    if not frames:
        raise FrameExtractionError(f"No frame_*.jpg files found in: {frames_dir}")
    return frames
