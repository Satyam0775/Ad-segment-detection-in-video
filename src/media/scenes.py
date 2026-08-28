"""
Scene / shot-change detection.

We use ffmpeg's `select='gt(scene,THRESHOLD)'` filter, which is cheap (single
decode pass, no extra dependency) and good enough to flag hard cuts — the
moments where a sponsor bumper or ad slate is most likely to start or end.
Scene cuts feed two things downstream:
  1. the frame sampler, which densifies sampling for a few seconds around a cut
     (this is the concrete answer to ambiguity item 3 — the 1.4s bumper)
  2. the signal fusion layer, as a weak independent signal
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import List

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


class SceneDetectionError(RuntimeError):
    pass


_PTS_TIME_RE = re.compile(r"pts_time:([0-9.]+)")


def detect_scene_cuts(video_path: Path, threshold: float = 0.35) -> List[float]:
    """
    Return a sorted list of timestamps (seconds) where a scene/shot change
    was detected. threshold is ffmpeg's 0..1 scene-change score; 0.35-0.4 is
    a reasonable default for hard cuts without being too sensitive to pans.
    """
    settings = get_settings()
    cmd = [
        settings.FFMPEG_PATH, "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
        "-loglevel", "info",
    ]
    logger.info("Detecting scene cuts in %s (threshold=%.2f)", video_path, threshold)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise SceneDetectionError(f"Scene detection timed out for {video_path}") from e

    # ffmpeg writes showinfo to stderr regardless of return code semantics here.
    stderr = result.stderr or ""
    timestamps = [float(m.group(1)) for m in _PTS_TIME_RE.finditer(stderr)]
    timestamps = sorted(set(round(t, 2) for t in timestamps))
    logger.info("Found %d scene cuts in %s", len(timestamps), video_path)
    return timestamps


def save_scene_cuts(scene_cuts: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([float(x) for x in scene_cuts], indent=2), encoding="utf-8")


def load_scene_cuts(path: Path) -> List[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [float(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("scene_cuts"), list):
        return [float(x) for x in data["scene_cuts"]]
    raise SceneDetectionError(f"Scene cuts JSON must be a list or dict with scene_cuts: {path}")
