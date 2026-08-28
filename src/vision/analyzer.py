"""
Vision analyzer: provider factory + selective calling strategy.

We do NOT call the VLM on every sampled frame — that's the fastest way to
blow the $10 test-set budget. Instead, analyze_selected_frames only calls
the VLM on frames that ASR/OCR/scene signals already flagged as candidates,
plus a light background sample so purely-visual ads (no speech, no on-screen
text — ambiguity item 6) aren't missed entirely. VLM_MAX_CALLS_PER_VIDEO is
a hard ceiling regardless of how many candidates are found.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.config import get_settings
from src.logging_setup import get_logger
from src.media.frames import SampledFrame
from src.vision.base import FrameAnalysis, VisionProvider, VLMError

logger = get_logger(__name__)


@dataclass
class VLMFrameResult:
    timestamp_s: float
    analysis: FrameAnalysis


def get_vision_provider() -> VisionProvider:
    settings = get_settings()
    provider = settings.VLM_PROVIDER.lower()
    if provider == "mock":
        from src.vision.local_provider import MockVLMProvider
        return MockVLMProvider()
    if provider == "gemini":
        from src.vision.gemini_provider import GeminiVisionProvider
        return GeminiVisionProvider()
    if provider == "openrouter":
        from src.vision.openrouter_provider import OpenRouterVisionProvider
        return OpenRouterVisionProvider()
    if provider == "local":
        from src.vision.local_provider import LocalVLMProvider
        return LocalVLMProvider()
    raise VLMError(f"Unknown VLM_PROVIDER: {settings.VLM_PROVIDER!r}. Use mock|gemini|openrouter|local.")


def select_candidate_frames(
    frames: List[SampledFrame],
    candidate_timestamps: List[float],
    max_calls: int,
    background_sample_every_n: int = 8,
) -> List[SampledFrame]:
    """Pick which frames to actually send to the (possibly costly) VLM."""
    candidate_set = set(round(t, 2) for t in candidate_timestamps)
    prioritized = [f for f in frames if round(f.timestamp_s, 2) in candidate_set]
    background = [f for i, f in enumerate(frames) if i % background_sample_every_n == 0]
    combined: Dict[float, SampledFrame] = {}
    for f in prioritized + background:
        combined[f.timestamp_s] = f
    ordered = sorted(combined.values(), key=lambda f: f.timestamp_s)
    return ordered[:max_calls]


def analyze_selected_frames(
    frames_to_analyze: List[SampledFrame],
    context_by_timestamp: Dict[float, str],
    provider: VisionProvider | None = None,
) -> List[VLMFrameResult]:
    settings = get_settings()
    provider = provider or get_vision_provider()
    results: List[VLMFrameResult] = []
    for frame in frames_to_analyze[: settings.VLM_MAX_CALLS_PER_VIDEO]:
        context = context_by_timestamp.get(frame.timestamp_s, "")
        try:
            analysis = provider.analyze_frame(frame.path, context_text=context)
        except VLMError as e:
            logger.warning("VLM analysis failed at t=%.2fs: %s", frame.timestamp_s, e)
            continue
        results.append(VLMFrameResult(timestamp_s=frame.timestamp_s, analysis=analysis))
    logger.info("VLM analyzed %d/%d selected frames", len(results), len(frames_to_analyze))
    return results
