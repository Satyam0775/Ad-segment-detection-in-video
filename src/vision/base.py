"""
Vision/VLM provider abstraction.

IMPORTANT DISTINCTION (do not confuse these in docs or code):
  - This module's providers (Gemini / OpenRouter / local / mock) are
    OPTIONAL INTERNAL AI services this pipeline may call out to.
  - The REQUIRED API for this assignment is OUR OWN FastAPI service
    (see src/api/routes.py). Swagger/Postman/curl are just clients that
    test *our* API — they are unrelated to which VLM provider is configured.

All providers return a validated FrameAnalysis object. Raw model text is
never passed through to the final JSON contract unvalidated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FrameAnalysis:
    is_promotional: bool
    product_or_service_visible: bool
    brand_or_logo_visible: bool
    sponsor_like_visual: bool
    product_placement: bool
    self_promotion: bool
    advertisement_likelihood: float  # 0..1
    brand_guess: Optional[str]
    explanation: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.advertisement_likelihood <= 1.0):
            raise ValueError(
                f"advertisement_likelihood must be in [0,1], got {self.advertisement_likelihood}"
            )


class VLMError(RuntimeError):
    pass


class VisionProvider(ABC):
    @abstractmethod
    def analyze_frame(self, image_path: Path, context_text: str = "") -> FrameAnalysis:
        """Analyze a single frame, optionally with nearby ASR/OCR text as context."""

    @property
    @abstractmethod
    def cost_per_call_usd(self) -> float:
        """Rough per-call cost estimate used to populate stats.estimated_cost_usd."""
