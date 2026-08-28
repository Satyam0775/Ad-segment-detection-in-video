"""
Gemini vision provider.

Calls Gemini's generateContent REST endpoint with an inline base64 image and
a JSON-schema-constrained prompt, then validates the parsed response into
FrameAnalysis before it can reach the rest of the pipeline. Requires
VLM_API_KEY in the environment (never hard-coded).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

from src.config import get_settings
from src.logging_setup import get_logger
from src.vision.base import FrameAnalysis, VisionProvider, VLMError

logger = get_logger(__name__)

_PROMPT = """You are analyzing a single video frame for advertising-detection research.
Given the frame image and optional nearby transcript/OCR context, answer strictly as JSON
with these exact keys and types, no prose outside the JSON:
{
  "is_promotional": bool,
  "product_or_service_visible": bool,
  "brand_or_logo_visible": bool,
  "sponsor_like_visual": bool,
  "product_placement": bool,
  "self_promotion": bool,
  "advertisement_likelihood": float between 0 and 1,
  "brand_guess": string or null,
  "explanation": short string
}
Context text near this frame (ASR/OCR, may be empty): {context}
"""


class GeminiVisionProvider(VisionProvider):
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.VLM_API_KEY:
            raise VLMError("VLM_PROVIDER=gemini requires VLM_API_KEY to be set in the environment/.env")

    @property
    def cost_per_call_usd(self) -> float:
        # Rough placeholder for Gemini 2.0 Flash-tier per-image call; override
        # with real observed cost once you've run the test set (see EVAL.md).
        return 0.002

    def analyze_frame(self, image_path: Path, context_text: str = "") -> FrameAnalysis:
        image_bytes = image_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = _PROMPT.replace('{context}', context_text[:500])

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.VLM_MODEL}:generateContent?key={self.settings.VLM_API_KEY}"
        )
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        try:
            resp = httpx.post(url, json=body, timeout=self.settings.VLM_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            raise VLMError(f"Gemini call/parse failed for {image_path}: {e}") from e

        try:
            return FrameAnalysis(
                is_promotional=bool(parsed["is_promotional"]),
                product_or_service_visible=bool(parsed["product_or_service_visible"]),
                brand_or_logo_visible=bool(parsed["brand_or_logo_visible"]),
                sponsor_like_visual=bool(parsed["sponsor_like_visual"]),
                product_placement=bool(parsed["product_placement"]),
                self_promotion=bool(parsed["self_promotion"]),
                advertisement_likelihood=float(parsed["advertisement_likelihood"]),
                brand_guess=parsed.get("brand_guess"),
                explanation=str(parsed.get("explanation", ""))[:500],
            )
        except (KeyError, ValueError, TypeError) as e:
            raise VLMError(f"Gemini returned malformed JSON for {image_path}: {parsed!r} ({e})") from e

