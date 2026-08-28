"""
OpenRouter vision provider — uses the OpenAI-compatible chat/completions
shape with an image_url content part (base64 data URI). Works with any
vision-capable model OpenRouter proxies (set VLM_MODEL accordingly, e.g.
"google/gemini-2.0-flash-001" or "anthropic/claude-3.5-sonnet").
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

_SYSTEM_PROMPT = (
    "You analyze a single video frame for advertising-detection research. "
    "Respond with ONLY a JSON object, no markdown fences, no prose, with exactly these keys: "
    "is_promotional (bool), product_or_service_visible (bool), brand_or_logo_visible (bool), "
    "sponsor_like_visual (bool), product_placement (bool), self_promotion (bool), "
    "advertisement_likelihood (float 0-1), brand_guess (string or null), explanation (short string)."
)


class OpenRouterVisionProvider(VisionProvider):
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.VLM_API_KEY:
            raise VLMError("VLM_PROVIDER=openrouter requires VLM_API_KEY to be set in the environment/.env")

    @property
    def cost_per_call_usd(self) -> float:
        return 0.003  # placeholder; varies by underlying model, override after measuring real spend

    def analyze_frame(self, image_path: Path, context_text: str = "") -> FrameAnalysis:
        image_bytes = image_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64}"

        body = {
            "model": self.settings.VLM_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Nearby transcript/OCR context: {context_text[:500]!r}"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.settings.VLM_API_KEY}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(
                f"{self.settings.OPENROUTER_BASE_URL}/chat/completions",
                json=body, headers=headers, timeout=self.settings.VLM_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            raise VLMError(f"OpenRouter call/parse failed for {image_path}: {e}") from e

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
            raise VLMError(f"OpenRouter returned malformed JSON for {image_path}: {parsed!r} ({e})") from e
