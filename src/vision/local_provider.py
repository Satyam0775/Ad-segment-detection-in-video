"""
Providers that require no external hosted API key.

MockVLMProvider: deterministic heuristic stand-in used by default
(VLM_PROVIDER=mock) and in tests, so the whole pipeline runs end-to-end
with zero cost and zero network dependency. It is intentionally simple —
it is NOT meant to be a real detector, only to keep the architecture honest
about where a real VLM call would plug in. Swap in Gemini/OpenRouter for
real detection quality.

LocalVLMProvider: a hook for a locally-hosted multimodal model (e.g. via
Ollama or a local llama.cpp server) reachable over HTTP. Left as a thin,
honestly-incomplete integration point — see the raised NotImplementedError
message for exactly what a real implementation needs, rather than silently
returning fake analysis under a "local" label.
"""
from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.logging_setup import get_logger
from src.vision.base import FrameAnalysis, VisionProvider, VLMError

logger = get_logger(__name__)


class MockVLMProvider(VisionProvider):
    """Zero-cost, zero-network stand-in. Flags frames whose filename-adjacent
    OCR/context text contains obvious promotional cues, so the pipeline is
    exercisable end-to-end without any API key. This is a scaffold for a
    real VLM call, not a substitute for one — do not use for real grading."""

    @property
    def cost_per_call_usd(self) -> float:
        return 0.0

    def analyze_frame(self, image_path: Path, context_text: str = "") -> FrameAnalysis:
        import re
        # Deliberately text-context-only heuristic. This mock does not inspect
        # image pixels and should never be represented as a real vision model.
        text = (context_text or "").lower()
        promo_terms = ["sponsor", "discount", "promo code", "buy now", "advertisement", "offer", "affiliate link"]
        
        # --- Generalized Multilingual/Noisy VLM Heuristics (Phase 2) ---
        phone_re = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}(?:[-.\s]?\d{3,5})?\b')
        price_re = re.compile(
            r'(?:rs\.?|inr|₹|usd|\$|only)\s*\d+(?:,\d+)*(?:\s*[\/-])?|\b\d+(?:,\d+)*\s*(?:/-|rs|only)\b',
            re.IGNORECASE
        )
        biz_core = [
            "clinic", "hospital", "specialist", "centre", "center", "diagnostics", "imaging", 
            "vascular", "dental", "implant", "cosmetic", "surgery", "treatment", "doctor", "clinique"
        ]
        biz_action = [
            "call", "phone", "book", "appointment", "offer", "discount", "price", "address", 
            "website", "contact", "consultation", "specialist", "expert", "best", "visit", "special"
        ]
        
        has_phone = False
        for match in phone_re.finditer(text):
            digits = re.sub(r'\D', '', match.group(0))
            if 8 <= len(digits) <= 12:
                has_phone = True
                break
                
        has_price = bool(price_re.search(text))
        has_biz_core = any(term in text for term in biz_core)
        has_biz_action = any(term in text for term in biz_action)
        has_combo = has_biz_core and has_biz_action
        
        hit = any(t in text for t in promo_terms) or has_phone or has_price or has_combo
        
        # Dynamic Brand Guessing Heuristic (Generic)
        brand_guess = None
        if hit:
            p_re = re.compile(
                r'\b([A-Z][a-zA-Z0-9_]{2,15}(?:\s+[A-Z][a-zA-Z0-9_]{2,15})*)\s+(?:centre|center|clinic|hospital|diagnostics|imaging|specialist|vascular)\b',
                re.IGNORECASE
            )
            brand_match = p_re.search(context_text or "")
            if brand_match:
                brand_guess = brand_match.group(1).strip()
            else:
                p_re_post = re.compile(
                    r'\b(?:centre|center|clinic|hospital|diagnostics|imaging|specialist|vascular|clinique)\s+([A-Z][a-zA-Z0-9_]{1,15})\b',
                    re.IGNORECASE
                )
                brand_match_post = p_re_post.search(context_text or "")
                if brand_match_post:
                    brand_guess = brand_match_post.group(0).strip()
                    
        return FrameAnalysis(
            is_promotional=hit,
            product_or_service_visible=hit,
            brand_or_logo_visible=bool(brand_guess),
            sponsor_like_visual=hit,
            product_placement=False,
            self_promotion=False,
            advertisement_likelihood=0.8 if hit else 0.05,
            brand_guess=brand_guess,
            explanation=(
                f"mock provider: detected advertisement cues or brand '{brand_guess}' in localized context window"
                if hit else "mock provider: no promotional keyword nearby"
            ),
        )


class LocalVLMProvider(VisionProvider):
    """Placeholder for a self-hosted multimodal model reachable over HTTP
    (e.g. Ollama with a vision model, or a local llama.cpp server exposing
    an OpenAI-compatible /v1/chat/completions endpoint)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def cost_per_call_usd(self) -> float:
        return 0.0

    def analyze_frame(self, image_path: Path, context_text: str = "") -> FrameAnalysis:
        raise VLMError(
            "LocalVLMProvider is an integration point, not a working implementation. "
            "To use a local model: point it at an OpenAI-compatible /v1/chat/completions "
            "endpoint (e.g. Ollama) and adapt gemini_provider.py's request/response shape. "
            "Set VLM_PROVIDER=mock to run the pipeline without a real VLM, or "
            "VLM_PROVIDER=gemini/openrouter with a real key for actual detection."
        )
