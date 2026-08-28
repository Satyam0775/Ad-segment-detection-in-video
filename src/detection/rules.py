"""
Rule-based signal extraction.

This module turns raw ASR/OCR/VLM output into per-timestamp "votes" for
ad-likelihood. It intentionally does NOT decide the final answer by itself
(`if "sponsor" in transcript: ad = True` is explicitly what the assignment
brief warns against) — it produces weighted evidence that fusion.py combines.

Keyword lists here are a STARTING SIGNAL, not the whole detector: OCR/ASR
text matching a promo term raises that timestamp's score, but VLM visual
judgment and scene-cut context also contribute, and thresholds are all in
config.py, not hard-coded here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.asr.transcriber import TranscriptSegment
from src.ocr.extractor import OCRResult
from src.vision.analyzer import VLMFrameResult

# Promotional language cues for ASR. Grouped so DESIGN.md can explain *why*
# each group exists (sponsor language vs. self-promo vs. CTA language), and
# so a reviewer can extend one group without touching the others.
_SPONSOR_PHRASES = [
    r"sponsor(?:ed)?\s+by", r"today'?s\s+video\s+is\s+sponsored", r"brought\s+to\s+you\s+by",
    r"paid\s+partnership", r"in\s+partnership\s+with",
]
_CTA_PHRASES = [
    r"promo\s?code", r"discount\s+code", r"use\s+code", r"link\s+in\s+(the\s+)?(description|bio)",
    r"affiliate\s+link", r"\d{1,2}%\s+off", r"free\s+trial", r"sign\s+up\s+(today|now)",
]
_SELF_PROMO_PHRASES = [
    r"subscribe\s+to\s+my", r"check\s+out\s+my", r"my\s+patreon", r"join\s+my\s+patreon",
    r"merch\s+store", r"my\s+other\s+channel",
]
_OCR_PROMO_TERMS = [
    "sponsored", "advertisement", "ad", "offer", "discount", "buy now",
    "promo code", "shop now", "sale", "% off", "subscribe", "affiliate",
]

_sponsor_re = re.compile("|".join(_SPONSOR_PHRASES), re.IGNORECASE)
_cta_re = re.compile("|".join(_CTA_PHRASES), re.IGNORECASE)
_self_promo_re = re.compile("|".join(_SELF_PROMO_PHRASES), re.IGNORECASE)

# --- Generalized Multilingual & Commercial Heuristics (Phase 1) ---
# Used to capture noisy ASR or visual OCR cues that don't match standard English phrases.
_PHONE_RE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}(?:[-.\s]?\d{3,5})?\b')
_PRICE_RE = re.compile(
    r'(?:rs\.?|inr|₹|usd|\$|only)\s*\d+(?:,\d+)*(?:\s*[\/-])?|\b\d+(?:,\d+)*\s*(?:/-|rs|only)\b',
    re.IGNORECASE
)

_BIZ_CORE = [
    "clinic", "hospital", "specialist", "centre", "center", "diagnostics", "imaging", 
    "vascular", "dental", "implant", "cosmetic", "surgery", "treatment", "doctor", "clinique"
]
_BIZ_ACTION = [
    "call", "phone", "book", "appointment", "offer", "discount", "price", "address", 
    "website", "contact", "consultation", "specialist", "expert", "best", "visit", "special"
]


def _commercial_feature_flags(text: str) -> tuple[bool, bool, bool, bool]:
    """Return (has_phone, has_price, has_business_core, has_business_action)."""
    text_lower = text.lower()
    has_phone = check_phone_number(text_lower)
    has_price = check_price(text_lower)
    has_business_core = any(term in text_lower for term in _BIZ_CORE)
    has_business_action = any(term in text_lower for term in _BIZ_ACTION)
    return has_phone, has_price, has_business_core, has_business_action

def check_phone_number(text: str) -> bool:
    """Helper to detect a valid looking phone number (8-12 digits when non-digits are stripped)."""
    for match in _PHONE_RE.finditer(text):
        digits = re.sub(r'\D', '', match.group(0))
        if 8 <= len(digits) <= 12:
            return True
    return False

def check_price(text: str) -> bool:
    """Helper to detect a currency/price pattern."""
    return bool(_PRICE_RE.search(text))

def check_business_promo(text: str) -> bool:
    """Helper to detect a combination of business keyword and commercial action."""
    text_lower = text.lower()
    has_phone, has_price, has_core, has_action = _commercial_feature_flags(text_lower)

    # Require business semantics plus at least one promotional intent marker.
    if has_core and (has_action or has_phone or has_price):
        matched_core = [t for t in _BIZ_CORE if t in text_lower]
        matched_action = [t for t in _BIZ_ACTION if t in text_lower]
        if has_phone or has_price:
            return True
        if set(matched_core) != set(matched_action) or len(matched_core) > 1 or len(matched_action) > 1:
            return True
    return False


@dataclass
class TimestampVotes:
    timestamp_s: float
    asr_sponsor_score: float = 0.0
    asr_cta_score: float = 0.0
    asr_self_promo_score: float = 0.0
    ocr_score: float = 0.0
    vlm_score: float = 0.0
    scene_cut: bool = False
    transcript_span: str = ""
    ocr_text: str = ""
    brand_guess: str | None = None
    signals_used: List[str] = field(default_factory=list)


def score_transcript_segments(segments: List[TranscriptSegment]) -> Dict[float, TimestampVotes]:
    """One vote-bucket per transcript segment, keyed by its start timestamp."""
    votes: Dict[float, TimestampVotes] = {}
    for seg in segments:
        v = TimestampVotes(timestamp_s=seg.start, transcript_span=seg.text)
        if _sponsor_re.search(seg.text):
            v.asr_sponsor_score = 1.0
            v.signals_used.append("asr")
        if _cta_re.search(seg.text):
            v.asr_cta_score = 1.0
            if "asr" not in v.signals_used:
                v.signals_used.append("asr")
        if _self_promo_re.search(seg.text):
            v.asr_self_promo_score = 1.0
            if "asr" not in v.signals_used:
                v.signals_used.append("asr")
        
        # --- Generalized Multilingual/Noisy ASR Commercial Heuristics ---
        # If we see a phone number, price, or business-action combination, map to CTA score
        if v.asr_cta_score == 0.0:
            has_phone, has_price, has_core, has_action = _commercial_feature_flags(seg.text)
            if has_core and (has_phone or has_price or has_action):
                v.asr_cta_score = 0.8  # Strong sub-cue, helper to trigger CTA
                if "asr" not in v.signals_used:
                    v.signals_used.append("asr")

        if v.asr_sponsor_score or v.asr_cta_score or v.asr_self_promo_score:
            votes[round(seg.start, 2)] = v
    return votes


def score_ocr_results(ocr_results: List[OCRResult]) -> Dict[float, TimestampVotes]:
    votes: Dict[float, TimestampVotes] = {}
    for r in ocr_results:
        text_lower = r.text.lower()
        hits = sum(1 for term in _OCR_PROMO_TERMS if term in text_lower)
        
        # --- Generalized Multilingual/Noisy OCR Commercial Heuristics ---
        # If we see a phone number, a price, or a business/service promotion, score it
        has_phone, has_price, has_core, has_action = _commercial_feature_flags(text_lower)
        has_biz_promo = has_core and (has_phone or has_price or has_action)
        
        ocr_score = 0.0
        if hits > 0:
            ocr_score = min(1.0, hits * 0.4)
        
        # Boost or trigger based on visual commercial properties (typical in clinical/reel overlay ads)
        if has_biz_promo:
            # Multilingual/local ads often encode commercial intent visually via
            # business/service terms + contact/price cues even when "sponsored"
            # is absent or ASR text is noisy.
            bonus = 0.75
            if has_phone and has_price and has_core:
                bonus = 0.9
            elif has_phone and has_core:
                bonus = 0.85
            ocr_score = max(ocr_score, bonus)
        elif has_phone or has_price:
            # Contact or price alone is weak evidence; keep non-zero but modest.
            ocr_score = max(ocr_score, 0.35)

        if ocr_score > 0.0:
            v = TimestampVotes(timestamp_s=r.timestamp_s, ocr_score=ocr_score, ocr_text=r.text)
            v.signals_used.append("ocr")
            votes[round(r.timestamp_s, 2)] = v
    return votes


def score_vlm_results(vlm_results: List[VLMFrameResult]) -> Dict[float, TimestampVotes]:
    votes: Dict[float, TimestampVotes] = {}
    for r in vlm_results:
        a = r.analysis
        if a.advertisement_likelihood > 0.0:
            v = TimestampVotes(
                timestamp_s=r.timestamp_s, vlm_score=a.advertisement_likelihood, brand_guess=a.brand_guess,
            )
            v.signals_used.append("vlm_frame")
            votes[round(r.timestamp_s, 2)] = v
    return votes


def mark_scene_cuts(votes_by_ts: Dict[float, TimestampVotes], scene_cuts: List[float], tolerance_s: float = 1.0) -> None:
    """Mutate existing vote buckets in place to flag whether a scene cut fell nearby."""
    for ts, v in votes_by_ts.items():
        if any(abs(ts - cut) <= tolerance_s for cut in scene_cuts):
            v.scene_cut = True
            if "scene_cut" not in v.signals_used:
                v.signals_used.append("scene_cut")
