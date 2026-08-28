"""
Ad-type classification: given a fused, temporally-grouped candidate segment,
decide which taxonomy label (Section 15 of the brief) it gets.

This is rule-based on purpose: the taxonomy question ("is this self_promo or
midroll_sponsor_read?") is a judgment call the brief wants us to *make and
defend* (Section 5, ambiguity pack), not something to leave to an opaque
model. See DESIGN.md for the write-up. Rulings encoded here are the
candidate's DEFAULT/PROPOSED behavior — you should review each one against
your own reading of the brief before submitting (see the REVIEW markers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.detection.fusion import FusedPoint
from src.schemas.response import AdType


@dataclass
class ClassifiedSegment:
    ad_type: AdType
    confidence: float
    brand: str | None
    description: str


def classify_segment(points: List[FusedPoint], position_in_video: float, video_duration: float) -> ClassifiedSegment:
    """
    position_in_video: start_s / video_duration, used to distinguish preroll
    from midroll purely on position when other signals are ambiguous.
    """
    max_conf = max((p.confidence for p in points), default=0.0)
    combined_signals = set()
    for p in points:
        combined_signals.update(p.signals_used)

    brand = next((p.brand_guess for p in points if p.brand_guess), None)
    transcript = " ".join(p.transcript_span for p in points if p.transcript_span).strip()
    ocr_text = " ".join(p.ocr_text for p in points if p.ocr_text).strip()

    all_self_promo_only = all(p.asr_self_promo_only for p in points) if points else False

    # REVIEW (ambiguity item 1 — Patreon/self-promo): our default ruling is that
    # pure creator self-promotion with no third-party brand is 'self_promo', a
    # distinct taxonomy bucket, not absence-of-ad. Confirm this matches your own
    # reading before submitting; the brief explicitly wants YOUR ruling, defended
    # in DESIGN.md, not an inherited default.
    if all_self_promo_only and not brand:
        ad_type = AdType.self_promo
    elif position_in_video <= 0.02:
        ad_type = AdType.preroll
    elif "affiliate" in transcript.lower() or "affiliate" in ocr_text.lower():
        ad_type = AdType.affiliate
    elif "scene_cut" in combined_signals and len(points) <= 2 and max_conf < 0.6:
        # Short, isolated, scene-cut-anchored, lower-confidence candidate:
        # consistent with a brief bumper card rather than a full sponsor read.
        # REVIEW (ambiguity item 3 — 1.4s bumper): confirm this heuristic boundary.
        ad_type = AdType.bumper
    else:
        ad_type = AdType.midroll_sponsor_read

    description = _build_description(ad_type, transcript, ocr_text, brand)
    return ClassifiedSegment(ad_type=ad_type, confidence=round(max_conf, 3), brand=brand, description=description)


def _build_description(ad_type: AdType, transcript: str, ocr_text: str, brand: str | None) -> str:
    brand_part = f" for {brand}" if brand else ""
    snippet = (transcript or ocr_text or "").strip()
    snippet = (snippet[:160] + "...") if len(snippet) > 160 else snippet
    base = {
        AdType.self_promo: f"Creator self-promotion{brand_part}.",
        AdType.preroll: f"Pre-roll advertisement{brand_part}.",
        AdType.affiliate: f"Affiliate/creator promotional link or code{brand_part}.",
        AdType.bumper: f"Short bumper/slate{brand_part}, likely a brief sponsor card.",
        AdType.midroll_sponsor_read: f"Mid-roll sponsor read{brand_part}.",
    }.get(ad_type, f"Detected advertisement segment{brand_part}.")
    if snippet:
        base += f' Evidence text: "{snippet}"'
    return base
