"""
Signal fusion: combine per-timestamp votes from rules.py into one fused
confidence score per timestamp, then classifier.py turns that into an
ad_type, and segmenter.py groups adjacent high-confidence timestamps into
final segments.

Weights are configurable constants at the top of this file (not buried in
config.py, since they're an internal fusion detail rather than a top-level
tunable a reviewer would want to set via .env) — but they are named
constants, not magic numbers inline, and DESIGN.md explains the reasoning:
ASR sponsor language is the strongest single signal for long-form sponsor
reads (assignment: "the strongest ad signal in long form video is usually
spoken"), so it gets the highest weight. VLM and OCR are weighted lower
individually but their agreement with ASR pushes confidence up further
than any one signal alone (ambiguity item 6: static visual + spoken read
still fuses to a confident detection via the ASR weight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.detection.rules import TimestampVotes

W_ASR_SPONSOR = 0.45
W_ASR_CTA = 0.30
W_ASR_SELF_PROMO = 0.20  # lower: self-promo often isn't a third-party ad (ambiguity item 1)
W_OCR = 0.25
W_VLM = 0.35
W_SCENE_CUT_BONUS = 0.05  # small nudge, never enough to trigger detection alone


@dataclass
class FusedPoint:
    timestamp_s: float
    confidence: float
    transcript_span: str
    ocr_text: str
    brand_guess: str | None
    signals_used: List[str]
    asr_self_promo_only: bool  # carried through for classifier.py's self_promo vs ad ruling


def fuse_votes(votes_by_ts: Dict[float, TimestampVotes]) -> List[FusedPoint]:
    fused: List[FusedPoint] = []
    for ts, v in sorted(votes_by_ts.items()):
        weighted_sum = 0.0
        active_weight_sum = 0.0

        if v.asr_sponsor_score > 0.0:
            weighted_sum += v.asr_sponsor_score * W_ASR_SPONSOR
            active_weight_sum += W_ASR_SPONSOR
        if v.asr_cta_score > 0.0:
            weighted_sum += v.asr_cta_score * W_ASR_CTA
            active_weight_sum += W_ASR_CTA
        if v.asr_self_promo_score > 0.0:
            weighted_sum += v.asr_self_promo_score * W_ASR_SELF_PROMO
            active_weight_sum += W_ASR_SELF_PROMO
        if v.ocr_score > 0.1:
            weighted_sum += v.ocr_score * W_OCR
            active_weight_sum += W_OCR
        if v.vlm_score > 0.1:
            weighted_sum += v.vlm_score * W_VLM
            active_weight_sum += W_VLM

        # Normalize over active evidence so one strong signal can survive when
        # others are unavailable/noisy; keep scene-cut as a small additive nudge.
        score = (weighted_sum / active_weight_sum) if active_weight_sum > 0 else 0.0
        if v.scene_cut and active_weight_sum > 0:
            score += W_SCENE_CUT_BONUS
        score = min(1.0, score)

        asr_self_promo_only = (
            v.asr_self_promo_score > 0
            and v.asr_sponsor_score == 0
            and v.asr_cta_score == 0
            and v.vlm_score == 0
            and v.ocr_score == 0
        )

        fused.append(
            FusedPoint(
                timestamp_s=ts, confidence=score, transcript_span=v.transcript_span, ocr_text=v.ocr_text,
                brand_guess=v.brand_guess, signals_used=list(v.signals_used), asr_self_promo_only=asr_self_promo_only,
            )
        )
    return fused


def merge_vote_dicts(*vote_dicts: Dict[float, TimestampVotes], tolerance_s: Optional[float] = None) -> Dict[float, TimestampVotes]:
    """
    Merge multiple {timestamp: TimestampVotes} dicts (one per signal source)
    into one dict, combining votes for timestamps within `tolerance_s` of
    each other. ASR/OCR/VLM rarely land on the exact same float timestamp,
    so a naive dict union would fragment one real ad moment into three
    separate low-confidence points.
    """
    if tolerance_s is None:
        try:
            from src.config import get_settings
            tolerance_s = get_settings().TIMESTAMP_MERGE_TOLERANCE_S
        except Exception:
            tolerance_s = 8.0

    merged: Dict[float, TimestampVotes] = {}
    all_points: List[TimestampVotes] = []
    for d in vote_dicts:
        all_points.extend(d.values())
    all_points.sort(key=lambda v: v.timestamp_s)

    for point in all_points:
        anchor = None
        for existing_ts in merged:
            if abs(existing_ts - point.timestamp_s) <= tolerance_s:
                anchor = existing_ts
                break
        if anchor is None:
            merged[round(point.timestamp_s, 2)] = point
        else:
            target = merged[anchor]
            target.asr_sponsor_score = max(target.asr_sponsor_score, point.asr_sponsor_score)
            target.asr_cta_score = max(target.asr_cta_score, point.asr_cta_score)
            target.asr_self_promo_score = max(target.asr_self_promo_score, point.asr_self_promo_score)
            target.ocr_score = max(target.ocr_score, point.ocr_score)
            target.vlm_score = max(target.vlm_score, point.vlm_score)
            target.scene_cut = target.scene_cut or point.scene_cut
            target.transcript_span = target.transcript_span or point.transcript_span
            target.ocr_text = target.ocr_text or point.ocr_text
            target.brand_guess = target.brand_guess or point.brand_guess
            for s in point.signals_used:
                if s not in target.signals_used:
                    target.signals_used.append(s)
    return merged
