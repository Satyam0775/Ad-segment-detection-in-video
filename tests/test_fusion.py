from pathlib import Path
from src.detection.fusion import fuse_votes, merge_vote_dicts
from src.detection.rules import (
    TimestampVotes,
    mark_scene_cuts,
    score_ocr_results,
    score_transcript_segments,
    score_vlm_results,
)
from src.asr.transcriber import TranscriptSegment
from src.ocr.extractor import OCRResult
from src.vision.analyzer import VLMFrameResult
from src.vision.base import FrameAnalysis


def test_sponsor_phrase_detected_in_transcript():
    segs = [TranscriptSegment(start=10.0, end=15.0, text="This video is sponsored by Acme VPN")]
    votes = score_transcript_segments(segs)
    assert 10.0 in votes
    assert votes[10.0].asr_sponsor_score == 1.0


def test_non_promotional_transcript_produces_no_votes():
    segs = [TranscriptSegment(start=10.0, end=15.0, text="Today we're talking about pasta recipes")]
    votes = score_transcript_segments(segs)
    assert votes == {}


def test_ocr_promo_terms_scored():
    results = [OCRResult(timestamp_s=5.0, text="USE CODE SAVE20 FOR 20% OFF", raw_boxes=3)]
    votes = score_ocr_results(results)
    assert 5.0 in votes
    assert votes[5.0].ocr_score > 0


def test_vlm_zero_likelihood_produces_no_vote():
    analysis = FrameAnalysis(
        is_promotional=False, product_or_service_visible=False, brand_or_logo_visible=False,
        sponsor_like_visual=False, product_placement=False, self_promotion=False,
        advertisement_likelihood=0.0, brand_guess=None, explanation="nothing",
    )
    votes = score_vlm_results([VLMFrameResult(timestamp_s=1.0, analysis=analysis)])
    assert votes == {}


def test_fuse_votes_combines_multiple_signals_higher_than_one():
    v_asr_only = {10.0: TimestampVotes(timestamp_s=10.0, asr_sponsor_score=1.0, signals_used=["asr"])}
    v_multi = {
        10.0: TimestampVotes(
            timestamp_s=10.0, asr_sponsor_score=1.0, ocr_score=1.0, vlm_score=1.0,
            signals_used=["asr", "ocr", "vlm_frame"],
        )
    }
    fused_asr_only = fuse_votes(v_asr_only)
    fused_multi = fuse_votes(v_multi)
    assert fused_multi[0].confidence >= fused_asr_only[0].confidence
    assert fused_multi[0].confidence <= 1.0


def test_merge_vote_dicts_combines_nearby_timestamps():
    asr_votes = {10.0: TimestampVotes(timestamp_s=10.0, asr_sponsor_score=1.0, signals_used=["asr"])}
    ocr_votes = {10.5: TimestampVotes(timestamp_s=10.5, ocr_score=0.8, signals_used=["ocr"])}
    merged = merge_vote_dicts(asr_votes, ocr_votes, tolerance_s=1.0)
    assert len(merged) == 1
    combined = list(merged.values())[0]
    assert combined.asr_sponsor_score == 1.0
    assert combined.ocr_score == 0.8
    assert set(combined.signals_used) == {"asr", "ocr"}


def test_scene_cut_marking():
    votes = {10.0: TimestampVotes(timestamp_s=10.0, asr_sponsor_score=1.0, signals_used=["asr"])}
    mark_scene_cuts(votes, scene_cuts=[10.3], tolerance_s=1.0)
    assert votes[10.0].scene_cut is True
    assert "scene_cut" in votes[10.0].signals_used


def test_noisy_multilingual_asr_phone_matching():
    # ASR containing business context + contact number should trigger CTA-like evidence.
    segs_phone = [TranscriptSegment(start=2.0, end=4.0, text="Our clinic consultation number is 98765.43210")]
    votes = score_transcript_segments(segs_phone)
    assert 2.0 in votes
    assert votes[2.0].asr_cta_score == 0.8
    assert "asr" in votes[2.0].signals_used

    # ASR containing a business combo + action term
    segs_combo = [TranscriptSegment(start=5.0, end=8.0, text="Visit our special diagnostics center for checkup")]
    votes_combo = score_transcript_segments(segs_combo)
    assert 5.0 in votes_combo
    assert votes_combo[5.0].asr_cta_score == 0.8


def test_ocr_generic_commercial_scoring():
    # OCR containing a price tag
    ocr_price = [OCRResult(timestamp_s=10.0, text="Baseline pricing only 4500/- rupees", raw_boxes=2)]
    votes = score_ocr_results(ocr_price)
    assert 10.0 in votes
    assert votes[10.0].ocr_score == 0.35

    # OCR containing a phone number and business keyword
    ocr_biz_phone = [OCRResult(timestamp_s=15.0, text="Vascular Centre: Call 91122.33445", raw_boxes=4)]
    votes_biz = score_ocr_results(ocr_biz_phone)
    assert 15.0 in votes_biz
    assert votes_biz[15.0].ocr_score == 0.85


def test_fusion_reinforcement():
    # Even partial signals can reinforce after active-signal normalization.
    v_asr = {10.0: TimestampVotes(timestamp_s=10.0, asr_cta_score=0.8, signals_used=["asr"])}
    v_ocr = {10.0: TimestampVotes(timestamp_s=10.0, ocr_score=0.5, signals_used=["ocr"])}
    merged = merge_vote_dicts(v_asr, v_ocr, tolerance_s=5.0)
    fused = fuse_votes(merged)
    assert fused[0].confidence > 0.55

    # Strong tri-signal evidence should remain high but bounded.
    v_asr_strong = {10.0: TimestampVotes(timestamp_s=10.0, asr_cta_score=0.8, signals_used=["asr"])}
    v_ocr_strong = {10.0: TimestampVotes(timestamp_s=10.0, ocr_score=0.8, vlm_score=0.8, signals_used=["ocr", "vlm_frame"])}
    merged_strong = merge_vote_dicts(v_asr_strong, v_ocr_strong, tolerance_s=5.0)
    fused_strong = fuse_votes(merged_strong)
    assert fused_strong[0].confidence > 0.55
    assert fused_strong[0].confidence <= 1.0


def test_dynamic_brand_guessing():
    from src.vision.local_provider import MockVLMProvider
    provider = MockVLMProvider()

    # Generic business + contact phrase should trigger promotional mode.
    analysis = provider.analyze_frame(Path("dummy.jpg"), context_text="Consult at Alpha Vascular Centre Call 98765-43210")
    assert analysis.is_promotional is True
    assert analysis.brand_guess == "Alpha Vascular"

    # Generic local-provider context cue remains deterministic.
    analysis_post = provider.analyze_frame(Path("dummy.jpg"), context_text="Visit Clinic Nova for appointment")
    assert analysis_post.is_promotional is True
    assert analysis_post.brand_guess is not None
