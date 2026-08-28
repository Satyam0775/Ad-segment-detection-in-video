from src.detection.fusion import FusedPoint
from src.detection.segmenter import build_segments, cluster_points


def _pt(ts, conf, brand=None, signals=None):
    return FusedPoint(
        timestamp_s=ts, confidence=conf, transcript_span="", ocr_text="",
        brand_guess=brand, signals_used=signals or ["asr"], asr_self_promo_only=False,
    )


def test_nearby_points_merge_into_one_cluster():
    points = [_pt(120, 0.8), _pt(122, 0.7), _pt(124, 0.9)]
    clusters = cluster_points(points, threshold=0.5, merge_gap=3.0, split_gap=1.5)
    assert len(clusters) == 1
    assert clusters[0].start_s == 120
    assert clusters[0].end_s == 124


def test_far_apart_points_form_separate_clusters():
    points = [_pt(10, 0.8), _pt(200, 0.8)]
    clusters = cluster_points(points, threshold=0.5, merge_gap=3.0, split_gap=1.5)
    assert len(clusters) == 2


def test_below_threshold_points_are_dropped():
    points = [_pt(10, 0.2), _pt(12, 0.1)]
    clusters = cluster_points(points, threshold=0.5, merge_gap=3.0, split_gap=1.5)
    assert clusters == []


def test_back_to_back_different_brands_split_even_when_close():
    points = [_pt(100, 0.9, brand="Acme"), _pt(101, 0.9, brand="Zenith")]
    clusters = cluster_points(points, threshold=0.5, merge_gap=3.0, split_gap=1.5)
    assert len(clusters) == 2


def test_back_to_back_same_brand_merges():
    points = [_pt(100, 0.9, brand="Acme"), _pt(101, 0.9, brand="Acme")]
    clusters = cluster_points(points, threshold=0.5, merge_gap=3.0, split_gap=1.5)
    assert len(clusters) == 1


def test_build_segments_produces_valid_schema_objects():
    points = [_pt(10, 0.9), _pt(11, 0.85)]
    segments = build_segments(points, video_duration_s=100.0, frames_sampled=50)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.end_s > seg.start_s
    assert seg.id == "seg_01"


def test_empty_points_produce_no_segments():
    segments = build_segments([], video_duration_s=100.0, frames_sampled=50)
    assert segments == []
