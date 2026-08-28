import pytest
from pydantic import ValidationError

from src.schemas.response import AdSegment, AdType, DetectionResponse, Evidence, Kind, Platform, SourceInfo, Stats


def _make_segment(**overrides):
    defaults = dict(
        id="seg_01", start_s=10.0, end_s=20.0, ad_type=AdType.midroll_sponsor_read,
        confidence=0.8, brand="Acme", description="test",
        evidence=Evidence(frame_timestamps=[10.0], transcript_span="hi", signals_used=["asr"]),
    )
    defaults.update(overrides)
    return AdSegment(**defaults)


def test_valid_segment_constructs():
    seg = _make_segment()
    assert seg.duration_s == 10.0


def test_end_must_be_after_start():
    with pytest.raises(ValidationError):
        _make_segment(start_s=20.0, end_s=10.0)


def test_end_equal_start_is_rejected():
    with pytest.raises(ValidationError):
        _make_segment(start_s=10.0, end_s=10.0)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _make_segment(confidence=1.5)
    with pytest.raises(ValidationError):
        _make_segment(confidence=-0.1)


def test_negative_start_rejected():
    with pytest.raises(ValidationError):
        _make_segment(start_s=-1.0, end_s=5.0)


def test_invalid_ad_type_rejected():
    with pytest.raises(ValidationError):
        _make_segment(ad_type="not_a_real_type")


def test_full_response_round_trips_through_json():
    resp = DetectionResponse(
        source=SourceInfo(url="https://x.com/v", platform=Platform.youtube, kind=Kind.vod, duration_s=812.4),
        segments=[_make_segment()],
        stats=Stats(wall_clock_s=47.2, estimated_cost_usd=0.083, frames_sampled=214, model_calls=19),
    )
    dumped = resp.model_dump_json()
    reloaded = DetectionResponse.model_validate_json(dumped)
    assert reloaded.segments[0].id == "seg_01"


def test_segments_are_sorted_by_start_s():
    seg_a = _make_segment(id="a", start_s=50.0, end_s=60.0)
    seg_b = _make_segment(id="b", start_s=5.0, end_s=8.0)
    resp = DetectionResponse(
        source=SourceInfo(url="https://x.com/v", platform=Platform.youtube, kind=Kind.vod, duration_s=100.0),
        segments=[seg_a, seg_b],
        stats=Stats(wall_clock_s=1.0, estimated_cost_usd=0.0, frames_sampled=1, model_calls=0),
    )
    assert [s.id for s in resp.segments] == ["b", "a"]
