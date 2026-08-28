from pathlib import Path

from src.asr.transcriber import (
    TranscriptSegment,
    load_transcript_segments,
    save_transcript_segments,
)
from src.media.frames import SampledFrame, load_sampled_frames_from_dir
from src.media.scenes import load_scene_cuts, save_scene_cuts
from src.ocr.extractor import OCRResult, load_ocr_results, save_ocr_results


def test_transcript_round_trip(tmp_path: Path):
    p = tmp_path / "transcript.json"
    inp = [TranscriptSegment(start=1.0, end=2.5, text="hello")]
    save_transcript_segments(inp, p)
    out = load_transcript_segments(p)
    assert len(out) == 1
    assert out[0].start == 1.0
    assert out[0].end == 2.5
    assert out[0].text == "hello"


def test_ocr_round_trip(tmp_path: Path):
    p = tmp_path / "ocr.json"
    inp = [OCRResult(timestamp_s=3.0, text="SALE", raw_boxes=2)]
    save_ocr_results(inp, p)
    out = load_ocr_results(p)
    assert len(out) == 1
    assert out[0].timestamp_s == 3.0
    assert out[0].text == "SALE"
    assert out[0].raw_boxes == 2


def test_scene_cuts_round_trip(tmp_path: Path):
    p = tmp_path / "scene.json"
    inp = [1.0, 2.5, 3.75]
    save_scene_cuts(inp, p)
    out = load_scene_cuts(p)
    assert out == [1.0, 2.5, 3.75]


def test_load_sampled_frames_from_dir(tmp_path: Path):
    d = tmp_path / "frames"
    d.mkdir()
    (d / "frame_10.00.jpg").write_bytes(b"")
    (d / "frame_2.50.jpg").write_bytes(b"")
    (d / "frame_1.00.jpg").write_bytes(b"")

    frames = load_sampled_frames_from_dir(d)
    assert [f.timestamp_s for f in frames] == [1.0, 2.5, 10.0]
    assert all(isinstance(f, SampledFrame) for f in frames)
