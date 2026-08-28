"""
Run downstream detection (VLM + fusion + segmentation) from precomputed assets.

This script intentionally skips acquisition, frame extraction, scene detection,
and OCR generation. It expects those outputs to already exist on disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition.downloader import AcquiredMedia, ffprobe_duration  # noqa: E402
from src.asr.transcriber import load_transcript_segments  # noqa: E402
from src.media.frames import load_sampled_frames_from_dir  # noqa: E402
from src.media.scenes import load_scene_cuts  # noqa: E402
from src.ocr.extractor import load_ocr_results  # noqa: E402
from src.pipeline import run_pipeline_from_precomputed  # noqa: E402
from src.schemas.response import Kind, Platform  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream detection from precomputed Video assets.")
    parser.add_argument("--source-url", required=True, help="Original source URL (for response.source.url)")
    parser.add_argument("--video-path", required=True, help="Local video path already downloaded")
    parser.add_argument("--frames-dir", required=True, help="Directory containing precomputed frame_*.jpg files")
    parser.add_argument("--transcript-json", required=True, help="Precomputed transcript JSON list: [{start,end,text}, ...]")
    parser.add_argument("--ocr-json", required=True, help="Precomputed OCR JSON list: [{timestamp_s,text,raw_boxes}, ...]")
    parser.add_argument("--scene-cuts-json", required=True, help="Precomputed scene cuts JSON list or {scene_cuts:[...]} ")
    parser.add_argument("--kind", choices=["vod", "short", "live"], default="live", help="Source kind")
    parser.add_argument("--platform", choices=["youtube", "instagram", "file", "other"], default="youtube")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional duration override")
    parser.add_argument("--out", default="artifacts/video3_livestream_or_recording.json", help="Output JSON path")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    frames_dir = Path(args.frames_dir)
    transcript_path = Path(args.transcript_json)
    ocr_path = Path(args.ocr_json)
    scene_cuts_path = Path(args.scene_cuts_json)

    for p in [video_path, frames_dir, transcript_path, ocr_path, scene_cuts_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input path does not exist: {p}")

    frames = load_sampled_frames_from_dir(frames_dir)
    transcript_segments = load_transcript_segments(transcript_path)
    ocr_results = load_ocr_results(ocr_path)
    scene_cuts = load_scene_cuts(scene_cuts_path)

    duration_s = args.duration_s if args.duration_s is not None else ffprobe_duration(video_path)
    media = AcquiredMedia(
        local_path=video_path,
        duration_s=float(duration_s),
        platform=Platform(args.platform),
        kind=Kind(args.kind),
        title=video_path.stem,
        is_live=(args.kind == "live"),
    )

    response = run_pipeline_from_precomputed(
        source_url=args.source_url,
        media=media,
        transcript_segments=transcript_segments,
        frames=frames,
        scene_cuts=scene_cuts,
        ocr_results=ocr_results,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    print(f"WROTE: {out_path}")
    print(f"segments={len(response.segments)} frames_sampled={response.stats.frames_sampled} model_calls={response.stats.model_calls}")


if __name__ == "__main__":
    main()
