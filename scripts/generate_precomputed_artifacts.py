"""
Generate precomputed artifacts from local assets (no download path):
- transcript JSON
- OCR JSON
- scene-cuts JSON

This script is intended for cases like Video 3 where acquisition was already
completed outside the normal downloader path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.transcriber import Transcriber, save_transcript_segments  # noqa: E402
from src.media.audio import extract_audio  # noqa: E402
from src.media.frames import load_sampled_frames_from_dir  # noqa: E402
from src.media.scenes import detect_scene_cuts, save_scene_cuts  # noqa: E402
from src.ocr.extractor import run_ocr_on_frames, save_ocr_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate precomputed transcript/OCR/scene-cut JSON from local files.")
    parser.add_argument("--video-path", required=True, help="Local video path")
    parser.add_argument("--frames-dir", required=True, help="Existing directory of frame_*.jpg files")
    parser.add_argument("--audio-path", default=None, help="Existing WAV path; if omitted, extract from video")
    parser.add_argument("--out-transcript", required=True, help="Output transcript JSON path")
    parser.add_argument("--out-ocr", required=True, help="Output OCR JSON path")
    parser.add_argument("--out-scene-cuts", required=True, help="Output scene-cuts JSON path")
    parser.add_argument("--max-ocr-frames", type=int, default=None, help="Optional limit for OCR rerun on existing frames")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if files already exist")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    frames_dir = Path(args.frames_dir)
    out_transcript = Path(args.out_transcript)
    out_ocr = Path(args.out_ocr)
    out_scene_cuts = Path(args.out_scene_cuts)

    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio path not found: {audio_path}")
    else:
        audio_path = video_path.with_name(f"{video_path.stem}_audio.wav")
        if not audio_path.exists():
            audio_path = extract_audio(video_path, out_path=audio_path)

    # Transcript
    if args.force or not out_transcript.exists():
        segments = Transcriber().transcribe(audio_path)
        save_transcript_segments(segments, out_transcript)
        print(f"WROTE transcript: {out_transcript} (segments={len(segments)})")
    else:
        print(f"SKIP transcript (exists): {out_transcript}")

    # Scene cuts
    if args.force or not out_scene_cuts.exists():
        scene_cuts = detect_scene_cuts(video_path)
        save_scene_cuts(scene_cuts, out_scene_cuts)
        print(f"WROTE scene cuts: {out_scene_cuts} (count={len(scene_cuts)})")
    else:
        print(f"SKIP scene cuts (exists): {out_scene_cuts}")

    # OCR from existing frames only
    if args.force or not out_ocr.exists():
        frames = load_sampled_frames_from_dir(frames_dir)
        if args.max_ocr_frames is not None:
            frames = frames[: max(0, args.max_ocr_frames)]
        ocr_results = run_ocr_on_frames(frames)
        save_ocr_results(ocr_results, out_ocr)
        print(f"WROTE ocr: {out_ocr} (results={len(ocr_results)})")
    else:
        print(f"SKIP ocr (exists): {out_ocr}")


if __name__ == "__main__":
    main()
