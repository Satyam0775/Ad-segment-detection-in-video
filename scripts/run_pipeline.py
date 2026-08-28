"""
Run the detection pipeline on a single video URL/path from the command line.

Usage (Windows CMD/PowerShell, from project root, with venv activated):
    python scripts\\run_pipeline.py "https://www.youtube.com/watch?v=ujFWRFYLGjY" --out artifacts\\video1.json
    python scripts\\run_pipeline.py "C:\\path\\to\\reel.mp4" --out artifacts\\reel1.json --kind short
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensure_dirs  # noqa: E402
from src.logging_setup import configure_logging, get_logger  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.schemas.response import Kind  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ad segment detection on a single video.")
    parser.add_argument("url", help="Video URL (YouTube) or local file path (Instagram Reel / live fallback recording)")
    parser.add_argument("--out", default="artifacts/result.json", help="Output JSON path")
    parser.add_argument("--kind", choices=["vod", "short", "live"], default=None, help="Override kind detection")
    parser.add_argument("--live-max-seconds", type=float, default=None, help="Cap capture duration for live URLs")
    args = parser.parse_args()

    configure_logging("INFO")
    ensure_dirs()

    kind_hint = Kind(args.kind) if args.kind else None
    result = run_pipeline(args.url, kind_hint=kind_hint, live_max_seconds=args.live_max_seconds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.response.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Wrote result to %s", out_path)
    print(json.dumps(result.response.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    main()
