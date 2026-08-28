"""
Run evaluation: compare predicted output JSON files against ground_truth.json
and print/save per-video and summary metrics.

Usage (Windows CMD/PowerShell, from project root, venv activated):
    python scripts\\run_evaluation.py

Expects:
    ground_truth/ground_truth.json   -- your hand labels, schema documented in that folder's README
    artifacts/<video_id>.json        -- pipeline output for each of the five videos (from run_pipeline.py)

This script does NOT invent labels or scores. If ground_truth.json or a
prediction file is missing for a video, that video is reported as PENDING.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_video_metrics, match_segments, segments_from_json_list  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = PROJECT_ROOT / "ground_truth" / "ground_truth.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_ground_truth() -> Dict[str, List[dict]]:
    if not GROUND_TRUTH_PATH.exists():
        print(f"PENDING: {GROUND_TRUTH_PATH} does not exist yet. See ground_truth/README.md.")
        return {}
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def load_prediction(video_id: str) -> List[dict] | None:
    path = ARTIFACTS_DIR / f"{video_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", [])


def main() -> None:
    gt_by_video = load_ground_truth()
    if not gt_by_video:
        print(
            "\nNo ground truth found. This is expected until you have watched the five official "
            "videos and filled in ground_truth/ground_truth.json yourself.\n"
            "Nothing further to evaluate — this run is intentionally a no-op."
        )
        return

    all_rows = []
    for video_id, gt_segments_raw in gt_by_video.items():
        pred_raw = load_prediction(video_id)
        if pred_raw is None:
            print(f"{video_id}: PENDING — no prediction file at artifacts/{video_id}.json yet")
            continue

        gt_segments = segments_from_json_list(gt_segments_raw)
        pred_segments = segments_from_json_list(pred_raw)
        matches = match_segments(pred_segments, gt_segments)
        metrics = compute_video_metrics(matches)

        print(f"\n=== {video_id} ===")
        print(f"  Precision: {metrics.precision}  Recall: {metrics.recall}  F1: {metrics.f1}")
        print(f"  Mean IoU (TP): {metrics.mean_iou_tp}  Mean start err (s): {metrics.mean_start_error_s}  Mean end err (s): {metrics.mean_end_error_s}")
        print(f"  TP={metrics.n_tp} FP={metrics.n_fp} FN={metrics.n_fn}")
        all_rows.append((video_id, metrics))

    if all_rows:
        avg_f1 = sum(m.f1 for _, m in all_rows) / len(all_rows)
        print(f"\n=== SUMMARY across {len(all_rows)} evaluated videos ===")
        print(f"Mean F1: {round(avg_f1, 3)}")
    else:
        print("\nNo videos were evaluated (all pending). Run scripts/run_pipeline.py for each test video first.")


if __name__ == "__main__":
    main()
