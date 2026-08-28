"""
Evaluation metrics: segment-level IoU, boundary error, matching, and
precision/recall/F1 built on top of that matching.

Ground truth is read from ground_truth.json (same schema as the detection
output, produced by the CANDIDATE by hand — never invented here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Segment:
    start_s: float
    end_s: float
    ad_type: str = ""
    brand: Optional[str] = None
    id: str = ""


def iou(a: Segment, b: Segment) -> float:
    """Intersection-over-union of two time intervals, in [0, 1]."""
    inter_start = max(a.start_s, b.start_s)
    inter_end = min(a.end_s, b.end_s)
    intersection = max(0.0, inter_end - inter_start)
    union = (a.end_s - a.start_s) + (b.end_s - b.start_s) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def boundary_error(a: Segment, b: Segment) -> Tuple[float, float]:
    """(start_error_s, end_error_s) — absolute differences in boundary placement."""
    return abs(a.start_s - b.start_s), abs(a.end_s - b.end_s)


@dataclass
class MatchResult:
    pred: Optional[Segment]
    gt: Optional[Segment]
    iou_score: float
    start_error_s: Optional[float]
    end_error_s: Optional[float]
    match_type: str  # "tp" | "fp" | "fn"


def match_segments(predicted: List[Segment], ground_truth: List[Segment], iou_threshold: float = 0.3) -> List[MatchResult]:
    """
    Greedy one-to-one matching: for each ground-truth segment, find the
    predicted segment with highest IoU above iou_threshold that hasn't
    already been claimed. Unmatched GT -> false negative. Unmatched
    predictions -> false positive.

    iou_threshold=0.3 is deliberately lenient: at the segment-detection
    stage we care more about "did we find roughly the right region" than
    pixel-perfect boundaries, which boundary_error reports on separately.
    """
    results: List[MatchResult] = []
    used_pred_idx = set()

    for gt in ground_truth:
        best_idx, best_iou = None, 0.0
        for i, pred in enumerate(predicted):
            if i in used_pred_idx:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= iou_threshold:
            used_pred_idx.add(best_idx)
            pred = predicted[best_idx]
            se, ee = boundary_error(pred, gt)
            results.append(MatchResult(pred=pred, gt=gt, iou_score=best_iou, start_error_s=se, end_error_s=ee, match_type="tp"))
        else:
            results.append(MatchResult(pred=None, gt=gt, iou_score=0.0, start_error_s=None, end_error_s=None, match_type="fn"))

    for i, pred in enumerate(predicted):
        if i not in used_pred_idx:
            results.append(MatchResult(pred=pred, gt=None, iou_score=0.0, start_error_s=None, end_error_s=None, match_type="fp"))

    return results


@dataclass
class VideoMetrics:
    precision: float
    recall: float
    f1: float
    mean_iou_tp: float
    mean_start_error_s: float
    mean_end_error_s: float
    n_tp: int
    n_fp: int
    n_fn: int


def compute_video_metrics(matches: List[MatchResult]) -> VideoMetrics:
    tp = [m for m in matches if m.match_type == "tp"]
    fp = [m for m in matches if m.match_type == "fp"]
    fn = [m for m in matches if m.match_type == "fn"]

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    mean_iou = sum(m.iou_score for m in tp) / len(tp) if tp else 0.0
    mean_start_err = sum(m.start_error_s for m in tp) / len(tp) if tp else 0.0
    mean_end_err = sum(m.end_error_s for m in tp) / len(tp) if tp else 0.0

    return VideoMetrics(
        precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3),
        mean_iou_tp=round(mean_iou, 3), mean_start_error_s=round(mean_start_err, 2),
        mean_end_error_s=round(mean_end_err, 2), n_tp=len(tp), n_fp=len(fp), n_fn=len(fn),
    )


def segments_from_json_list(raw_segments: List[dict]) -> List[Segment]:
    return [
        Segment(
            start_s=float(s["start_s"]), end_s=float(s["end_s"]),
            ad_type=s.get("ad_type", ""), brand=s.get("brand"), id=s.get("id", ""),
        )
        for s in raw_segments
    ]
