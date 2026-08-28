"""
Temporal segmentation: group fused candidate points into coherent ad
segments with start_s/end_s.

Merge rule: points above DETECTION_CONFIDENCE_THRESHOLD are grouped if the
gap between consecutive points is <= SEGMENT_MERGE_GAP_S. This directly
answers ambiguity item 8 (two sponsors back-to-back): if the gap is small
AND the brand/context clearly changes between the two clusters, we split
them into two segments instead of merging (BACK_TO_BACK_SPLIT_GAP_S,
smaller than the general merge gap) — see DESIGN.md for the full ruling.

Ambiguity item 7 (a Reel that is an ad from frame one to the last frame):
our default is start_s=0.0, end_s=duration_s — the whole clip is one
segment. REVIEW this against your own labeling before submitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import get_settings
from src.detection.classifier import classify_segment
from src.detection.fusion import FusedPoint
from src.schemas.response import AdSegment, AdType, Evidence, SignalName


@dataclass
class RawCluster:
    points: List[FusedPoint]

    @property
    def start_s(self) -> float:
        return min(p.timestamp_s for p in self.points)

    @property
    def end_s(self) -> float:
        return max(p.timestamp_s for p in self.points)


def cluster_points(points: List[FusedPoint], threshold: float, merge_gap: float, split_gap: float) -> List[RawCluster]:
    above = [p for p in points if p.confidence >= threshold]
    above.sort(key=lambda p: p.timestamp_s)
    if not above:
        return []

    clusters: List[List[FusedPoint]] = [[above[0]]]
    for p in above[1:]:
        prev = clusters[-1][-1]
        gap = p.timestamp_s - prev.timestamp_s
        brand_changed = (
            p.brand_guess and prev.brand_guess and p.brand_guess != prev.brand_guess
        )
        if gap <= split_gap and brand_changed:
            # ambiguity item 8: back-to-back, different brand -> split even though close
            clusters.append([p])
        elif gap <= merge_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [RawCluster(points=c) for c in clusters]


def build_segments(
    points: List[FusedPoint],
    video_duration_s: float,
    frames_sampled: int,
) -> List[AdSegment]:
    settings = get_settings()
    merge_gap = settings.SEGMENT_MERGE_GAP_S
    if merge_gap <= 3.0:
        merge_gap = 15.0

    clusters = cluster_points(
        points,
        threshold=settings.DETECTION_CONFIDENCE_THRESHOLD,
        merge_gap=merge_gap,
        split_gap=settings.BACK_TO_BACK_SPLIT_GAP_S,
    )

    segments: List[AdSegment] = []
    for i, cluster in enumerate(clusters, start=1):
        classified = classify_segment(
            cluster.points, position_in_video=cluster.start_s / max(video_duration_s, 0.001),
            video_duration=video_duration_s,
        )
        start_s = max(0.0, cluster.start_s)
        end_s = cluster.end_s + max(settings.MIN_SEGMENT_DURATION_S, 0.1)
        end_s = min(end_s, video_duration_s) if video_duration_s > 0 else end_s
        if end_s <= start_s:
            end_s = start_s + settings.MIN_SEGMENT_DURATION_S

        signals: List[SignalName] = []
        for p in cluster.points:
            for s in p.signals_used:
                sig = SignalName(s)
                if sig not in signals:
                    signals.append(sig)

        transcript_span = " ".join(p.transcript_span for p in cluster.points if p.transcript_span).strip() or None

        segments.append(
            AdSegment(
                id=f"seg_{i:02d}",
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                ad_type=classified.ad_type,
                confidence=classified.confidence,
                brand=classified.brand,
                description=classified.description,
                evidence=Evidence(
                    frame_timestamps=[round(p.timestamp_s, 2) for p in cluster.points],
                    transcript_span=transcript_span,
                    signals_used=signals,
                ),
            )
        )
    return segments
