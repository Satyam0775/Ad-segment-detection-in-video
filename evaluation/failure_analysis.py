"""
Failure analysis framework.

This module provides the STRUCTURE for recording failure cases
(expected / predicted / error / root cause / failed signal / proposed fix),
per assignment Section 24. It does NOT contain any invented failure cases —
those must come from your own runs against the five real test videos once
you have executed the pipeline and compared against your hand-labeled
ground_truth.json. Populate FailureCase entries in EVAL.md, or write them
here and have generate_report() render them for EVAL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FailureCase:
    video_id: str
    expected: str  # what the ground truth says (e.g. "ad from 124.0 to 187.5, midroll_sponsor_read")
    predicted: str  # what the system produced (e.g. "no detection" or "ad from 130.0 to 145.0")
    error_description: str
    root_cause: str
    failed_signal: str  # e.g. "asr", "ocr", "vlm_frame", "scene_cut", "fusion_threshold"
    proposed_improvement: str


def render_failure_case_markdown(case: FailureCase) -> str:
    return (
        f"### {case.video_id}\n\n"
        f"- **Expected:** {case.expected}\n"
        f"- **Predicted:** {case.predicted}\n"
        f"- **What went wrong:** {case.error_description}\n"
        f"- **Root cause:** {case.root_cause}\n"
        f"- **Failed signal:** {case.failed_signal}\n"
        f"- **Proposed improvement:** {case.proposed_improvement}\n"
    )


def render_report(cases: List[FailureCase]) -> str:
    if not cases:
        return (
            "PENDING EXECUTION — no failure cases recorded yet. Run the pipeline against the five "
            "official test videos, compare against ground_truth.json, and populate at least three "
            "cases here before submission (assignment Section 24 requires this; it is weighted "
            "heavily under Evaluation quality, 25%)."
        )
    return "\n\n".join(render_failure_case_markdown(c) for c in cases)
