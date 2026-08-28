"""
Pydantic schemas implementing the Option B required JSON output contract
(assignment Section 6 / 18), plus request schemas for the FastAPI API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class Platform(str, Enum):
    youtube = "youtube"
    instagram = "instagram"
    file = "file"
    other = "other"


class Kind(str, Enum):
    vod = "vod"
    short = "short"
    live = "live"


class AdType(str, Enum):
    preroll = "preroll"
    midroll_sponsor_read = "midroll_sponsor_read"
    product_placement = "product_placement"
    self_promo = "self_promo"
    affiliate = "affiliate"
    platform_inserted = "platform_inserted"
    bumper = "bumper"
    other = "other"


class SignalName(str, Enum):
    asr = "asr"
    ocr = "ocr"
    vlm_frame = "vlm_frame"
    scene_cut = "scene_cut"


class SourceInfo(BaseModel):
    url: str
    platform: Platform
    kind: Kind
    duration_s: float = Field(ge=0)
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Evidence(BaseModel):
    frame_timestamps: List[float] = Field(default_factory=list)
    transcript_span: Optional[str] = None
    signals_used: List[SignalName] = Field(default_factory=list)


class AdSegment(BaseModel):
    id: str
    start_s: float = Field(ge=0)
    end_s: float
    ad_type: AdType
    confidence: float = Field(ge=0.0, le=1.0)
    brand: Optional[str] = None
    description: str
    evidence: Evidence

    @model_validator(mode="after")
    def check_end_after_start(self) -> "AdSegment":
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be strictly greater than start_s ({self.start_s})")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Stats(BaseModel):
    model_config = {"protected_namespaces": ()}

    wall_clock_s: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    frames_sampled: int = Field(ge=0)
    model_calls: int = Field(ge=0)


class DetectionResponse(BaseModel):
    source: SourceInfo
    segments: List[AdSegment]
    stats: Stats

    @field_validator("segments")
    @classmethod
    def segments_sorted_and_non_overlapping_warning(cls, v: List[AdSegment]) -> List[AdSegment]:
        # We don't hard-fail on overlap (a movie-trailer-inside-sponsor edge case could legitimately
        # nest), but we do enforce a stable, sorted order for consumers (viewer, evaluation).
        return sorted(v, key=lambda s: s.start_s)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    url: str = Field(..., description="Video URL (YouTube, Instagram Reel) or a local file:// / absolute path")
    kind_hint: Optional[Kind] = Field(
        default=None, description="Optional hint: vod|short|live. Auto-detected from URL when omitted."
    )
    live_max_seconds: Optional[float] = Field(
        default=None, description="For live URLs: cap capture duration for this run (e.g. testing)"
    )

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str = "0.1.0"
