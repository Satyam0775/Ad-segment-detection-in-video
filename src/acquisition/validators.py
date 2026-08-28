"""
URL validation and platform/kind classification.

This module never touches the network — it is pure string/URL logic so it
is trivially unit-testable without mocks.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from src.schemas.response import Kind, Platform


class InvalidURLError(ValueError):
    """Raised when a URL is empty, malformed, or clearly unsupported."""


class UnsupportedPlatformError(ValueError):
    """Raised when a URL is well-formed but not from a platform we handle."""


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}

_YT_SHORTS_RE = re.compile(r"/shorts/")
_YT_LIVE_HINT_RE = re.compile(r"[?&]feature=live|/live/", re.IGNORECASE)


@dataclass
class ParsedSource:
    url: str
    platform: Platform
    kind: Kind
    video_id: str | None = None


def validate_and_classify(url: str, kind_hint: Kind | None = None) -> ParsedSource:
    """
    Validate a URL/path and classify it into (platform, kind).

    Supports:
      - YouTube long-form video, Shorts, and live URLs
      - Instagram Reel URLs (acquisition itself must be manual/local per the brief)
      - Local file paths / file:// URIs (kind defaults to 'vod' unless hinted)

    Raises InvalidURLError / UnsupportedPlatformError on bad input.
    """
    if not url or not url.strip():
        raise InvalidURLError("URL must not be empty")
    url = url.strip()

    # Local file support (explicit requirement: pipeline must ingest local files,
    # e.g. manually captured Reels, or a fallback recording for the live test case)
    if url.startswith("file://") or (os.path.sep in url and not url.startswith("http")):
        path = url[len("file://"):] if url.startswith("file://") else url
        if not os.path.exists(path):
            raise InvalidURLError(f"Local file does not exist: {path}")
        return ParsedSource(url=url, platform=Platform.file, kind=kind_hint or Kind.vod)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidURLError(f"Not a valid http(s) URL or existing local path: {url}")

    host = parsed.netloc.lower()

    if host in _YOUTUBE_HOSTS or host.endswith(".youtube.com"):
        if _YT_SHORTS_RE.search(parsed.path):
            kind = kind_hint or Kind.short
        elif kind_hint:
            kind = kind_hint
        else:
            # Long-form vs live is ambiguous from URL alone for youtube.com/watch?v=...
            # Live must be confirmed by the acquisition layer (yt-dlp metadata: is_live).
            # We default to vod here; downloader.py corrects this after probing metadata.
            kind = Kind.vod
        video_id = _extract_youtube_id(parsed)
        return ParsedSource(url=url, platform=Platform.youtube, kind=kind, video_id=video_id)

    if host in _INSTAGRAM_HOSTS or host.endswith(".instagram.com"):
        if "/reel/" not in parsed.path and "/reels/" not in parsed.path:
            raise UnsupportedPlatformError(
                f"Only Instagram Reel URLs are supported (got path: {parsed.path}). "
                "Per the assignment, acquire the Reel manually and ingest the local file instead."
            )
        video_id = parsed.path.strip("/").split("/")[-1] or parsed.path.strip("/").split("/")[-2]
        return ParsedSource(url=url, platform=Platform.instagram, kind=kind_hint or Kind.short, video_id=video_id)

    raise UnsupportedPlatformError(
        f"Unsupported host: {host}. Supported: YouTube, Instagram Reel URLs, or local file paths."
    )


def _extract_youtube_id(parsed) -> str | None:
    if parsed.netloc.lower() in ("youtu.be",):
        return parsed.path.strip("/") or None
    qs = parsed.query
    m = re.search(r"(?:^|&)v=([^&]+)", qs)
    if m:
        return m.group(1)
    m = re.search(r"/shorts/([^/?]+)", parsed.path)
    if m:
        return m.group(1)
    return None
