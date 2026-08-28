"""
ASR: timestamped transcription using faster-whisper.

The output is a list of TranscriptSegment(start, end, text) — never a single
flattened string — so downstream detection can query "what was said between
t1 and t2" for any candidate window (assignment requirement, Section 10).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import get_settings
from src.logging_setup import get_logger

logger = get_logger(__name__)


class ASRError(RuntimeError):
    pass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class Transcriber:
    """Thin wrapper around faster-whisper. Model is lazy-loaded and cached
    on the instance so repeated calls in a process don't reload weights."""

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        settings = get_settings()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ASRError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from e
        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            settings.WHISPER_MODEL, settings.WHISPER_DEVICE, settings.WHISPER_COMPUTE_TYPE,
        )
        self._model = WhisperModel(
            settings.WHISPER_MODEL, device=settings.WHISPER_DEVICE, compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        return self._model

    def transcribe(self, audio_path: Path) -> List[TranscriptSegment]:
        if not audio_path.exists():
            raise ASRError(f"Audio file does not exist: {audio_path}")
        settings = get_settings()
        language = (settings.ASR_LANGUAGE or "").strip() or None
        model = self._load_model()
        logger.info("Transcribing %s", audio_path)
        try:
            segments_iter, info = model.transcribe(
                str(audio_path), language=language, vad_filter=True,
            )
            segments = [
                TranscriptSegment(start=float(s.start), end=float(s.end), text=s.text.strip())
                for s in segments_iter
            ]
        except ValueError as e:
            if "empty" in str(e).lower() or "max()" in str(e).lower():
                logger.warning("ASR generator returned an empty sequence or VAD issue: %s. Continuing with empty transcript.", e)
                segments = []
            else:
                raise ASRError(f"Whisper transcription failed for {audio_path}: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise ASRError(f"Whisper transcription failed for {audio_path}: {e}") from e

        logger.info("Transcribed %d segments (detected language=%s)", len(segments), getattr(info, "language", "?"))
        if not segments:
            logger.warning("Transcription produced zero segments for %s (silent audio or ASR miss)", audio_path)
        return segments


def query_transcript_window(segments: List[TranscriptSegment], start_s: float, end_s: float) -> str:
    """Concatenate transcript text overlapping [start_s, end_s]. Used by
    detection to fetch the transcript_span evidence field for a candidate."""
    parts = [s.text for s in segments if s.end >= start_s and s.start <= end_s]
    return " ".join(parts).strip()


def transcript_segments_to_json(segments: List[TranscriptSegment]) -> list[dict]:
    return [
        {
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
        }
        for s in segments
    ]


def save_transcript_segments(segments: List[TranscriptSegment], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(transcript_segments_to_json(segments), indent=2, ensure_ascii=False), encoding="utf-8")


def load_transcript_segments(path: Path) -> List[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ASRError(f"Transcript JSON must be a list: {path}")
    return [
        TranscriptSegment(start=float(x["start"]), end=float(x["end"]), text=str(x["text"]))
        for x in data
    ]
