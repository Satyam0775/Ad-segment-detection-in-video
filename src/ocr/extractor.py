"""
OCR over sampled frames.

Abstracted behind OCRProvider so the implementation can be swapped (e.g. for
a cloud OCR API) without touching detection code. EasyOCR is the default: it
is pure-Python-installable, works reasonably on Windows/Python 3.12, and
needs no external system binary the way Tesseract does.

We do NOT rely purely on a fixed keyword list to decide "is this an ad" —
that judgment lives in detection/rules.py, which looks at OCR text alongside
ASR and VLM signals. This module's job is only: extract text + timestamp.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.config import get_settings
from src.logging_setup import get_logger
from src.media.frames import SampledFrame

logger = get_logger(__name__)


class OCRError(RuntimeError):
    pass


@dataclass
class OCRResult:
    timestamp_s: float
    text: str
    raw_boxes: int  # number of text boxes detected, useful as a weak density signal


class OCRProvider(ABC):
    @abstractmethod
    def read(self, image_path: Path) -> tuple[str, int]:
        """Return (joined_text, num_boxes)."""


class EasyOCRProvider(OCRProvider):
    def __init__(self) -> None:
        self._reader = None

    def _load(self):
        if self._reader is not None:
            return self._reader
        settings = get_settings()
        try:
            import easyocr
        except ImportError as e:
            raise OCRError("easyocr is not installed. Run: pip install easyocr") from e
        logger.info("Loading EasyOCR reader for languages=%s", settings.ocr_language_list)
        self._reader = easyocr.Reader(settings.ocr_language_list, gpu=False)
        return self._reader

    def read(self, image_path: Path) -> tuple[str, int]:
        reader = self._load()
        results = reader.readtext(str(image_path))
        texts = [r[1] for r in results]
        return " ".join(texts).strip(), len(results)


class NullOCRProvider(OCRProvider):
    """Used in tests / environments without OCR deps installed."""

    def read(self, image_path: Path) -> tuple[str, int]:
        return "", 0


def get_ocr_provider() -> OCRProvider:
    settings = get_settings()
    if settings.OCR_PROVIDER == "easyocr":
        return EasyOCRProvider()
    if settings.OCR_PROVIDER == "null":
        return NullOCRProvider()
    raise OCRError(f"Unknown OCR_PROVIDER: {settings.OCR_PROVIDER}")


def run_ocr_on_frames(frames: List[SampledFrame], provider: OCRProvider | None = None) -> List[OCRResult]:
    provider = provider or get_ocr_provider()
    results: List[OCRResult] = []
    for frame in frames:
        try:
            text, n_boxes = provider.read(frame.path)
        except OCRError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR failed on frame %s: %s", frame.path, e)
            text, n_boxes = "", 0
        results.append(OCRResult(timestamp_s=frame.timestamp_s, text=text, raw_boxes=n_boxes))
    logger.info("OCR complete: %d frames processed", len(results))
    return results


def ocr_results_to_json(results: List[OCRResult]) -> list[dict]:
    return [
        {
            "timestamp_s": float(r.timestamp_s),
            "text": r.text,
            "raw_boxes": int(r.raw_boxes),
        }
        for r in results
    ]


def save_ocr_results(results: List[OCRResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ocr_results_to_json(results), indent=2, ensure_ascii=False), encoding="utf-8")


def load_ocr_results(path: Path) -> List[OCRResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise OCRError(f"OCR JSON must be a list: {path}")
    return [
        OCRResult(
            timestamp_s=float(x["timestamp_s"]),
            text=str(x.get("text", "")),
            raw_boxes=int(x.get("raw_boxes", 0)),
        )
        for x in data
    ]
