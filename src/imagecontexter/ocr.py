"""Fast OCR pre-filter for specific keyword-driven categories.

Checks for:
  - youtube: "ago", "subscribers", "views", "comment"
  - spotify: "spotify", "album", "playlist", "songs", "queue"

Runs before the main VLM classifier. If a target keyword is found,
it returns the category immediately in ~150-250ms, skipping VLM inference.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Keywords for YouTube and Spotify detection
YOUTUBE_KEYWORDS: tuple[str, ...] = ("ago", "subscribers", "views", "comment")
SPOTIFY_KEYWORDS: tuple[str, ...] = ("spotify", "album", "playlist", "songs", "queue")


class FastOCRClassifier:
    """Ultra-fast ONNX-based OCR pre-filter."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
            self._available = True
        except ImportError:
            logger.warning(
                "rapidocr_onnxruntime not found. OCR pre-filtering will be disabled."
            )
            self._available = False

    def check(self, image_path: Path) -> Optional[tuple[str, str]]:
        """Run OCR on the image and check for target category keywords.

        Returns (category, matched_keyword_info) if matched, or None.
        """
        if not self._available:
            return None

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                # Downscale extremely large images for OCR speedup
                max_dim = 1280
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
                img_array = np.array(img)

            ocr_results, _ = self.ocr(img_array)
            if not ocr_results:
                return None

            # Collect and lowercase all recognized text
            extracted_text = " ".join([str(line[1]).lower() for line in ocr_results])

            # 1. Check YouTube keywords: "ago", "subscribers", "views", "comment"
            yt_matches = [
                kw for kw in YOUTUBE_KEYWORDS
                if kw in extracted_text
            ]
            if yt_matches:
                return ("youtube", f"OCR matched: {', '.join(yt_matches)}")

            # 2. Check Spotify keywords: "spotify", "album", "playlist", "songs", "queue"
            sp_matches = [
                kw for kw in SPOTIFY_KEYWORDS
                if kw in extracted_text
            ]
            if sp_matches:
                return ("spotify", f"OCR matched: {', '.join(sp_matches)}")

            return None

        except Exception as exc:
            logger.debug(f"OCR check failed for {image_path.name}: {exc}")
            return None
