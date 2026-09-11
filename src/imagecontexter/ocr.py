"""Fast OCR pre-filter for specific keyword-driven categories.

Checks for:
  - youtube: "ago", "subscribers", "views", "comment"
  - spotify: "spotify", "album", "playlist", "songs", "queue"
  - coding: code syntax like "import", "def", "class", "return", "function", etc.

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

# Keywords for Coding syntax detection
CODING_KEYWORDS: tuple[str, ...] = (
    "import",
    "from",
    "def",
    "class",
    "return",
    "function",
    "const",
    "let",
    "var",
    "console.log",
    "public static",
    "#include",
    "select",
    "where",
    "printf",
    "println",
)


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

            raw_lines = [str(line[1]) for line in ocr_results]
            extracted_text = " ".join([l.lower() for l in raw_lines])

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

            # 3. Check Coding syntax
            code_matches = []
            for kw in CODING_KEYWORDS:
                pattern = r"(?:^|\W)" + re.escape(kw) + r"(?:\W|$)"
                if re.search(pattern, extracted_text, re.IGNORECASE):
                    code_matches.append(kw)
                elif kw in extracted_text:
                    code_matches.append(kw)

            unique_code_matches = list(set(code_matches))
            if len(unique_code_matches) >= 2:
                return ("coding", f"OCR matched code syntax: {', '.join(unique_code_matches)}")

            return None

        except Exception as exc:
            logger.debug(f"OCR check failed for {image_path.name}: {exc}")
            return None
