"""Core image classification engine using Moondream VLM.

Handles model loading, image processing, and the classification prompt
strategy.  The classifier works in one or two stages:

  1-stage (default):  Ask the VLM to pick a category directly.
  2-stage (--describe): Caption the image first, then classify.
     Slower, but the caption is saved in the report for auditing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Category

logger = logging.getLogger(__name__)

# Image file extensions we recognise (case-insensitive check at call site).
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Outcome of classifying a single image."""

    image_path: str
    category: str
    description: str | None  # populated only when --describe is used
    raw_answer: str           # the literal text the VLM returned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(result: object, key: str) -> str:
    """Pull text out of a Moondream result regardless of its shape.

    The ``moondream`` library has changed its return types across versions
    (plain str, dict, dataclass).  This helper tries them all.
    """
    if isinstance(result, str):
        return result
    if hasattr(result, key):
        return str(getattr(result, key))
    if isinstance(result, dict) and key in result:
        return str(result[key])
    return str(result)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ImageClassifier:
    """Classify images into user-defined categories via Moondream."""

    MODEL_MAP: dict[str, str] = {
        "0.5b": "moondream-0.5b-int8",
        "2b": "moondream-2b-int8",
    }

    def __init__(self, model_size: str = "2b") -> None:
        if model_size not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown model size '{model_size}'. "
                f"Choose from: {', '.join(self.MODEL_MAP)}"
            )

        # Import here so the rest of the package can be used (config,
        # organizer, …) without having moondream installed yet.
        import moondream as md           # noqa: F811

        model_id = self.MODEL_MAP[model_size]
        logger.info("Loading Moondream model %s …", model_id)
        self._md = md
        self.model = md.VL(model=model_id, local=True)

    # ---- public API -------------------------------------------------------

    def classify(
        self,
        image_path: Path,
        categories: list[Category],
        *,
        describe: bool = False,
    ) -> ClassificationResult:
        """Classify a single image.

        Parameters
        ----------
        image_path:
            Path to the image file.
        categories:
            The set of target categories (from the YAML config).
        describe:
            If *True*, run an extra caption pass and store the description
            in the result (slower but useful for auditing).
        """
        image = self._md.load_image(str(image_path))

        # --- optional stage 1: describe ------------------------------------
        description: str | None = None
        if describe:
            cap_result = self.model.caption(image)
            description = _extract_text(cap_result, "caption")

        # --- stage 2 (or only stage): classify -----------------------------
        prompt = self._build_prompt(categories)
        query_result = self.model.query(image, prompt)
        raw_answer = _extract_text(query_result, "answer").strip()

        matched = self._match_category(raw_answer, categories)

        return ClassificationResult(
            image_path=str(image_path),
            category=matched,
            description=description,
            raw_answer=raw_answer,
        )

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _build_prompt(categories: list[Category]) -> str:
        """Construct a classification prompt for the VLM."""
        lines = []
        for cat in categories:
            if cat.description:
                lines.append(f"- {cat.name}: {cat.description}")
            else:
                lines.append(f"- {cat.name}")

        cat_block = "\n".join(lines)
        return (
            "Classify this image into exactly ONE of the following categories.\n"
            "\n"
            f"{cat_block}\n"
            "\n"
            "Reply with ONLY the category name — no punctuation, no explanation."
        )

    @staticmethod
    def _match_category(answer: str, categories: list[Category]) -> str:
        """Best-effort match of the VLM's free-text answer to a category name.

        Strategy (in order):
          1. Exact match (case-insensitive).
          2. Answer contains a category name.
          3. A category name contains the answer.
          4. Fall back to "other" if present, else "uncategorized".
        """
        answer_lower = answer.lower().strip()

        # 1 — exact
        for cat in categories:
            if cat.name.lower() == answer_lower:
                return cat.name

        # 2 — answer is a superset (e.g. "the category is landscapes")
        for cat in categories:
            if cat.name.lower() in answer_lower:
                return cat.name

        # 3 — answer is a subset (e.g. "land" → "landscapes")
        for cat in categories:
            if answer_lower in cat.name.lower() and answer_lower:
                return cat.name

        # 4 — fallback
        for cat in categories:
            if cat.name.lower() == "other":
                return cat.name

        return "uncategorized"


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def collect_images(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Return sorted image files from *directory*.

    Parameters
    ----------
    directory:
        Folder to scan.
    recursive:
        If *True*, descend into sub-directories.
    """
    pattern = "**/*" if recursive else "*"
    return sorted(
        f
        for f in directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
