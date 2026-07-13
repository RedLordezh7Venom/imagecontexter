"""File organizer and report writer.

After classification, this module handles:
  * Copying / moving images into category sub-folders.
  * Writing CSV and JSON reports.
  * Loading previous reports so a run can be resumed.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .classifier import ClassificationResult


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def organize_file(
    result: ClassificationResult,
    output_dir: Path,
    mode: str = "copy",
) -> Path:
    """Copy or move an image into its category sub-folder.

    Returns the destination path.
    """
    category_dir = output_dir / _sanitize_dirname(result.category)
    category_dir.mkdir(parents=True, exist_ok=True)

    src = Path(result.image_path)
    dest = category_dir / src.name

    # Collision handling: append _1, _2, etc. to avoid overwriting.
    # This is important when multiple source dirs have same-named files.
    if dest.exists():
        stem, suffix = src.stem, src.suffix
        counter = 1
        while dest.exists():
            dest = category_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    if mode == "move":
        shutil.move(str(src), str(dest))
    else:  # copy
        shutil.copy2(str(src), str(dest))

    return dest


def _sanitize_dirname(name: str) -> str:
    """Make a category name safe for use as a directory name."""
    return "".join(
        c if (c.isalnum() or c in "-_ ") else "_" for c in name
    ).strip()


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------

# CSV fields are written with UTF-8 encoding for Windows compatibility
# with non-ASCII characters in file paths and category names.
_CSV_FIELDS = ["image_path", "category", "description", "raw_answer"]


def write_report_csv(
    results: list[ClassificationResult],
    output_path: Path,
) -> None:
    """Write (or overwrite) a CSV report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image_path": r.image_path,
                    "category": r.category,
                    "description": r.description or "",
                    "raw_answer": r.raw_answer,
                }
            )


def write_report_json(
    results: list[ClassificationResult],
    output_path: Path,
) -> None:
    """Write (or overwrite) a JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_existing_results(output_dir: Path) -> set[str]:
    """Return the set of image paths that have already been classified.

    Scans for ``report.csv`` or ``report.json`` inside *output_dir*.
    """
    classified: set[str] = set()

    csv_path = output_dir / "report.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                classified.add(row["image_path"])

    json_path = output_dir / "report.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as fh:
            for item in json.load(fh):
                classified.add(item["image_path"])

    return classified
