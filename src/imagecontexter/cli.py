"""Command-line interface for ImageContexter.

Usage::

    imagecontexter classify ./photos --categories categories.yaml
    imagecontexter classify ./photos -c cats.yaml --dry-run
    imagecontexter classify ./photos -c cats.yaml --model-size 0.5b --mode move
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click
from tqdm import tqdm

from .classifier import ClassificationResult, ImageClassifier, collect_images
from .config import ClassifyConfig
from .organizer import (
    load_existing_results,
    organize_file,
    write_report_csv,
    write_report_json,
)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="imagecontexter")
def main() -> None:
    """ImageContexter — classify images into category folders using a local VLM."""


# ---------------------------------------------------------------------------
# classify command
# ---------------------------------------------------------------------------

@main.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--categories", "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a categories YAML file.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory.  [default: <input_dir>_classified]",
)
@click.option(
    "--mode",
    type=click.Choice(["copy", "move"], case_sensitive=False),
    default="copy",
    show_default=True,
    help="Copy or move images into category folders.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without touching any files.",
)
@click.option(
    "--model-size",
    type=click.Choice(["0.5b", "2b"], case_sensitive=False),
    default="2b",
    show_default=True,
    help="Moondream model size.  0.5b = fast / low RAM,  2b = more accurate.",
)
@click.option(
    "--describe",
    is_flag=True,
    help="Run an extra captioning pass per image (slower, but saves a "
         "description in the report).",
)
@click.option(
    "--recursive", "-r",
    is_flag=True,
    help="Scan the input directory recursively.",
)
@click.option(
    "--report",
    type=click.Choice(["csv", "json", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Report format written to the output directory.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Skip images that appear in an existing report (resume an "
         "interrupted run).",
)
def classify(
    input_dir: Path,
    categories: Path,
    output: Path | None,
    mode: str,
    dry_run: bool,
    model_size: str,
    describe: bool,
    recursive: bool,
    report: str,
    resume: bool,
) -> None:
    """Classify images in INPUT_DIR into category sub-folders."""

    # -- load categories ----------------------------------------------------
    config = ClassifyConfig.from_yaml(categories)
    cat_names = ", ".join(c.name for c in config.categories)
    click.echo(f"📋 Categories: {cat_names}")

    # -- discover images ----------------------------------------------------
    images = collect_images(input_dir, recursive=recursive)
    if not images:
        click.secho("❌ No images found in the input directory.", fg="red", err=True)
        sys.exit(1)
    click.echo(f"🖼️  Found {len(images)} image(s)")

    # -- resolve output dir -------------------------------------------------
    if output is None:
        output = input_dir.parent / f"{input_dir.name}_classified"

    # -- resume support -----------------------------------------------------
    if resume and not dry_run and output.exists():
        already_done = load_existing_results(output)
        before = len(images)
        images = [img for img in images if str(img) not in already_done]
        skipped = before - len(images)
        if skipped:
            click.echo(f"⏩ Skipping {skipped} already-classified image(s)")

    if not images:
        click.secho("✅ All images already classified!", fg="green")
        return

    # -- load model ---------------------------------------------------------
    classifier: ImageClassifier | None = None
    if not dry_run:
        dl_size = "~0.5 GB" if model_size == "0.5b" else "~1.5 GB"
        click.echo(
            f"🧠 Loading Moondream {model_size} model "
            f"(first run will download {dl_size}) …"
        )
        try:
            classifier = ImageClassifier(model_size=model_size)
        except Exception as exc:
            click.secho(f"❌ Failed to load model: {exc}", fg="red", err=True)
            click.echo(
                "💡 Try --model-size 0.5b for lower memory usage, or check "
                "that you have enough free RAM / VRAM.",
                err=True,
            )
            sys.exit(1)
        click.secho("✅ Model loaded!", fg="green")

    # -- classify -----------------------------------------------------------
    results: list[ClassificationResult] = []
    errors: list[tuple[str, str]] = []

    with tqdm(images, desc="Classifying", unit="img", dynamic_ncols=True) as pbar:
        for image_path in pbar:
            pbar.set_postfix_str(image_path.name[:30], refresh=False)
            try:
                if dry_run:
                    result = ClassificationResult(
                        image_path=str(image_path),
                        category="[dry-run]",
                        description=None,
                        raw_answer="[dry-run]",
                    )
                else:
                    assert classifier is not None
                    result = classifier.classify(
                        image_path,
                        config.categories,
                        describe=describe,
                    )
                    organize_file(result, output, mode=mode)

                results.append(result)

            except Exception as exc:  # noqa: BLE001
                errors.append((str(image_path), str(exc)))
                tqdm.write(f"⚠️  Error processing {image_path.name}: {exc}")

    # -- write reports ------------------------------------------------------
    if not dry_run and results:
        output.mkdir(parents=True, exist_ok=True)
        if report in ("csv", "both"):
            write_report_csv(results, output / "report.csv")
        if report in ("json", "both"):
            write_report_json(results, output / "report.json")

    # -- summary ------------------------------------------------------------
    _print_summary(results, errors, output, dry_run)


# ---------------------------------------------------------------------------
# summary helper
# ---------------------------------------------------------------------------

def _print_summary(
    results: list[ClassificationResult],
    errors: list[tuple[str, str]],
    output: Path,
    dry_run: bool,
) -> None:
    click.echo()
    click.echo("═" * 52)
    click.echo("  📊  Classification Summary")
    click.echo("═" * 52)

    counts = Counter(r.category for r in results)
    max_count = max(counts.values(), default=1)

    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar_len = int((count / max_count) * 30)
        bar = "█" * bar_len
        click.echo(f"  {cat:20s} │ {count:5d} │ {bar}")

    click.echo("─" * 52)
    click.echo(f"  ✅ Classified : {len(results)}")
    if errors:
        click.secho(f"  ⚠️  Errors     : {len(errors)}", fg="yellow")
    if dry_run:
        click.echo("  ℹ️  Dry run — no files were moved / copied")
    else:
        click.echo(f"  📁 Output     : {output}")
    click.echo()
