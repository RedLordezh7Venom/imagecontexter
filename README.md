# ImageContexter

> Classify images into category folders using a **local** Vision Language Model — 100% free, runs entirely offline on your machine.

ImageContexter uses [Moondream](https://moondream.ai/), a lightweight VLM that runs on consumer hardware (even CPU-only), to automatically sort your images into user-defined categories.

## Features

- **Zero cost** — everything runs locally, no API keys needed
- **Simple YAML config** — define your categories with plain-English descriptions
- **Two model sizes** — 0.5B (fast, ~1 GB) or 2B (accurate, ~1.5 GB)
- **Copy or Move** — non-destructive by default (copies), with move option
- **Dry-run mode** — preview classifications before committing
- **Resume support** — interrupt and restart without re-processing
- **Detailed reports** — CSV and JSON output with every classification

## Quick Start

### 1. Install

```bash
cd imagecontexter
pip install -e .
```

> **First run** will download the Moondream model (~1.5 GB for 2B, ~0.5 GB for 0.5B). This only happens once.

### 2. Define Your Categories

Create a `categories.yaml` (or copy `categories.example.yaml`):

```yaml
categories:
  - name: landscapes
    description: "Nature scenes — mountains, beaches, forests, sunsets"

  - name: people
    description: "Photos with people — portraits, selfies, group shots"

  - name: screenshots
    description: "Screen captures, UI mockups, text-heavy images"

  - name: other
    description: "Anything that doesn't fit above"
```

> **Tip:** The more specific your descriptions, the better the classification accuracy.

### 3. Classify

```bash
# Basic usage — copies images into ./my-photos_classified/<category>/
imagecontexter classify ./my-photos -c categories.yaml

# Preview first (no files touched)
imagecontexter classify ./my-photos -c categories.yaml --dry-run

# Use the smaller, faster model
imagecontexter classify ./my-photos -c categories.yaml --model-size 0.5b

# Move instead of copy (saves disk space)
imagecontexter classify ./my-photos -c categories.yaml --mode move

# Scan subdirectories too
imagecontexter classify ./my-photos -c categories.yaml --recursive

# Custom output location
imagecontexter classify ./my-photos -c categories.yaml -o ./sorted-photos

# Generate captions too (saved in the report)
imagecontexter classify ./my-photos -c categories.yaml --describe

# Resume an interrupted run
imagecontexter classify ./my-photos -c categories.yaml --resume
```

## Output Structure

```
my-photos_classified/
├── landscapes/
│   ├── sunset_beach.jpg
│   └── mountain_view.png
├── people/
│   ├── family_dinner.jpg
│   └── selfie.png
├── screenshots/
│   └── error_msg.png
├── other/
│   └── abstract_pattern.jpg
├── report.csv
└── report.json
```

## Hardware Requirements

| Model  | RAM / VRAM | Speed (per image) | Accuracy  |
| ------ | ---------- | ------------------ | --------- |
| **0.5B** | ~1 GB    | ~2-5s (GPU), ~10-20s (CPU) | Good      |
| **2B**   | ~3 GB    | ~3-8s (GPU), ~15-40s (CPU) | Very Good |

- **GPU recommended** for large batches but CPU works fine
- Supports NVIDIA (CUDA) and Apple Silicon (MPS)

## Categories YAML Reference

```yaml
categories:
  # Full format (recommended)
  - name: category_name
    description: "Detailed description of what belongs here"

  # Shorthand (name only — less accurate)
  - simple_category
```

## CLI Reference

```
imagecontexter classify INPUT_DIR [OPTIONS]

Options:
  -c, --categories PATH    Path to categories YAML file  [required]
  -o, --output PATH        Output directory  [default: <input>_classified]
  --mode [copy|move]       File operation mode  [default: copy]
  --dry-run                Preview without touching files
  --model-size [0.5b|2b]   Moondream model size  [default: 2b]
  --describe               Generate captions (slower, saved in report)
  -r, --recursive          Scan input directory recursively
  --report [csv|json|both] Report format  [default: both]
  --resume                 Skip already-classified images
  --help                   Show this message and exit
```

## License

MIT
