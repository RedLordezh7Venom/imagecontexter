# ImageContexter

> Fast, GPU-accelerated local image categorizer using **Fast OCR pre-filtering** + **[SmolVLM-256M-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF)** on NVIDIA GPUs. 100% free, runs completely offline.

## Classification Pipeline

### Step 1: Fast OCR Pre-filter (~150-250ms)
Before loading or querying the VLM, images are quickly scanned via ONNX OCR for specific keyword patterns:
- **`youtube`**: Triggered if any of: `"ago"`, `"subscribers"`, `"views"`, `"comment"` appear.
- **`spotify`**: Triggered if any of: `"spotify"`, `"album"`, `"playlist"`, `"songs"`, `"queue"` appear.

If a match is found, the image is immediately categorized without running the VLM, saving compute and time!

### Step 2: GPU Vision Language Model (~0.5-0.8s)
If the OCR keywords are not found, **SmolVLM-256M-Instruct Q8_0 GGUF** classifies the image on your GPU into:
1. **`anime`** — Anime, manga illustrations, animated characters, 2D art
2. **`game`** — Video games, gameplay screenshots, 3D graphics, gaming UI
3. **`movie`** — Live-action films, TV series, cinematic stills, real actors
4. **`meme`** — Memes, reaction images, humorous captioned posts
5. **`coding`** — IDEs, code editors, terminal output, programming syntax

---

## Quick Start

```bash
# Classify images in a directory
imagecontexter classify ./my-photos

# Dry run preview (no files moved/copied)
imagecontexter classify ./my-photos --dry-run

# Move files instead of copying
imagecontexter classify ./my-photos --mode move

# Disable OCR pre-filter (VLM only)
imagecontexter classify ./my-photos --no-ocr

# Recursive folder scan
imagecontexter classify ./my-photos -r
```

## CLI Options

```
imagecontexter classify [OPTIONS] INPUT_DIR

Options:
  -c, --categories FILE     Optional custom categories YAML.
  -o, --output PATH         Output directory [default: <input_dir>_classified].
  --mode [copy|move]        Copy or move images into category folders.
  --dry-run                 Show classifications without touching files.
  --ocr / --no-ocr          Enable/disable Fast OCR pre-filtering [default: True].
  --gpu-layers INTEGER      Layers to offload to GPU (-1 = all). [default: -1].
  --describe                Generate concise descriptions in report.
  -r, --recursive           Scan input directory recursively.
  --report [csv|json|both]  Report format [default: both].
  --resume                  Skip images that appear in existing report.
  --help                    Show this message and exit.
```
