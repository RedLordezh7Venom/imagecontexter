# ImageContexter

> Fast, GPU-accelerated local image categorizer using [SmolVLM-256M-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF). 100% free, runs completely offline on your NVIDIA GPU.

## Hardcoded Default Categories

ImageContexter automatically classifies images into 5 categories:

1. **`anime`** — Anime, manga illustrations, animated characters, 2D art
2. **`game`** — Video games, gameplay screenshots, 3D graphics, gaming UI
3. **`movie`** — Live-action films, TV series, cinematic stills, real actors
4. **`meme`** — Memes, reaction images, humorous captioned posts
5. **`coding`** — IDEs, code editors, terminal output, programming syntax

## Features

- **Blazing Fast GPU Inference** — Powered by `llama-cpp-python` with full CUDA offloading (`~0.5s - 0.9s` per image on RTX 3050).
- **Ultra-lightweight VLM** — SmolVLM 256M Q8_0 GGUF + mmproj (~278 MB total).
- **Zero Config Required** — Categories are hardcoded in the classifier; simply point to an image directory.
- **Copy or Move Modes** — Organizes images cleanly into category folders.
- **Dry-run Mode** — Preview without altering any files.
- **Reports** — Automatic CSV and JSON output with classifications.

## Quick Start

```bash
# Basic usage (classifies into anime, game, movie, meme, coding)
imagecontexter classify ./my-photos

# Dry run preview (does not touch files)
imagecontexter classify ./my-photos --dry-run

# Move files instead of copying (saves disk space)
imagecontexter classify ./my-photos --mode move

# Custom output directory
imagecontexter classify ./my-photos -o ./sorted

# Recursive subfolder scan
imagecontexter classify ./my-photos -r

# (Optional) Override with custom categories YAML
imagecontexter classify ./my-photos -c custom.yaml
```

## CLI Reference

```
imagecontexter classify [OPTIONS] INPUT_DIR

Options:
  -c, --categories FILE     Optional path to a custom categories YAML file.
  -o, --output PATH         Output directory [default: <input_dir>_classified].
  --mode [copy|move]        Copy or move images into category folders.
  --dry-run                 Show classification without touching any files.
  --gpu-layers INTEGER      Number of layers to offload to GPU (-1 = all).
  --describe                Generate concise image description in report.
  -r, --recursive           Scan input directory recursively.
  --report [csv|json|both]  Report format written to output directory.
  --resume                  Skip images that appear in existing report.
  --help                    Show this message and exit.
```
