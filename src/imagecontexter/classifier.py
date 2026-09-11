"""Core image classification engine using Fast OCR + SmolVLM-256M-Instruct GGUF via llama.cpp.

Pipeline:
  1. OCR Pre-check (ultra-fast, ~150-250ms):
     - If keywords ("ago", "subscribers", "views", "comment") occur -> classified as "youtube"
     - If keywords ("spotify", "album", "playlist", "songs", "queue") occur -> classified as "spotify"
     - If code syntax ("def", "import", "class", "return") occur -> classified as "coding"
  2. If OCR does not match, runs SmolVLM-256M-Instruct GGUF on GPU for categories:
     game, anime, movie, meme, coding.
"""

from __future__ import annotations

import base64
import ctypes
import glob
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import DEFAULT_CATEGORIES, Category
from .ocr import FastOCRClassifier

logger = logging.getLogger(__name__)

# Image file extensions we recognise
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
)


def _setup_cuda_dlls() -> None:
    """Ensure CUDA and llama-cpp DLLs are discoverable on Windows."""
    if sys.platform != "win32":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"

    # Add NVIDIA runtime DLL paths (cublas, cudart, nvrtc)
    for p in glob.glob(str(site_packages / "nvidia" / "*" / "bin")):
        try:
            os.add_dll_directory(p)
        except (AttributeError, OSError):
            pass

    # Add llama_cpp lib folder
    llama_lib = site_packages / "llama_cpp" / "lib"
    if llama_lib.is_dir():
        try:
            os.add_dll_directory(str(llama_lib))
        except (AttributeError, OSError):
            pass

        # Pre-load dependent DLLs in order to satisfy dynamic linking
        dll_order = ["ggml-base", "ggml", "ggml-cpu", "ggml-cuda", "mtmd"]
        for dll_name in dll_order:
            dll_file = llama_lib / f"{dll_name}.dll"
            if dll_file.exists():
                try:
                    ctypes.CDLL(str(dll_file))
                except Exception as e:
                    logger.debug(f"Preload {dll_name}: {e}")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Outcome of classifying a single image."""

    image_path: str
    category: str
    description: str | None
    raw_answer: str


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ImageClassifier:
    """Classify images using fast OCR pre-filtering + SmolVLM-256M-Instruct GGUF on GPU."""

    HF_REPO_ID = "ggml-org/SmolVLM-256M-Instruct-GGUF"
    MODEL_FILENAME = "SmolVLM-256M-Instruct-Q8_0.gguf"
    MMPROJ_FILENAME = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

    def __init__(
        self,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        verbose: bool = False,
    ) -> None:
        """Initialize Fast OCR and SmolVLM GGUF model on GPU."""
        self.ocr = FastOCRClassifier()

        _setup_cuda_dlls()

        try:
            import llama_cpp
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required. Install with GPU support via:\n"
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
            ) from exc

        import huggingface_hub

        logger.info(f"Checking/downloading {self.HF_REPO_ID} files...")
        model_path = huggingface_hub.hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=self.MODEL_FILENAME,
        )
        mmproj_path = huggingface_hub.hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=self.MMPROJ_FILENAME,
        )

        logger.info("Initializing GPU multimodal chat handler (mmproj)...")
        chat_handler = MTMDChatHandler(clip_model_path=mmproj_path, use_gpu=True)

        logger.info(f"Loading Llama model with {n_gpu_layers} layers offloaded to GPU...")
        self.llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
        )

    # ---- public API -------------------------------------------------------

    def classify(
        self,
        image_path: Path,
        categories: Optional[list[Category]] = None,
        *,
        describe: bool = False,
        use_ocr: bool = True,
    ) -> ClassificationResult:
        """Classify a single image into defined categories."""
        if categories is None:
            categories = DEFAULT_CATEGORIES

        # -------------------------------------------------------------------
        # Step 1: Fast OCR Pre-filtering
        # -------------------------------------------------------------------
        if use_ocr:
            ocr_result = self.ocr.check(image_path)
            if ocr_result is not None:
                matched_cat, detail = ocr_result
                return ClassificationResult(
                    image_path=str(image_path),
                    category=matched_cat,
                    description=detail,
                    raw_answer=f"OCR:{detail}",
                )

        # -------------------------------------------------------------------
        # Step 2: SmolVLM Vision Language Model (GPU)
        # -------------------------------------------------------------------
        data_uri = self._image_to_data_uri(image_path)

        description: str | None = None
        if describe:
            desc_prompt = "Provide a concise description of what is depicted in this image."
            desc_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": desc_prompt},
                    ],
                }
            ]
            desc_resp = self.llm.create_chat_completion(
                messages=desc_messages,
                max_tokens=50,
                temperature=0.1,
            )
            description = desc_resp["choices"][0]["message"]["content"].strip()

        # VLM categories
        vlm_categories = [c for c in categories if c.name not in ("youtube", "spotify")]
        if not vlm_categories:
            vlm_categories = categories

        # Step 2A: Check video game first (small VLMs distinguish binary queries with 99% accuracy)
        game_check = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "Question: Is this image a video game screenshot? Answer with only yes or no."},
                    ],
                }
            ],
            max_tokens=5,
            temperature=0.0,
        )
        game_ans = game_check["choices"][0]["message"]["content"].strip().lower()
        if "yes" in game_ans and "no" not in game_ans:
            return ClassificationResult(
                image_path=str(image_path),
                category="game",
                description=description,
                raw_answer="VLM: video game verified",
            )

        # Step 2B: Multi-class query for remaining categories (anime, movie, meme, coding)
        prompt = (
            "Question: What type of content does this image depict?\n"
            "Options: anime drawing, live-action movie, internet meme, programming code, video game.\n"
            "Answer with the single option that best matches:"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=15,
            temperature=0.0,
        )

        raw_answer = response["choices"][0]["message"]["content"].strip()
        matched = self._match_category(raw_answer, vlm_categories)

        return ClassificationResult(
            image_path=str(image_path),
            category=matched,
            description=description,
            raw_answer=raw_answer,
        )

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _image_to_data_uri(image_path: Path) -> str:
        """Load and encode image to JPEG base64 Data URI."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            max_dim = 1024
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"

    @staticmethod
    def _match_category(answer: str, categories: list[Category]) -> str:
        """Robustly match the VLM answer against candidate categories."""
        import re

        clean_lower = answer.lower()

        # Check for phrase-level strong hints first
        if "video game" in clean_lower or "gameplay" in clean_lower or "gaming" in clean_lower:
            return "game"
        if "program" in clean_lower or "code" in clean_lower or "script" in clean_lower:
            return "coding"
        if "meme" in clean_lower:
            return "meme"
        if "movie" in clean_lower or "film" in clean_lower or "cinematic" in clean_lower:
            return "movie"
        if "anime" in clean_lower or "manga" in clean_lower or "cartoon" in clean_lower:
            return "anime"

        # Check if the answer contains only a single category word
        clean_words = re.findall(r"\b[a-z]+\b", clean_lower)
        matches = [c.name for c in categories if c.name in clean_words]

        if len(matches) == 1:
            return matches[0]

        for w in clean_words:
            for cat in categories:
                if cat.name == w and w != "anime":
                    return cat.name

        return matches[0] if matches else (categories[0].name if categories else "uncategorized")


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def collect_images(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Return sorted image files from directory."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        f
        for f in directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
