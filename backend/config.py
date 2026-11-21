"""Backend settings. Each one can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
TRAINING_DIR = PROJECT_DIR / "training"

# The model trained on the RTX 3050 in WSL (batch size 16, 10 epochs).
DEFAULT_RUN_DIR = TRAINING_DIR / "runs" / "default"


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


class Config:
    # Folder holding best.weights.h5, config.json and vocab.json.
    RUN_DIR = Path(os.environ.get("LATEX_OCR_RUN_DIR", DEFAULT_RUN_DIR))

    # Largest request body Flask accepts; larger uploads get 413.
    MAX_CONTENT_LENGTH = _int("LATEX_OCR_MAX_UPLOAD_MB", 10) * 1024 * 1024

    # Largest decoded image, checked before the pixels are loaded.
    MAX_IMAGE_PIXELS = _int("LATEX_OCR_MAX_IMAGE_PIXELS", 25_000_000)

    # Most images accepted by one /api/predict/batch request.
    MAX_BATCH_SIZE = _int("LATEX_OCR_MAX_BATCH", 16)

    # "*" or a comma-separated list such as "http://localhost:5173,http://localhost:3000".
    CORS_ORIGINS = os.environ.get("LATEX_OCR_CORS_ORIGINS", "*")

    # Keep non-ASCII characters readable in JSON responses.
    JSON_AS_ASCII = False
