"""Runs the trained formula reader from ../training on uploaded images.

TensorFlow is imported only when `OcrService` is created, so the API layer and
its tests can run without loading the model.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import TRAINING_DIR

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# The model code lives in training/latex_ocr. Appended rather than prepended so
# that training/tests can never shadow backend/tests.
if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

REQUIRED_FILES = ("best.weights.h5", "config.json", "vocab.json")


class NoContentError(ValueError):
    """The image is blank, so there is no formula to read."""


@dataclass
class Reading:
    latex: str
    markdown: str
    markdown_inline: str
    # Geometric mean of the chosen tokens' probabilities, from 0 to 1. None when
    # the model never ended the formula, which usually means runaway repetition.
    confidence: float | None
    complete: bool
    # False when { and } do not pair up, so the LaTeX will not render.
    balanced: bool
    multi_scale: bool
    elapsed_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


def braces_balanced(tokens: list[str]) -> bool:
    depth = 0
    for token in tokens:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


class OcrService:
    """Loads one training run and reads formulas with it.

    Prediction is serialised with a lock: TensorFlow already uses every core
    for a single image, so parallel requests would only compete for them.
    """

    def __init__(self, run_dir: Path, warm_up: bool = True):
        missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{run_dir} is missing {', '.join(missing)}")

        import tensorflow as tf
        from latex_ocr.inference import Predictor

        started = time.perf_counter()
        self.run_dir = run_dir
        self._predictor = Predictor(run_dir)
        self._lock = threading.Lock()
        self._devices = [device.name for device in tf.config.list_physical_devices("GPU")]
        self._tf_version = tf.__version__
        metrics_path = run_dir / "test_metrics.json"
        self._test_metrics = (json.loads(metrics_path.read_text(encoding="utf-8"))
                              if metrics_path.is_file() else None)
        if warm_up:
            self._warm_up()
        self.load_seconds = round(time.perf_counter() - started, 2)

    def _warm_up(self) -> None:
        """Trace the decoder's graphs now so the first request is not slow."""
        from latex_ocr.data import pad_images
        ink = np.zeros((24, 48), dtype=np.uint8)
        ink[8:16, 8:40] = 255
        self._predictor.decoder.decode(pad_images([ink]))

    def read(self, image: Image.Image, multi_scale: bool = True) -> Reading:
        """Read the formula in `image`. Raises NoContentError if it is blank."""
        from latex_ocr.data import image_to_ink, pad_images, rescale
        from latex_ocr.inference import MIN_HEIGHT, SCALES
        from latex_ocr.markdown import to_markdown, tokens_to_latex

        started = time.perf_counter()
        try:
            ink = image_to_ink(image)
        except ValueError as error:
            raise NoContentError(str(error)) from error

        # Same strategy as Predictor.ink_tokens, but keeping the confidence.
        scales = SCALES if multi_scale else (1.0,)
        variants = [rescale(ink, scale) for scale in scales]
        variants = [variant for variant in variants if variant.shape[0] >= MIN_HEIGHT] or [ink]
        with self._lock:
            outputs, confidence = self._predictor.decoder.decode(pad_images(variants))
        best = int(np.argmax(confidence))
        tokens = self._predictor.tokenizer.decode(outputs[best])
        latex = tokens_to_latex(tokens)
        complete = bool(np.isfinite(confidence[best]))
        return Reading(
            latex=latex,
            markdown=to_markdown(latex),
            markdown_inline=to_markdown(latex, inline=True),
            confidence=round(math.exp(float(confidence[best])), 4) if complete else None,
            complete=complete,
            balanced=braces_balanced(tokens),
            multi_scale=multi_scale,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )

    def info(self) -> dict:
        model = self._predictor.decoder.model
        return {
            "run": self.run_dir.name,
            "parameters": int(model.count_params()),
            "config": asdict(model.config),
            "test_metrics": self._test_metrics,
            "device": "GPU" if self._devices else "default",
            "gpus": self._devices,
            "tensorflow": self._tf_version,
            "load_seconds": self.load_seconds,
        }
