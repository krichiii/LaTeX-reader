"""Image preprocessing and batching.

Images are reduced to a tightly cropped "ink" map (255 = ink, 0 = background).
Cropping removes most of the blank canvas, and batches are grouped by width and
padded only to their own maximum: both matter a lot for CPU training speed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from PIL import Image

from .tokenizer import BOS, EOS, PAD, Tokenizer

MAX_WIDTH = 480
MAX_HEIGHT = 128
MARGIN = 4
STRIDE = 8  # total CNN downsampling; padded sizes must be a multiple of this
INK_THRESHOLD = 55
MAX_FORMULA_TOKENS = 150
POOL_BATCHES = 50  # batches' worth of samples grouped by width at a time


def image_to_ink(image: Image.Image) -> np.ndarray:
    """Convert any image to a cropped uint8 ink map; raises ValueError if blank."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    gray = np.asarray(Image.alpha_composite(white, rgba).convert("L")).astype(np.int16)

    # The median pixel is background, so distance from it works for light,
    # grey and dark-mode images alike.
    ink = np.abs(gray - int(np.median(gray)))
    peak = ink.max()
    if peak == 0:
        raise ValueError("image has no visible content")
    ink = (ink * (255.0 / peak)).astype(np.uint8)

    mask = ink > INK_THRESHOLD
    if not mask.any():
        raise ValueError("image has no visible content")
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    cropped = ink[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    return rescale(cropped, 1.0)


def rescale(ink: np.ndarray, factor: float) -> np.ndarray:
    """Apply `factor`, then shrink further if needed to fit the model's input limit."""
    height, width = ink.shape
    limit = min((MAX_WIDTH - 2 * MARGIN) / width, (MAX_HEIGHT - 2 * MARGIN) / height)
    factor = min(factor, limit)
    if factor == 1.0:
        return ink
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return np.asarray(Image.fromarray(ink).resize(size, Image.Resampling.BILINEAR))


def pad_images(images: list[np.ndarray], offsets: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Stack ink maps into a float batch of shape (N, H, W, 1) with values in [0, 1]."""
    height = max(image.shape[0] for image in images) + 2 * MARGIN
    width = max(image.shape[1] for image in images) + 2 * MARGIN
    height = int(np.ceil(height / STRIDE)) * STRIDE
    width = int(np.ceil(width / STRIDE)) * STRIDE

    batch = np.zeros((len(images), height, width, 1), dtype=np.float32)
    for index, image in enumerate(images):
        top, left = offsets[index] if offsets else (MARGIN, MARGIN)
        rows, columns = image.shape
        batch[index, top:top + rows, left:left + columns, 0] = image
    return batch / 255.0


def pad_tokens(sequences: list[list[int]]) -> np.ndarray:
    """Add BOS/EOS and pad to the longest sequence in the batch."""
    length = max(len(sequence) for sequence in sequences) + 2
    batch = np.full((len(sequences), length), PAD, dtype=np.int32)
    for index, sequence in enumerate(sequences):
        full = [BOS, *sequence, EOS]
        batch[index, :len(full)] = full
    return batch


@dataclass
class Samples:
    images: list[np.ndarray]
    texts: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    def take(self, count: int) -> "Samples":
        return Samples(self.images[:count], self.texts[:count])


def load_split(path: Path, limit: int | None = None) -> Samples:
    """Read a parquet split, dropping unusable rows (blank images, empty labels)."""
    table = pq.read_table(path)
    if limit:
        table = table.slice(0, limit)
    png_bytes = table["image"].combine_chunks().field("bytes").to_pylist()
    texts = table["text"].to_pylist()

    images: list[np.ndarray] = []
    kept_texts: list[str] = []
    for data, text in zip(png_bytes, texts):
        if not text or not text.strip() or len(text.split()) > MAX_FORMULA_TOKENS:
            continue
        try:
            images.append(image_to_ink(Image.open(io.BytesIO(data))))
        except ValueError:
            continue
        kept_texts.append(text)
    return Samples(images, kept_texts)


def augment(ink: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    """Vary scale, stroke strength and placement so real screenshots generalise better."""
    ink = rescale(ink, float(rng.uniform(0.75, 1.1)))
    ink = (ink * rng.uniform(0.7, 1.0)).astype(np.uint8)
    return ink, (int(rng.integers(0, 2 * MARGIN + 1)), int(rng.integers(0, 2 * MARGIN + 1)))


def _batch_indices(widths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle, then group similar widths inside large pools to minimise padding."""
    order = rng.permutation(len(widths))
    pool_size = batch_size * POOL_BATCHES
    batches: list[np.ndarray] = []
    for start in range(0, len(order), pool_size):
        pool = order[start:start + pool_size]
        pool = pool[np.argsort(widths[pool], kind="stable")]
        batches.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
    rng.shuffle(batches)
    return batches


def make_dataset(
    samples: Samples,
    tokenizer: Tokenizer,
    batch_size: int,
    training: bool,
    seed: int = 0,
) -> tf.data.Dataset:
    """Yield ((images, decoder_input), labels) batches with per-batch dynamic shapes."""
    sequences = [tokenizer.encode(text) for text in samples.texts]
    widths = np.array([image.shape[1] for image in samples.images])
    rng = np.random.default_rng(seed)
    batch_count = int(np.ceil(len(samples) / batch_size))

    def generate():
        for indices in _batch_indices(widths, batch_size, rng):
            images = [samples.images[i] for i in indices]
            offsets = None
            if training:
                augmented = [augment(image, rng) for image in images]
                images, offsets = (list(column) for column in zip(*augmented))
            pixels = pad_images(images, offsets)
            if training:
                pixels = np.clip(pixels + rng.normal(0, 0.02, pixels.shape).astype(np.float32), 0, 1)
            tokens = pad_tokens([sequences[i] for i in indices])
            yield (pixels, tokens[:, :-1]), tokens[:, 1:]

    signature = (
        (
            tf.TensorSpec((None, None, None, 1), tf.float32),
            tf.TensorSpec((None, None), tf.int32),
        ),
        tf.TensorSpec((None, None), tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(batch_count))
    return dataset.prefetch(4)
