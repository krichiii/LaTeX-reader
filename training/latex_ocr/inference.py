"""Greedy decoding, the image-to-Markdown predictor, and test-set evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .data import Samples, image_to_ink, pad_images, rescale
from .markdown import to_markdown, tokens_to_latex
from .metrics import score
from .model import FormulaReader, ModelConfig, build_model
from .tokenizer import BOS, EOS, Tokenizer

# The model only reads glyphs near the dataset's size, and an image does not say
# what size its glyphs are, so prediction tries a range of scales.
SCALES = (1.25, 1.0, 0.8, 0.64, 0.51, 0.41, 0.33, 0.26, 0.21, 0.17)
MIN_HEIGHT = 6


class GreedyDecoder:
    def __init__(self, model: FormulaReader):
        self.model = model
        config = model.config
        history_spec = [
            tf.TensorSpec((None, None, config.d_model), tf.float32)
            for _ in range(config.decoder_layers)
        ]
        self._encode = tf.function(
            lambda images: model.encode(images, training=False),
            input_signature=[tf.TensorSpec((None, None, None, 1), tf.float32)],
        )
        self._step = tf.function(
            self._step_impl,
            input_signature=[
                tf.TensorSpec((None,), tf.int32),
                tf.TensorSpec((), tf.int32),
                history_spec,
                tf.TensorSpec((None, None, config.d_model), tf.float32),
            ],
        )

    def _step_impl(self, token, position, histories, memory):
        logits, histories = self.model.decode_step(token, position, histories, memory)
        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predicted, tf.reduce_max(tf.nn.log_softmax(logits), axis=-1), histories

    def decode(self, images: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        """Return token ids (without BOS/EOS) and a confidence for each image.

        Confidence is the mean log-probability of the chosen tokens. Sequences
        that never reach EOS get -inf: runaway repetition can look confident.
        """
        config = self.model.config
        memory = self._encode(tf.constant(images))
        count = images.shape[0]
        outputs: list[list[int]] = [[] for _ in range(count)]
        scores = np.zeros(count)
        steps = np.zeros(count)
        finished = np.zeros(count, dtype=bool)
        active = np.arange(count)
        token = np.full(count, BOS, dtype=np.int32)
        histories = [tf.zeros((count, 0, config.d_model)) for _ in range(config.decoder_layers)]

        for position in range(config.max_positions - 1):
            predicted, log_probs, histories = self._step(
                tf.constant(token), tf.constant(position, tf.int32), histories, memory)
            predicted = predicted.numpy()
            scores[active] += log_probs.numpy()
            steps[active] += 1
            unfinished = predicted != EOS
            finished[active[~unfinished]] = True
            for index, value in zip(active[unfinished], predicted[unfinished]):
                outputs[index].append(int(value))
            if not unfinished.any():
                break
            if not unfinished.all():
                # Finished sequences leave the batch, so they stop costing time.
                mask = tf.constant(unfinished)
                memory = tf.boolean_mask(memory, mask)
                histories = [tf.boolean_mask(history, mask) for history in histories]
                active = active[unfinished]
            token = predicted[unfinished]
        confidence = scores / np.maximum(steps, 1)
        confidence[~finished] = -np.inf
        return outputs, confidence


def transcribe(decoder: GreedyDecoder, tokenizer: Tokenizer, samples: Samples,
               batch_size: int = 32, verbose: bool = False) -> list[list[str]]:
    """Predict tokens for every sample, keeping the input order."""
    order = np.argsort([image.shape[1] for image in samples.images], kind="stable")
    results: list[list[str]] = [[] for _ in range(len(samples))]
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        predicted, _ = decoder.decode(pad_images([samples.images[i] for i in indices]))
        for index, ids in zip(indices, predicted):
            results[index] = tokenizer.decode(ids)
        if verbose:
            done = min(start + batch_size, len(order))
            print(f"\r{done}/{len(order)}", end="", flush=True)
    if verbose:
        print()
    return results


class Predictor:
    """Loads a finished run and converts formula images to Markdown."""

    def __init__(self, run_dir: Path):
        self.tokenizer = Tokenizer.load(run_dir / "vocab.json")
        model = build_model(ModelConfig.load(run_dir / "config.json"))
        model.load_weights(run_dir / "best.weights.h5")
        self.decoder = GreedyDecoder(model)

    def ink_tokens(self, ink: np.ndarray) -> list[str]:
        """Read an ink map at every scale and keep the most confident result."""
        variants = [rescale(ink, scale) for scale in SCALES]
        variants = [variant for variant in variants if variant.shape[0] >= MIN_HEIGHT] or [ink]
        outputs, confidence = self.decoder.decode(pad_images(variants))
        return self.tokenizer.decode(outputs[int(np.argmax(confidence))])

    def latex(self, image: Image.Image) -> str:
        return tokens_to_latex(self.ink_tokens(image_to_ink(image)))

    def markdown(self, image: Image.Image, inline: bool = False) -> str:
        return to_markdown(self.latex(image), inline=inline)


def evaluate(run_dir: Path, samples: Samples, batch_size: int = 32, multi_scale: bool = False) -> dict:
    """Score a run on `samples` and save the result into the run directory.

    Single-scale is fast and suits dataset images, which are already at the
    training scale. Multi-scale scores exactly what `predict` does.
    """
    predictor = Predictor(run_dir)
    if multi_scale:
        predictions = []
        for index, ink in enumerate(samples.images, start=1):
            predictions.append(predictor.ink_tokens(ink))
            print(f"\r{index}/{len(samples)}", end="", flush=True)
        print()
    else:
        predictions = transcribe(predictor.decoder, predictor.tokenizer, samples, batch_size, verbose=True)
    references = [text.split() for text in samples.texts]
    report = {"samples": len(samples), "multi_scale": multi_scale, **score(predictions, references)}
    (run_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pairs = [{"reference": " ".join(reference), "prediction": " ".join(prediction)}
             for reference, prediction in zip(references, predictions)]
    (run_dir / "test_predictions.json").write_text(json.dumps(pairs, indent=0), encoding="utf-8")
    return report
