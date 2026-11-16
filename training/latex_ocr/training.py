"""Training loop. Interrupted runs continue from the last finished epoch."""

from __future__ import annotations

import csv
import ctypes
import dataclasses
import sys
import time
from pathlib import Path

import keras
import tensorflow as tf

from .data import Samples, load_split, make_dataset
from .download import split_path
from .inference import GreedyDecoder, transcribe
from .metrics import score
from .model import ModelConfig, build_model
from .tokenizer import PAD, Tokenizer

LABEL_SMOOTHING = 0.1
MONITOR_SAMPLES = 256
PROGRESS_STEPS = 200


def keep_system_awake() -> None:
    """Stop Windows sleeping while this process runs; it ends with the process."""
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def masked_loss(labels, logits):
    """Label-smoothed cross-entropy that ignores padding."""
    classes = tf.shape(logits)[-1]
    targets = tf.one_hot(tf.cast(labels, tf.int32), classes)
    losses = keras.losses.categorical_crossentropy(
        targets, logits, from_logits=True, label_smoothing=LABEL_SMOOTHING)
    mask = tf.cast(tf.not_equal(tf.cast(labels, tf.int32), PAD), losses.dtype)
    return tf.reduce_sum(losses * mask) / tf.reduce_sum(mask)


def token_accuracy(labels, logits):
    """Teacher-forced accuracy over real (non-padding) tokens."""
    labels = tf.cast(labels, tf.int32)
    predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
    mask = tf.not_equal(labels, PAD)
    hits = tf.logical_and(tf.equal(predicted, labels), mask)
    return tf.reduce_sum(tf.cast(hits, tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))


class TranscriptionMonitor(keras.callbacks.Callback):
    """Measures free-running output each epoch; val_loss only sees teacher forcing."""

    def __init__(self, samples: Samples, tokenizer: Tokenizer, decoder: GreedyDecoder):
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.decoder = decoder

    def on_epoch_end(self, epoch, logs=None):
        started = time.time()
        predictions = transcribe(self.decoder, self.tokenizer, self.samples)
        results = score(predictions, [text.split() for text in self.samples.texts])
        logs.update({f"val_{name}": value for name, value in results.items()})
        print(f"\nvalidation sample: exact_match={results['exact_match']:.3f} "
              f"edit_similarity={results['edit_similarity']:.3f} ({time.time() - started:.0f}s)")
        print(f"  reference : {self.samples.texts[0]}")
        print(f"  prediction: {' '.join(predictions[0])}", flush=True)


class ProgressLog(keras.callbacks.Callback):
    """Periodic log lines, since an epoch can take half an hour on a CPU."""

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % PROGRESS_STEPS == 0:
            seconds_per_step = (time.time() - self.started) / (batch + 1)
            print(f"  step {batch + 1}: loss={float(logs['loss']):.3f} "
                  f"token_accuracy={float(logs['token_accuracy']):.3f} "
                  f"({seconds_per_step:.2f}s/step)", flush=True)


def best_val_loss(history_path: Path) -> float | None:
    """Best score from an earlier session, so a resumed run cannot overwrite it."""
    if not history_path.exists():
        return None
    with open(history_path, newline="", encoding="utf-8") as history:
        values = [float(row["val_loss"]) for row in csv.DictReader(history)]
    return min(values, default=None)


def train(data_dir: Path, run_dir: Path, epochs: int, batch_size: int, learning_rate: float,
          limit: int | None = None, overrides: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    keep_system_awake()
    keras.utils.set_random_seed(42)

    print("loading data...", flush=True)
    train_samples = load_split(split_path(data_dir, "train"), limit)
    val_samples = load_split(split_path(data_dir, "validation"), limit)
    print(f"train={len(train_samples)} validation={len(val_samples)}", flush=True)

    vocab_path = run_dir / "vocab.json"
    config_path = run_dir / "config.json"
    if vocab_path.exists() and config_path.exists():
        tokenizer = Tokenizer.load(vocab_path)
        config = ModelConfig.load(config_path)
        print(f"continuing run in {run_dir}")
    else:
        tokenizer = Tokenizer.build(train_samples.texts)
        config = dataclasses.replace(ModelConfig(vocab_size=len(tokenizer)), **(overrides or {}))
        tokenizer.save(vocab_path)
        config.save(config_path)

    model = build_model(config)
    steps_per_epoch = (len(train_samples) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=learning_rate,
        warmup_steps=min(steps_per_epoch, total_steps // 10),
    )
    optimizer = keras.optimizers.AdamW(schedule, weight_decay=0.01, global_clipnorm=1.0)
    # jit_compile stays off: XLA would recompile for every new image size.
    model.compile(optimizer=optimizer, loss=masked_loss, metrics=[token_accuracy], jit_compile=False)
    print(f"parameters: {model.count_params():,}", flush=True)

    callbacks = [
        keras.callbacks.BackupAndRestore(str(run_dir / "backup"), delete_checkpoint=False),
        ProgressLog(),
        TranscriptionMonitor(val_samples.take(MONITOR_SAMPLES), tokenizer, GreedyDecoder(model)),
        keras.callbacks.ModelCheckpoint(
            run_dir / "best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=best_val_loss(run_dir / "history.csv"),
        ),
        keras.callbacks.CSVLogger(run_dir / "history.csv", append=True),
    ]
    model.fit(
        make_dataset(train_samples, tokenizer, batch_size, training=True),
        validation_data=make_dataset(val_samples, tokenizer, batch_size, training=False),
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,  # make_dataset already shuffles
        verbose=2,
    )
