"""CNN encoder + Transformer decoder.

The model works with dynamic image width/height and token length. Weights are
stored without architecture; `ModelConfig` (saved next to them) rebuilds it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import tensorflow as tf
from keras import layers

from .data import MAX_FORMULA_TOKENS, MARGIN, STRIDE
from .tokenizer import PAD


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 192
    num_heads: int = 6
    encoder_layers: int = 0  # self-attention over the image is the priciest part on CPU
    decoder_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.1
    max_positions: int = MAX_FORMULA_TOKENS + 2  # BOS and EOS

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def sinusoids(length: tf.Tensor, dim: int) -> tf.Tensor:
    """Sinusoidal position encoding of shape (length, dim)."""
    positions = tf.cast(tf.range(length), tf.float32)[:, tf.newaxis]
    half = dim // 2
    exponents = tf.cast(tf.range(half), tf.float32) / float(half)
    angles = positions / tf.pow(10000.0, exponents)[tf.newaxis, :]
    return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)


def conv_block(filters: int) -> keras.Sequential:
    return keras.Sequential([
        layers.Conv2D(filters, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
    ])


class EncoderBlock(layers.Layer):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization()
        self.attention = layers.MultiHeadAttention(
            config.num_heads, config.d_model // config.num_heads, dropout=config.dropout)
        self.norm2 = layers.LayerNormalization()
        self.feed_forward = keras.Sequential([
            layers.Dense(config.ff_dim, activation="gelu"),
            layers.Dense(config.d_model),
        ])
        self.dropout = layers.Dropout(config.dropout)

    def call(self, x, training=False):
        normed = self.norm1(x)
        attended = self.attention(normed, normed, training=training)
        x = x + self.dropout(attended, training=training)
        return x + self.dropout(self.feed_forward(self.norm2(x)), training=training)


class DecoderBlock(layers.Layer):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization()
        self.self_attention = layers.MultiHeadAttention(
            config.num_heads, config.d_model // config.num_heads, dropout=config.dropout)
        self.norm2 = layers.LayerNormalization()
        self.cross_attention = layers.MultiHeadAttention(
            config.num_heads, config.d_model // config.num_heads, dropout=config.dropout)
        self.norm3 = layers.LayerNormalization()
        self.feed_forward = keras.Sequential([
            layers.Dense(config.ff_dim, activation="gelu"),
            layers.Dense(config.d_model),
        ])
        self.dropout = layers.Dropout(config.dropout)

    def _finish(self, x, memory, training):
        """Cross-attention and feed-forward, shared by training and cached decoding."""
        cross = self.cross_attention(self.norm2(x), memory, training=training)
        x = x + self.dropout(cross, training=training)
        return x + self.dropout(self.feed_forward(self.norm3(x)), training=training)

    def call(self, x, memory, attention_mask, training=False):
        normed = self.norm1(x)
        attended = self.self_attention(normed, normed, attention_mask=attention_mask, training=training)
        x = x + self.dropout(attended, training=training)
        return self._finish(x, memory, training)

    def step(self, x, history, memory):
        """Process one new position; `history` holds the normed earlier positions.

        With causal masking, earlier positions never depend on later ones, so
        they do not need recomputing and each decoding step stays cheap.
        """
        normed = self.norm1(x)
        history = tf.concat([history, normed], axis=1)
        x = x + self.self_attention(normed, history, training=False)
        return self._finish(x, memory, False), history


class FormulaReader(keras.Model):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.cnn = keras.Sequential([
            conv_block(24),
            layers.MaxPool2D(2),
            conv_block(48),
            layers.MaxPool2D(2),
            conv_block(96),
            conv_block(96),
            layers.MaxPool2D(2),
            conv_block(192),
            layers.Conv2D(config.d_model, 1),
        ])
        self.encoder_blocks = [EncoderBlock(config) for _ in range(config.encoder_layers)]
        self.encoder_norm = layers.LayerNormalization()

        self.token_embedding = layers.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = layers.Embedding(config.max_positions, config.d_model)
        self.embedding_dropout = layers.Dropout(config.dropout)
        self.decoder_blocks = [DecoderBlock(config) for _ in range(config.decoder_layers)]
        self.decoder_norm = layers.LayerNormalization()
        self.classifier = layers.Dense(config.vocab_size)

    def encode(self, images, training=False):
        features = self.cnn(images, training=training)
        shape = tf.shape(features)
        half = self.config.d_model // 2
        rows = tf.tile(sinusoids(shape[1], half)[:, tf.newaxis, :], [1, shape[2], 1])
        columns = tf.tile(sinusoids(shape[2], half)[tf.newaxis, :, :], [shape[1], 1, 1])
        features = features + tf.concat([rows, columns], axis=-1)[tf.newaxis]
        memory = tf.reshape(features, [shape[0], shape[1] * shape[2], self.config.d_model])
        for block in self.encoder_blocks:
            memory = block(memory, training=training)
        return self.encoder_norm(memory)

    def decode(self, tokens, memory, training=False):
        length = tf.shape(tokens)[1]
        x = self.token_embedding(tokens) + self.position_embedding(tf.range(length))
        x = self.embedding_dropout(x, training=training)

        causal = tf.linalg.band_part(tf.ones((length, length), tf.bool), -1, 0)
        # Padded key positions are hidden; a position may always attend to itself
        # so that fully padded rows do not produce an empty attention distribution.
        not_padding = tokens != PAD
        attention_mask = causal[tf.newaxis] & (not_padding[:, tf.newaxis, :] | tf.eye(length, dtype=tf.bool))

        for block in self.decoder_blocks:
            x = block(x, memory, attention_mask, training=training)
        return self.classifier(self.decoder_norm(x))

    def decode_step(self, token, position, histories, memory):
        """Logits for one new token per sequence, plus updated histories."""
        x = self.token_embedding(token[:, tf.newaxis])
        x = x + self.position_embedding(position[tf.newaxis])[tf.newaxis]
        updated = []
        for block, history in zip(self.decoder_blocks, histories):
            x, history = block.step(x, history, memory)
            updated.append(history)
        return self.classifier(self.decoder_norm(x))[:, 0], updated

    def call(self, inputs, training=False):
        images, tokens = inputs
        return self.decode(tokens, self.encode(images, training=training), training=training)


def build_model(config: ModelConfig) -> FormulaReader:
    """Create a model with all weights allocated."""
    model = FormulaReader(config)
    size = (MARGIN * 2 + STRIDE) // STRIDE * STRIDE
    model((tf.zeros((1, size, size, 1)), tf.ones((1, 2), tf.int32)))
    return model
