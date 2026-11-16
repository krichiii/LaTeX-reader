"""Run from the project root with: python -m unittest tests.test_project"""

import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from latex_ocr.data import (
    MAX_HEIGHT, MAX_WIDTH, STRIDE, Samples, image_to_ink, make_dataset, pad_images, pad_tokens,
)
from latex_ocr.inference import GreedyDecoder
from latex_ocr.markdown import to_markdown, tokens_to_latex
from latex_ocr.metrics import edit_distance, score
from latex_ocr.model import ModelConfig, build_model
from latex_ocr.tokenizer import BOS, EOS, PAD, UNK, Tokenizer
from latex_ocr.training import masked_loss, token_accuracy


class MarkdownTests(unittest.TestCase):
    def test_compact_join(self):
        self.assertEqual(tokens_to_latex("\\frac { 1 } { 2 }".split()), "\\frac{1}{2}")

    def test_space_kept_between_command_and_letter(self):
        self.assertEqual(tokens_to_latex("\\alpha x".split()), "\\alpha x")
        self.assertEqual(tokens_to_latex("\\alpha 2".split()), "\\alpha2")

    def test_control_space_and_row_break(self):
        self.assertEqual(tokens_to_latex(["a", "\\", "b"]), "a\\ b")
        self.assertEqual(tokens_to_latex(["a", "\\\\", "[", "b"]), "a\\\\ [b")

    def test_no_trailing_space(self):
        self.assertEqual(tokens_to_latex(["a", "\\"]), "a\\")

    def test_wrapping(self):
        self.assertEqual(to_markdown("x"), "$$\nx\n$$")
        self.assertEqual(to_markdown("x", inline=True), "$x$")


class TokenizerTests(unittest.TestCase):
    def test_round_trip_and_unknown(self):
        tokenizer = Tokenizer.build(["a b", "a b"], min_count=2)
        ids = tokenizer.encode("a z")
        self.assertEqual(ids[1], UNK)
        self.assertEqual(tokenizer.decode(ids), ["a"])

    def test_decode_drops_specials(self):
        tokenizer = Tokenizer.build(["a a"], min_count=1)
        self.assertEqual(tokenizer.decode([BOS, tokenizer.ids["a"], EOS, PAD]), ["a"])


class PreprocessTests(unittest.TestCase):
    @staticmethod
    def drawn(background, ink, size=(300, 100)):
        image = Image.new("RGB", size, background)
        ImageDraw.Draw(image).rectangle((50, 20, 150, 60), fill=ink)
        return image

    def test_crops_to_content(self):
        self.assertEqual(image_to_ink(self.drawn("white", "black")).shape, (41, 101))

    def test_dark_mode_matches_light_mode(self):
        light = image_to_ink(self.drawn((255, 255, 255), (0, 0, 0)))
        dark = image_to_ink(self.drawn((0, 0, 0), (255, 255, 255)))
        np.testing.assert_array_equal(light, dark)

    def test_grey_background_gives_full_contrast(self):
        self.assertEqual(image_to_ink(self.drawn((200, 200, 200), (60, 60, 60))).max(), 255)

    def test_blank_image_is_rejected(self):
        with self.assertRaises(ValueError):
            image_to_ink(Image.new("RGB", (100, 100), "white"))

    def test_oversized_image_fits_limit(self):
        image = Image.new("RGB", (3000, 900), "white")
        ImageDraw.Draw(image).rectangle((10, 10, 2990, 890), fill="black")
        height, width = image_to_ink(image).shape
        self.assertLessEqual(width, MAX_WIDTH)
        self.assertLessEqual(height, MAX_HEIGHT)

    def test_batch_is_stride_aligned(self):
        batch = pad_images([np.ones((10, 30), np.uint8), np.ones((5, 50), np.uint8)])
        self.assertEqual(batch.shape[1] % STRIDE, 0)
        self.assertEqual(batch.shape[2] % STRIDE, 0)

    def test_pad_tokens(self):
        batch = pad_tokens([[7], [7, 8]])
        np.testing.assert_array_equal(batch, [[BOS, 7, EOS, PAD], [BOS, 7, 8, EOS]])


class DatasetTests(unittest.TestCase):
    def test_every_sample_appears_once_per_epoch(self):
        tokenizer = Tokenizer.build(["a"], min_count=1)
        count = 10
        samples = Samples(
            [np.full((10, 20 + index), 255, np.uint8) for index in range(count)],
            ["a"] * count,
        )
        dataset = make_dataset(samples, tokenizer, batch_size=3, training=True)
        for _ in range(2):  # the second pass proves the generator restarts
            sizes = [images.shape[0] for (images, _), _ in dataset]
            self.assertEqual(sum(sizes), count)
            self.assertEqual(len(sizes), 4)


class MetricTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance(list("kitten"), list("sitting")), 3)
        self.assertEqual(edit_distance([], ["a"]), 1)

    def test_score(self):
        result = score([["a", "b"], ["c"]], [["a", "b"], ["d"]])
        self.assertAlmostEqual(result["exact_match"], 0.5)
        self.assertAlmostEqual(result["edit_similarity"], 2 / 3)

    def test_loss_and_accuracy_ignore_padding(self):
        labels = tf.constant([[5, PAD]])
        good = tf.one_hot([[5, 1]], 8) * 20.0
        bad_padding = tf.one_hot([[5, 2]], 8) * 20.0
        self.assertAlmostEqual(float(masked_loss(labels, good)), float(masked_loss(labels, bad_padding)), 5)
        self.assertAlmostEqual(float(token_accuracy(labels, good)), 1.0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.random.set_seed(0)
        cls.config = ModelConfig(vocab_size=20, d_model=32, num_heads=2, encoder_layers=1,
                                 decoder_layers=2, ff_dim=64, max_positions=12)
        cls.model = build_model(cls.config)

    def test_output_shape_with_dynamic_sizes(self):
        for height, width in ((16, 40), (24, 72)):
            logits = self.model((tf.zeros((2, height, width, 1)), tf.ones((2, 5), tf.int32)))
            self.assertEqual(tuple(logits.shape), (2, 5, 20))

    def test_cached_step_matches_full_decode(self):
        images = tf.random.uniform((1, 16, 32, 1))
        tokens = tf.constant([[BOS, 6, 9, 4, 11]])
        memory = self.model.encode(images)
        full = self.model.decode(tokens, memory).numpy()
        histories = [tf.zeros((1, 0, self.config.d_model)) for _ in range(self.config.decoder_layers)]
        for position in range(tokens.shape[1]):
            logits, histories = self.model.decode_step(
                tokens[:, position], tf.constant(position), histories, memory)
            np.testing.assert_allclose(logits.numpy(), full[:, position], atol=1e-4)

    def test_future_tokens_do_not_change_past_logits(self):
        memory = self.model.encode(tf.random.uniform((1, 16, 32, 1)))
        first = self.model.decode(tf.constant([[BOS, 6, 9]]), memory).numpy()
        second = self.model.decode(tf.constant([[BOS, 6, 15]]), memory).numpy()
        np.testing.assert_allclose(first[:, :2], second[:, :2], atol=1e-5)

    def test_padding_does_not_change_real_positions(self):
        memory = self.model.encode(tf.random.uniform((1, 16, 32, 1)))
        short = self.model.decode(tf.constant([[BOS, 6, 9]]), memory).numpy()
        padded = self.model.decode(tf.constant([[BOS, 6, 9, PAD, PAD]]), memory).numpy()
        np.testing.assert_allclose(short, padded[:, :3], atol=1e-5)

    def test_greedy_decoder_handles_mixed_lengths(self):
        outputs, confidence = GreedyDecoder(self.model).decode(
            np.random.rand(3, 16, 32, 1).astype(np.float32))
        self.assertEqual(len(outputs), 3)
        for sequence, value in zip(outputs, confidence):
            if len(sequence) == self.config.max_positions - 1:
                self.assertEqual(value, -np.inf)  # never reached EOS
            self.assertLessEqual(len(sequence), self.config.max_positions - 1)
            self.assertNotIn(EOS, sequence)


if __name__ == "__main__":
    unittest.main()
