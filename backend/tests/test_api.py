"""API tests.

`ApiTests` use a stand-in for the model and run in a second. `ModelTests` load
the real model; set LATEX_OCR_SKIP_MODEL_TESTS=1 to skip them.

    python -m unittest tests.test_api
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image, ImageDraw  # noqa: E402

from app import create_app  # noqa: E402
from config import Config, TRAINING_DIR  # noqa: E402
from ocr_service import NoContentError, Reading, braces_balanced  # noqa: E402

DEMO_IMAGES = TRAINING_DIR / "runs" / "default" / "demo" / "images"


def png_bytes(blank: bool = False, size: tuple[int, int] = (120, 40)) -> bytes:
    image = Image.new("RGB", size, "white")
    if not blank:
        ImageDraw.Draw(image).rectangle((20, 10, 100, 30), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeService:
    """Returns a fixed reading; reports blank images like the real service."""

    def __init__(self):
        self.calls = []

    def read(self, image, multi_scale=True):
        self.calls.append({"size": image.size, "multi_scale": multi_scale})
        if image.convert("L").getextrema()[0] == 255:
            raise NoContentError("image has no visible content")
        return Reading(latex="E=mc^{2}", markdown="$$\nE=mc^{2}\n$$", markdown_inline="$E=mc^{2}$",
                       confidence=0.9, complete=True, balanced=True, multi_scale=multi_scale,
                       elapsed_ms=1)

    def info(self):
        return {"run": "fake", "device": "default"}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.app = create_app(service=self.service, TESTING=True)
        self.client = self.app.test_client()

    def post_file(self, data: bytes, url="/api/predict", **form):
        return self.client.post(url, data={"image": (io.BytesIO(data), "f.png"), **form},
                                content_type="multipart/form-data")

    def assert_error(self, response, status: int, code: str):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["error"]["code"], code)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok", "model_loaded": True})

    def test_model_info(self):
        self.assertEqual(self.client.get("/api/model").get_json()["run"], "fake")

    def test_predict_multipart(self):
        response = self.post_file(png_bytes())
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["latex"], "E=mc^{2}")
        self.assertEqual(body["markdown"], "$$\nE=mc^{2}\n$$")
        self.assertEqual(body["markdown_inline"], "$E=mc^{2}$")
        self.assertTrue(body["multi_scale"])
        self.assertEqual(self.service.calls[0]["size"], (120, 40))

    def test_predict_json_data_url(self):
        encoded = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
        response = self.client.post("/api/predict", json={"image": encoded, "multi_scale": False})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["multi_scale"])

    def test_predict_json_plain_base64(self):
        encoded = base64.b64encode(png_bytes()).decode()
        self.assertEqual(self.client.post("/api/predict", json={"image": encoded}).status_code, 200)

    def test_multi_scale_form_option(self):
        self.post_file(png_bytes(), multi_scale="false")
        self.assertFalse(self.service.calls[0]["multi_scale"])

    def test_invalid_option(self):
        self.assert_error(self.post_file(png_bytes(), multi_scale="maybe"), 400, "invalid_option")

    def test_missing_image(self):
        self.assert_error(self.client.post("/api/predict", data={}), 400, "no_image")
        self.assert_error(self.client.post("/api/predict", json={}), 400, "no_image")

    def test_not_an_image(self):
        self.assert_error(self.post_file(b"hello, this is text"), 400, "invalid_image")
        self.assert_error(self.post_file(b""), 400, "invalid_image")

    def test_bad_base64(self):
        response = self.client.post("/api/predict", json={"image": "not base64!!"})
        self.assert_error(response, 400, "invalid_image")

    def test_json_not_an_object(self):
        response = self.client.post("/api/predict", data="[1, 2]", content_type="application/json")
        self.assert_error(response, 400, "invalid_json")

    def test_blank_image(self):
        self.assert_error(self.post_file(png_bytes(blank=True)), 422, "no_content")

    def test_too_many_pixels(self):
        self.app.config["MAX_IMAGE_PIXELS"] = 1000
        self.assert_error(self.post_file(png_bytes()), 413, "image_too_large")
        self.assertEqual(self.service.calls, [])

    def test_upload_too_large(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
        response = self.post_file(b"\0" * (2 * 1024 * 1024))
        self.assert_error(response, 413, "upload_too_large")

    def test_batch_multipart_reports_each_image(self):
        files = [(io.BytesIO(png_bytes()), "a.png"), (io.BytesIO(png_bytes(blank=True)), "b.png"),
                 (io.BytesIO(b"junk"), "c.txt")]
        response = self.client.post("/api/predict/batch", data={"images": files},
                                    content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        results = response.get_json()["results"]
        self.assertEqual([r["filename"] for r in results], ["a.png", "b.png", "c.txt"])
        self.assertEqual([r["ok"] for r in results], [True, False, False])
        self.assertEqual(results[0]["latex"], "E=mc^{2}")
        self.assertEqual(results[1]["error"]["code"], "no_content")
        self.assertEqual(results[2]["error"]["code"], "invalid_image")

    def test_batch_json(self):
        encoded = base64.b64encode(png_bytes()).decode()
        response = self.client.post("/api/predict/batch", json={"images": [encoded, encoded]})
        self.assertEqual([r["index"] for r in response.get_json()["results"]], [0, 1])

    def test_batch_limits(self):
        self.assert_error(self.client.post("/api/predict/batch", json={"images": []}), 400, "no_image")
        self.app.config["MAX_BATCH_SIZE"] = 2
        encoded = base64.b64encode(png_bytes()).decode()
        response = self.client.post("/api/predict/batch", json={"images": [encoded] * 3})
        self.assert_error(response, 400, "too_many_images")

    def test_unknown_route_and_method(self):
        self.assert_error(self.client.get("/api/nope"), 404, "not_found")
        self.assert_error(self.client.get("/api/predict"), 405, "method_not_allowed")

    def test_internal_error_is_json(self):
        def explode(image, multi_scale=True):
            raise RuntimeError("boom")
        self.service.read = explode
        self.app.config["TESTING"] = False  # otherwise Flask re-raises
        with self.assertLogs("latex_ocr.backend", "ERROR"):
            self.assert_error(self.post_file(png_bytes()), 500, "internal_error")

    def test_cors_any_origin(self):
        response = self.client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_cors_preflight(self):
        response = self.client.options("/api/predict", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("POST", response.headers["Access-Control-Allow-Methods"])

    def test_cors_allow_list(self):
        app = create_app(service=self.service, TESTING=True, CORS_ORIGINS="http://localhost:5173")
        client = app.test_client()
        allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(allowed.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        other = client.get("/api/health", headers={"Origin": "http://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", other.headers)

    def test_braces_balanced(self):
        self.assertTrue(braces_balanced("\\frac { a } { b }".split()))
        self.assertTrue(braces_balanced("\\{ x \\}".split()))
        self.assertFalse(braces_balanced("\\frac { a } { b".split()))
        self.assertFalse(braces_balanced("} {".split()))


@unittest.skipIf(os.environ.get("LATEX_OCR_SKIP_MODEL_TESTS"), "LATEX_OCR_SKIP_MODEL_TESTS is set")
class ModelTests(unittest.TestCase):
    """End-to-end through Flask with the model in Config.RUN_DIR."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app(TESTING=True)
        cls.client = cls.app.test_client()

    def predict(self, name: str, **form):
        with open(DEMO_IMAGES / name, "rb") as file:
            return self.client.post("/api/predict", data={"image": (file, name), **form},
                                    content_type="multipart/form-data")

    def test_model_info(self):
        info = self.client.get("/api/model").get_json()
        self.assertEqual(info["run"], Config.RUN_DIR.name)
        self.assertEqual(info["parameters"], 2_052_731)
        self.assertIn(info["device"], ("default", "GPU"))

    def test_reads_formula(self):
        body = self.predict("euler_cm_100.png").get_json()
        self.assertEqual(body["latex"], "e^{i\\pi}+1=0")
        self.assertEqual(body["markdown"], "$$\ne^{i\\pi}+1=0\n$$")
        self.assertTrue(body["complete"])
        self.assertTrue(body["balanced"])
        self.assertGreater(body["confidence"], 0.5)

    def test_single_scale_reads_dataset_image(self):
        body = self.predict("dataset_4.png", multi_scale="false").get_json()
        self.assertEqual(body["latex"], "A_{\\mu}^{n}=\\frac{1}{e}\\partial_{\\mu}\\phi M_{n}a(\\rho)")

    def test_dark_mode_matches_light(self):
        light = self.predict("quadratic_cm_100.png").get_json()["latex"]
        dark = self.predict("quadratic_cm_100_dark.png").get_json()["latex"]
        self.assertEqual(light, dark)

    def test_blank_image(self):
        response = self.predict("blank.png")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "no_content")


if __name__ == "__main__":
    unittest.main()
