"""HTTP endpoints under /api. Every response, including errors, is JSON."""

from __future__ import annotations

import base64
import binascii
import io
import re

from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError

from ocr_service import NoContentError

api = Blueprint("api", __name__, url_prefix="/api")

_DATA_URL = re.compile(r"^data:[\w.+-]+/[\w.+-]+;base64,", re.IGNORECASE)
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def service():
    return current_app.extensions["ocr"]


def parse_bool(value, name: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ApiError(400, "invalid_option", f"'{name}' must be true or false")


def decode_base64(text: str) -> bytes:
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, "invalid_image", "the image must be a base64 string or a data URL")
    payload = _DATA_URL.sub("", text.strip(), count=1)
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "invalid_image", "the image is not valid base64") from None


def open_image(data: bytes) -> Image.Image:
    if not data:
        raise ApiError(400, "invalid_image", "the image is empty")
    try:
        image = Image.open(io.BytesIO(data))
        # Checked before decoding, so an oversized image never reaches memory.
        if image.width * image.height > current_app.config["MAX_IMAGE_PIXELS"]:
            raise ApiError(413, "image_too_large",
                           f"the image is {image.width}x{image.height} pixels, which is too large")
        image.load()
    except ApiError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise ApiError(400, "invalid_image", "the file is not an image that can be read") from None
    return image


def read_one(data: bytes, multi_scale: bool) -> dict:
    image = open_image(data)
    try:
        return service().read(image, multi_scale=multi_scale).to_dict()
    except NoContentError:
        raise ApiError(422, "no_content", "the image is blank, there is no formula to read") from None
    finally:
        image.close()


def json_body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_json", "the request body must be a JSON object")
    return body


@api.get("/health")
def health():
    return jsonify(status="ok", model_loaded=True)


@api.get("/model")
def model_info():
    return jsonify(service().info())


@api.post("/predict")
def predict():
    """One image: a multipart file field `image`, or JSON {"image": "<base64>"}."""
    if request.is_json:
        body = json_body()
        if "image" not in body:
            raise ApiError(400, "no_image", "send the image as the 'image' key")
        data = decode_base64(body["image"])
        multi_scale = parse_bool(body.get("multi_scale"), "multi_scale", True)
    else:
        upload = request.files.get("image")
        if upload is None:
            raise ApiError(400, "no_image", "send the image as the multipart file field 'image'")
        data = upload.read()
        multi_scale = parse_bool(request.form.get("multi_scale"), "multi_scale", True)
    return jsonify(read_one(data, multi_scale))


@api.post("/predict/batch")
def predict_batch():
    """Several images: multipart file fields `images`, or JSON {"images": [...]}.

    A bad image does not fail the request: its entry carries an error instead.
    """
    if request.is_json:
        body = json_body()
        items = body.get("images")
        if not isinstance(items, list):
            raise ApiError(400, "no_image", "send the images as the 'images' list")
        multi_scale = parse_bool(body.get("multi_scale"), "multi_scale", True)
        names = [None] * len(items)
    else:
        uploads = request.files.getlist("images")
        items = uploads
        multi_scale = parse_bool(request.form.get("multi_scale"), "multi_scale", True)
        names = [upload.filename for upload in uploads]

    if not items:
        raise ApiError(400, "no_image", "send at least one image")
    limit = current_app.config["MAX_BATCH_SIZE"]
    if len(items) > limit:
        raise ApiError(400, "too_many_images", f"send at most {limit} images per request")

    results = []
    for index, (item, name) in enumerate(zip(items, names)):
        entry = {"index": index, "filename": name}
        try:
            data = decode_base64(item) if request.is_json else item.read()
            entry.update(ok=True, **read_one(data, multi_scale))
        except ApiError as error:
            entry.update(ok=False, **error_body(error.code, error.message))
        results.append(entry)
    return jsonify(results=results)
