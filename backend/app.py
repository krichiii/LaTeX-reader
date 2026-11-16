"""Flask backend for the LaTeX formula OCR model.

    python app.py                       # http://127.0.0.1:5000
    python app.py --host 0.0.0.0 --port 8000
    flask --app app run                 # also works
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from config import Config
from routes import ApiError, api, error_body

log = logging.getLogger("latex_ocr.backend")


def create_app(service=None, **overrides) -> Flask:
    """Build the app. Pass `service` to use a ready OcrService (or a test double)."""
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config.update(overrides)

    if service is None:
        from ocr_service import OcrService
        run_dir = Path(app.config["RUN_DIR"])
        log.info("loading model from %s", run_dir)
        service = OcrService(run_dir)
        log.info("model ready in %ss", service.load_seconds)
    app.extensions["ocr"] = service

    app.register_blueprint(api)
    _register_errors(app)
    _register_cors(app)
    return app


def _register_errors(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def api_error(error: ApiError):
        return jsonify(error_body(error.code, error.message)), error.status

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        codes = {404: "not_found", 405: "method_not_allowed", 413: "upload_too_large"}
        code = codes.get(error.code, error.name.lower().replace(" ", "_"))
        message = error.description
        if error.code == 413:
            limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
            message = f"the upload is larger than {limit} MB"
        return jsonify(error_body(code, message)), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception):
        log.exception("unhandled error on %s %s", request.method, request.path)
        return jsonify(error_body("internal_error", "the server failed to process the request")), 500


def _register_cors(app: Flask) -> None:
    """Let a frontend served from another origin (e.g. a dev server) call /api."""
    setting = app.config["CORS_ORIGINS"].strip()
    allowed = {origin.strip().rstrip("/") for origin in setting.split(",") if origin.strip()}

    @app.after_request
    def add_cors_headers(response):
        if not request.path.startswith("/api"):
            return response
        origin = request.headers.get("Origin")
        if setting == "*":
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin.rstrip("/") in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers.add("Vary", "Origin")
        else:
            return response
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "600"
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LaTeX OCR model over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--run-dir", type=Path, help=f"model folder (default {Config.RUN_DIR})")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode with auto-reload")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = {"RUN_DIR": args.run_dir} if args.run_dir else {}
    app = create_app(**overrides)
    # The reloader would load the model twice, so it stays off even in debug mode.
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
