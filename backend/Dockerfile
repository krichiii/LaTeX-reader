# syntax=docker/dockerfile:1
# LaTeX Reader backend: the Flask API plus the trained model, on the CPU.
# Build from the kurs/ folder, because the backend imports ../training/latex_ocr:
#   docker build -f backend/Dockerfile -t latex-reader-backend .
# backend/Dockerfile.dockerignore keeps training/data (the 450 MB dataset) out of the build.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall TensorFlow.
# backend/requirements.txt pulls in ../training/requirements.txt, so both keep their places.
COPY training/requirements.txt training/requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt gunicorn==23.0.0

# Model code and the served run: the three files the model needs, plus its test scores for /api/model.
COPY training/latex_ocr training/latex_ocr
COPY training/runs/default/best.weights.h5 \
     training/runs/default/config.json \
     training/runs/default/vocab.json \
     training/runs/default/test_metrics.json \
     training/runs/default/
COPY backend/*.py backend/

RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /app/backend

EXPOSE 5000

# The model takes about 10 s to load; /api/health answers once it has.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4)"]

# One worker: each worker loads its own copy of the model, and OcrService already runs one
# prediction at a time with TensorFlow using every core. Threads keep /api/health and
# /api/model answering while a prediction runs.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "app:create_app()"]
