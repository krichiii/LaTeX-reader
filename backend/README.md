# LaTeX OCR backend

A Flask API that serves the trained formula reader: an image of a printed formula goes
in, and LaTeX and Markdown come out. It is meant to be called by the frontend.

The model code is not copied here. The backend imports `latex_ocr` from `../training`
and loads the GPU-trained run `training/runs_wsl/train16`.

## Run

Use the Python 3.12 environment, which already has Flask and TensorFlow:

```
cd backend
C:/Users/krichi/lab-space-win/py3.12/Scripts/python.exe app.py
```

The server listens on `http://127.0.0.1:5000` and loads the model at startup, which
takes about 10 seconds. Options:

| Option | Default | |
|---|---|---|
| `--host` | `127.0.0.1` | `0.0.0.0` makes it reachable from other devices |
| `--port` | `5000` | |
| `--run-dir` | `training/runs_wsl/train16` | serve another training run, e.g. `../training/runs/default` |
| `--debug` | off | Flask debug mode |

`flask --app app run` works too.

**The GPU is not needed to serve the model.** "GPU-trained" only describes how the
weights were made. On native Windows TensorFlow runs on the CPU, and a prediction takes
about 0.2–3 s. To predict on the GPU, run the backend inside WSL with the venv activated
(`source ~/venvs/data/bin/activate`). Flask is not installed in that venv yet
(`pip install flask`). `GET /api/model` reports which device is in use.

### Settings

These environment variables override the defaults in `config.py`:

| Variable | Default | |
|---|---|---|
| `LATEX_OCR_RUN_DIR` | `training/runs_wsl/train16` | model folder |
| `LATEX_OCR_CORS_ORIGINS` | `*` | allowed frontend origins, comma separated, e.g. `http://localhost:5173` |
| `LATEX_OCR_MAX_UPLOAD_MB` | `10` | larger requests get 413 |
| `LATEX_OCR_MAX_IMAGE_PIXELS` | `25000000` | larger images get 413 |
| `LATEX_OCR_MAX_BATCH` | `16` | images per batch request |

## API

Every response is JSON, including errors. All endpoints are under `/api`.

### `POST /api/predict`

Reads one image. Send it either way:

- **multipart/form-data**: a file field `image`, plus an optional field `multi_scale`.
- **JSON**: `{"image": "<base64>", "multi_scale": true}`. The base64 string may be a data
  URL (`data:image/png;base64,...`), as returned by `FileReader.readAsDataURL` or
  `canvas.toDataURL`, which is handy for pasted screenshots.

`multi_scale` defaults to `true`, which reads the image at 10 scales and keeps the most
confident result. This handles formulas of any size. `false` is faster but only suits
images whose glyphs are at the dataset's size.

```json
{
  "latex": "e^{i\\pi}+1=0",
  "markdown": "$$\ne^{i\\pi}+1=0\n$$",
  "markdown_inline": "$e^{i\\pi}+1=0$",
  "confidence": 0.8887,
  "complete": true,
  "balanced": true,
  "multi_scale": true,
  "elapsed_ms": 209
}
```

| Field | Meaning |
|---|---|
| `latex` | the formula, without delimiters |
| `markdown` / `markdown_inline` | the same formula wrapped as `$$...$$` or `$...$` |
| `confidence` | 0 to 1, the geometric mean of the probabilities of the chosen tokens; `null` when `complete` is false |
| `complete` | `false` when the model never ended the formula (usually runaway repetition), so the result is unreliable |
| `balanced` | `false` when `{` and `}` do not pair up; such LaTeX will not render |

Confidence is a rough signal, not a calibrated probability. On the 27 demo images, every
reading below 0.75 was wrong, but so were many above it.

### `POST /api/predict/batch`

Reads up to 16 images. Send the multipart file fields `images` (repeated), or JSON
`{"images": ["<base64>", ...], "multi_scale": true}`. One bad image does not fail the
request; its entry has `ok: false` and an `error`:

```json
{
  "results": [
    {"index": 0, "filename": "a.png", "ok": true, "latex": "E=mc^{2}", "...": "..."},
    {"index": 1, "filename": "b.png", "ok": false,
     "error": {"code": "no_content", "message": "the image is blank, there is no formula to read"}}
  ]
}
```

`filename` is `null` for JSON requests.

### `GET /api/model`

Describes the loaded model: run name, parameter count, architecture, test metrics,
device and load time.

### `GET /api/health`

`{"status": "ok", "model_loaded": true}`. The server only starts listening once the
model is loaded.

### Errors

```json
{"error": {"code": "no_content", "message": "the image is blank, there is no formula to read"}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `no_image` | no `image` field or key, or an empty batch |
| 400 | `invalid_image` | not a readable image, or invalid base64 |
| 400 | `invalid_json` | the JSON body is not an object |
| 400 | `invalid_option` | `multi_scale` is not true or false |
| 400 | `too_many_images` | the batch is over the limit |
| 413 | `upload_too_large` | the request is over 10 MB |
| 413 | `image_too_large` | the image is over 25 megapixels |
| 422 | `no_content` | the image is blank |
| 404 / 405 | `not_found` / `method_not_allowed` | wrong URL or method |
| 500 | `internal_error` | anything else; the server log has the traceback |

### Example calls

```
curl -F "image=@formula.png" http://127.0.0.1:5000/api/predict
curl -F "images=@a.png" -F "images=@b.png" http://127.0.0.1:5000/api/predict/batch
```

```js
const form = new FormData();
form.append("image", file);  // a File from <input type="file"> or a drop event
const response = await fetch("http://127.0.0.1:5000/api/predict", { method: "POST", body: form });
const result = await response.json();
if (!response.ok) throw new Error(result.error.message);
console.log(result.latex);
```

## Tests

```
C:/Users/krichi/lab-space-win/py3.12/Scripts/python.exe -m unittest tests.test_api
```

There are 28 tests. Most use a stand-in for the model and take under a second. Five
load the real model and read demo images from `training/runs/default/demo/images`. Set
`LATEX_OCR_SKIP_MODEL_TESTS=1` to skip those five.

## Files

| File | |
|---|---|
| `app.py` | app factory, JSON error handling, CORS, command line |
| `routes.py` | the `/api` endpoints and upload parsing |
| `ocr_service.py` | loads the run and reads images; the only part that touches TensorFlow |
| `config.py` | settings and their environment variables |

## Which model

| Run | Trained on | Test exact match | Test edit similarity |
|---|---|---|---|
| `training/runs_wsl/train16` (served) | RTX 3050, batch 16 | **45.4%** | **0.932** |
| `training/runs/default` | CPU, batch 32 | 40.1% | 0.917 |

On the test split the GPU run is clearly better. On the 10 Computer Modern formulas
rendered with matplotlib in `demo/images`, it was worse with multi-scale reading: 2
exact against 5 for the CPU run, with errors such as `{\cal E}` for `E`. That sample is
tiny, but if real screenshots read badly, compare with `--run-dir ../training/runs/default`.
