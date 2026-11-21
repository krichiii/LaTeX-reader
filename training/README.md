# LaTeX formula OCR

Reads a printed formula from an image and outputs it as Markdown math. Built with
TensorFlow and Keras, trained on the `full` config of
[linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR).

```
> python main.py predict formula.png
$$
\frac{a}{b}
$$
```

## Setup

Use the Python 3.12 environment: TensorFlow is not installed for the system Python 3.14.

```
C:/Users/krichi/lab-space-win/py3.12/Scripts/python.exe -m pip install -r requirements.txt
```

The commands below use `python` to mean that interpreter.

## Usage

| Command | What it does |
|---|---|
| `python main.py download` | Fetch the dataset into `data/` (474 MB) |
| `python main.py train` | Train into `runs/default/` |
| `python main.py evaluate` | Score the best model on the test split |
| `python main.py evaluate --multi-scale` | Score exactly what `predict` does (about 4 hours on the full split) |
| `python main.py predict a.png b.png` | Print Markdown for each image |
| `python main.py predict a.png --output a.md` | Write the Markdown to a file |
| `python main.py predict a.png --inline` | Use `$...$` instead of a `$$` block |

Run the tests with `python -m unittest tests.test_project`.

### Interrupted training

Run the same `train` command again. It continues from the last finished epoch, so
an interruption only loses the epoch in progress. Keep `--epochs` the same, because
the learning-rate schedule depends on it.

## How it works

1. **Preprocessing** (`latex_ocr/data.py`): the image becomes an ink map, whichever
   colour the background is, and is cropped to the formula. Images too large for the
   model are scaled down; small ones are never scaled up.
2. **Encoder** (`latex_ocr/model.py`): a 5-layer CNN reduces the image 8× and adds 2D
   position encodings, giving a sequence of image features.
3. **Decoder**: a 3-layer Transformer attends to those features and predicts one LaTeX
   token at a time.
4. **Multi-scale prediction** (`latex_ocr/inference.py`): the model only reads glyphs close
   to the dataset's size, and an image does not say what size its glyphs are. `predict` reads
   each image at 10 scales (×1.25 down to ×0.17) and keeps the most confident reading. A
   reading that never ends, which is typical of runaway repetition, is rejected.
5. **Output** (`latex_ocr/markdown.py`): tokens are joined into compact LaTeX and wrapped
   as Markdown math.

Training uses random scale, stroke strength, placement and noise, so the model is less
tied to the exact look of the dataset images.

### Designed for CPU

TensorFlow has no GPU support on native Windows, so the model is sized for a laptop CPU.
Measured train-step time drove these choices:

- Blank canvas is cropped away and batches group similar widths, so little time goes on
  padding.
- The CNN is narrow (24 to 192 filters). This was about 25% faster than the first design.
- There is no self-attention over the image. It cost more than it was likely to return.
- Depthwise separable convolutions were tried and were slower here, so they are not used.

## Results

The model in `runs/default` has 2.05M parameters. It trained for 10 epochs on the GPU via 
WSL2 enviroment (RTX 3050 Mobile)(due to enviroment's specifics, it can only use ~1.7GB out 
of 4GB of VRAM), which took about 4 hours.

**Test split** (9,441 images, single-scale, `runs/default/test_metrics.json`):

| Metric | Value |
|---|---|
| Edit similarity (token level) | **0.917** |
| Exact match | **40.1%** |
| Output with balanced braces | 97.4% |

| Formula length (tokens) | Images | Exact match | Edit similarity |
|---|---|---|---|
| up to 30 | 1,591 | 62.1% | 0.946 |
| 31 to 60 | 4,411 | 46.5% | 0.942 |
| 61 to 100 | 2,720 | 25.0% | 0.915 |
| over 100 | 719 | 8.8% | 0.854 |

Exact match compares tokens strictly. A prediction that renders identically but is written
differently, such as `\ldots` for `\dots` or without redundant `{ }`, counts as wrong.

**Multi-scale vs single-scale** on the first 400 test images: exact match 36.3% → 42.3%,
edit similarity 0.920 → 0.934. Trying several scales helps even on dataset images.

**Images rendered outside the dataset** (matplotlib, 5 formulas):

| Rendering | Single-scale | Multi-scale |
|---|---|---|
| Computer Modern font, 100 and 300 dpi | 1 of 10 exact, 300 dpi unreadable | 5 of 10 exact, the rest with small slips (a wrong exponent, a dropped `\pi^{2}`) |
| STIX and DejaVu Sans fonts | 0 of 10 exact | 0 of 10 exact, mostly close |

## Limitations

- It was trained on printed formulas in the dataset's LaTeX style (Computer Modern) only.
  Handwriting is out of scope, and other fonts read clearly worse: letters get confused,
  though the formula structure is usually right.
- Prediction takes about 0.3–4 s per image, because it reads every image at 10 scales.
- It expects one formula per image, not a page of text.
- Greedy decoding can produce invalid LaTeX, such as unbalanced braces, which will not
  render.
