"""Read a printed LaTeX formula from an image and output Markdown.

    python main.py download
    python main.py train
    python main.py evaluate
    python main.py predict formula.png --output formula.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"
DEFAULT_RUN = ROOT / "runs" / "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="fetch the dataset")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)

    train = commands.add_parser("train", help="train, or continue an interrupted run")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    train.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--limit", type=int, help="use only the first N rows of each split")
    train.add_argument("--d-model", type=int, help="model width")
    train.add_argument("--encoder-layers", type=int)
    train.add_argument("--decoder-layers", type=int)

    evaluate = commands.add_parser("evaluate", help="score the best model on the test split")
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--multi-scale", action="store_true",
                          help="score exactly what predict does (about 10x slower)")

    predict = commands.add_parser("predict", help="convert formula images to Markdown")
    predict.add_argument("images", type=Path, nargs="+")
    predict.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    predict.add_argument("--output", type=Path, help="write Markdown here instead of the console")
    predict.add_argument("--inline", action="store_true", help="use $...$ instead of a $$ block")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "download":
        from latex_ocr.download import download_all
        download_all(args.data_dir)

    elif args.command == "train":
        from latex_ocr.training import train
        overrides = {
            name: value
            for name, value in (
                ("d_model", args.d_model),
                ("encoder_layers", args.encoder_layers),
                ("decoder_layers", args.decoder_layers),
            )
            if value is not None
        }
        train(args.data_dir, args.run_dir, args.epochs, args.batch_size, args.learning_rate,
              args.limit, overrides)

    elif args.command == "evaluate":
        from latex_ocr.data import load_split
        from latex_ocr.download import split_path
        from latex_ocr.inference import evaluate
        samples = load_split(split_path(args.data_dir, "test"), args.limit)
        print(evaluate(args.run_dir, samples, multi_scale=args.multi_scale))

    elif args.command == "predict":
        from PIL import Image

        from latex_ocr.inference import Predictor
        predictor = Predictor(args.run_dir)
        blocks = []
        for path in args.images:
            with Image.open(path) as image:
                try:
                    blocks.append(predictor.markdown(image, inline=args.inline))
                except ValueError as error:
                    print(f"{path}: {error}", file=sys.stderr)
                    return 1
        text = "\n\n".join(blocks) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
