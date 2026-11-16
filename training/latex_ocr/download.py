"""Download the `full` config of linxy/LaTeX_OCR from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

import requests

BASE_URL = "https://huggingface.co/datasets/linxy/LaTeX_OCR/resolve/main/full"
SPLITS = ("train", "validation", "test")


def split_path(data_dir: Path, split: str) -> Path:
    return data_dir / f"{split}.parquet"


def download_split(data_dir: Path, split: str) -> Path:
    """Fetch one split; the rename at the end means a partial file is never used."""
    target = split_path(data_dir, split)
    if target.exists():
        print(f"{target} already exists, skipping")
        return target

    url = f"{BASE_URL}/{split}-00000-of-00001.parquet"
    partial = target.with_suffix(".parquet.part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with open(partial, "wb") as output:
            for chunk in response.iter_content(chunk_size=1 << 20):
                output.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{split}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
    print()
    if total and done != total:
        partial.unlink()
        raise IOError(f"{split}: got {done} bytes, expected {total}")
    os.replace(partial, target)
    return target


def download_all(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        download_split(data_dir, split)
