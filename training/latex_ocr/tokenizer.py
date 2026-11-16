"""Token <-> id mapping. Dataset labels are already space-separated LaTeX tokens."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Tokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.ids = {token: index for index, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 2) -> "Tokenizer":
        counts = Counter(token for text in texts for token in text.split())
        kept = sorted(token for token, count in counts.items() if count >= min_count)
        return cls(SPECIAL_TOKENS + kept)

    def encode(self, text: str) -> list[int]:
        return [self.ids.get(token, UNK) for token in text.split()]

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Map ids to tokens, dropping specials (there is no useful text for <unk>)."""
        first_real = len(SPECIAL_TOKENS)
        return [self.tokens[i] for i in ids if i >= first_real]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.tokens, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))
