"""Sequence-level quality metrics, computed on LaTeX tokens."""

from __future__ import annotations


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    return previous[-1]


def score(predictions: list[list[str]], references: list[list[str]]) -> dict[str, float]:
    """Return exact-match rate and edit similarity (1.0 is a perfect transcription)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references differ in length")
    exact = 0
    distance = 0
    length = 0
    for prediction, reference in zip(predictions, references):
        exact += prediction == reference
        distance += edit_distance(prediction, reference)
        length += max(len(prediction), len(reference))
    return {
        "exact_match": exact / len(references),
        "edit_similarity": 1.0 - distance / max(length, 1),
    }
