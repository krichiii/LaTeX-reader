"""Turn predicted LaTeX tokens into Markdown math."""

from __future__ import annotations

import re

_CONTROL_WORD = re.compile(r"\\[A-Za-z]+")


def tokens_to_latex(tokens: list[str]) -> str:
    r"""Join tokens compactly without changing what the LaTeX means.

    A space is only needed in two cases: after a control word followed by a
    letter (``\alpha x`` must not become ``\alphax``), and after the ``\``
    (control space) and ``\\`` (new row) tokens, where it keeps the row break
    from swallowing a following ``[`` as an optional argument.
    """
    parts: list[str] = []
    for index, token in enumerate(tokens):
        parts.append(token)
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if not following:
            break
        if token in ("\\", "\\\\"):
            parts.append(" ")
        elif _CONTROL_WORD.fullmatch(token) and following[0].isalpha():
            parts.append(" ")
    return "".join(parts)


def to_markdown(latex: str, inline: bool = False) -> str:
    """Wrap LaTeX as a Markdown math block, or as inline math."""
    if inline:
        return f"${latex}$"
    return f"$$\n{latex}\n$$"
