"""Identifier-aware tokenization shared by indexing and retrieval."""

import re
from typing import List

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

STOPWORDS = frozenset(
    """
    a an the of to in on for and or is are was be by with what how which
    does do can when where why this that it as from at its into
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase identifier tokens, split into parts, stopwords dropped."""
    tokens: List[str] = []
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        tokens.append(lowered)
        parts = [part.lower() for part in _PART_RE.findall(word)]
        if len(parts) > 1:
            tokens.extend(parts)
    return [token for token in tokens if token not in STOPWORDS]
