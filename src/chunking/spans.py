"""Shared span utilities for the Python and Markdown chunkers."""

from typing import List, Protocol, Tuple

from src.models import Chunk

Span = Tuple[int, int]


class Chunker(Protocol):
    """Splits a single file's text into bounded, addressed chunks."""

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Chunk this file's text into a list of Chunks."""
        ...


def line_starts(text: str) -> List[int]:
    """Character offsets where each line begins, including offset 0."""
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def hard_split(start: int, end: int, max_chunk_size: int) -> List[Span]:
    """Cut a span into pieces of at most max_chunk_size characters."""
    spans = []
    position = start
    while position < end:
        next_position = min(position + max_chunk_size, end)
        spans.append((position, next_position))
        position = next_position
    return spans


def split_to_fit(
    start: int,
    end: int,
    max_chunk_size: int,
    starts: List[int],
) -> List[Span]:
    """Split a span on line boundaries, hard-cutting only an over-wide line."""
    breaks = [point for point in starts if start < point < end]
    points = [start, *breaks, end]
    lines = [
        (points[i], points[i + 1])
        for i in range(len(points) - 1)
        if points[i] < points[i + 1]
    ]
    pieces: List[Span] = []
    for line_start, line_end in lines:
        if line_end - line_start > max_chunk_size:
            pieces.extend(hard_split(line_start, line_end, max_chunk_size))
        else:
            pieces.append((line_start, line_end))
    return pack(pieces, max_chunk_size)


def pack(spans: List[Span], max_chunk_size: int) -> List[Span]:
    """Greedily merge adjacent, contiguous spans while they still fit."""
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start == last_end and end - last_start <= max_chunk_size:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def to_chunks(file_path: str, text: str, spans: List[Span]) -> List[Chunk]:
    """Trim whitespace from each span (moving its offsets) and drop blanks."""
    chunks = []
    for start, end in spans:
        snippet = text[start:end]
        left_trimmed = snippet.lstrip()
        leading = len(snippet) - len(left_trimmed)
        trimmed = left_trimmed.rstrip()
        if not trimmed:
            continue
        trimmed_start = start + leading
        trimmed_end = trimmed_start + len(trimmed)
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=trimmed_start,
                last_character_index=trimmed_end,
                text=trimmed,
            )
        )
    return chunks
