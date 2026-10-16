"""Header-based chunking for Markdown and plain text files."""

import re
from typing import List, Tuple

from src.chunking.spans import Span, line_starts, pack, split_to_fit, to_chunks
from src.models import Chunk

_HEADER_RE = re.compile(r"^#{1,6} ")
_FENCE_RE = re.compile(r"^(```|~~~)")


class MarkdownChunker:
    """Segments Markdown/text along header lines, outside code fences."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Raise ValueError if max_chunk_size is not positive."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Chunk Markdown/text along header-bounded, size-bounded spans."""
        starts = line_starts(text)
        lines = text.split("\n")
        section_lines = self._section_start_lines(lines)
        spans = self._section_spans(section_lines, starts, len(text))

        refined: List[Span] = []
        for start, end in spans:
            if end - start > self.max_chunk_size:
                refined.extend(self._split_section(text, start, end, starts))
            else:
                refined.append((start, end))
        return to_chunks(file_path, text, pack(refined, self.max_chunk_size))

    def _section_start_lines(self, lines: List[str]) -> List[int]:
        """1-indexed line numbers where a new header section begins."""
        section_lines = [1]
        in_fence = False
        for line_no, line in enumerate(lines, start=1):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if not in_fence and line_no != 1 and _HEADER_RE.match(line):
                section_lines.append(line_no)
        return section_lines

    def _section_spans(
        self, section_lines: List[int], starts: List[int], text_len: int
    ) -> List[Span]:
        """Convert section start lines into contiguous character spans."""
        spans = []
        for i, line_no in enumerate(section_lines):
            start = starts[line_no - 1]
            end = (
                starts[section_lines[i + 1] - 1]
                if i + 1 < len(section_lines)
                else text_len
            )
            spans.append((start, end))
        return spans

    def _split_section(
        self, text: str, start: int, end: int, starts: List[int]
    ) -> List[Span]:
        """Split an oversized section at paragraph boundaries, then lines."""
        paragraphs = self._paragraph_starts(text, start, end, starts)
        if len(paragraphs) <= 1:
            return split_to_fit(start, end, self.max_chunk_size, starts)

        spans: List[Span] = []
        for i, p_start in enumerate(paragraphs):
            p_end = paragraphs[i + 1] if i + 1 < len(paragraphs) else end
            if p_end - p_start > self.max_chunk_size:
                spans.extend(
                    split_to_fit(p_start, p_end, self.max_chunk_size, starts)
                )
            else:
                spans.append((p_start, p_end))
        return spans

    def _paragraph_starts(
        self, text: str, start: int, end: int, starts: List[int]
    ) -> List[int]:
        """Char offsets of lines that open a paragraph (after a blank line)."""
        lines = self._lines_in_range(text, start, end, starts)
        paragraph_starts = [start]
        previous_blank = False
        for offset, line_text in lines[1:]:
            if not line_text.strip():
                previous_blank = True
                continue
            if previous_blank:
                paragraph_starts.append(offset)
            previous_blank = False
        return paragraph_starts

    def _lines_in_range(
        self, text: str, start: int, end: int, starts: List[int]
    ) -> List[Tuple[int, str]]:
        """Pair each line's start offset with its text within [start, end)."""
        breaks = [point for point in starts if start <= point < end]
        points = [*breaks, end]
        return [
            (points[i], text[points[i]:points[i + 1]])
            for i in range(len(points) - 1)
        ]
