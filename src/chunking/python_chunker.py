"""AST-based chunking for Python source files."""

import ast
from typing import List

from src.chunking.spans import Span, line_starts, pack, split_to_fit, to_chunks
from src.models import Chunk


def _first_line(stmt: ast.stmt) -> int:
    """The line a statement starts on, including any decorator."""
    decorators = getattr(stmt, "decorator_list", None)
    if decorators:
        line: int = decorators[0].lineno
        return line
    return stmt.lineno


def _statement_span(stmt: ast.stmt, starts: List[int], text_len: int) -> Span:
    """The char span covering a statement, from its first decorator line."""
    start = starts[_first_line(stmt) - 1]
    end_line = stmt.end_lineno if stmt.end_lineno is not None else stmt.lineno
    end = starts[end_line] if end_line < len(starts) else text_len
    return (start, end)


class PythonChunker:
    """Segments Python source along top-level function and class boundaries."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Raise ValueError if max_chunk_size is not positive."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Chunk Python source; fall back to line splits if it won't parse."""
        starts = line_starts(text)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            spans = split_to_fit(0, len(text), self.max_chunk_size, starts)
            return to_chunks(file_path, text, spans)

        spans = self._segment(tree.body, text, starts)
        return to_chunks(file_path, text, pack(spans, self.max_chunk_size))

    def _segment(
        self, stmts: List[ast.stmt], text: str, starts: List[int]
    ) -> List[Span]:
        """Build a span per statement, refining any that don't fit."""
        spans: List[Span] = []
        for stmt in stmts:
            span = _statement_span(stmt, starts, len(text))
            width = span[1] - span[0]
            if width <= self.max_chunk_size:
                spans.append(span)
            elif isinstance(stmt, ast.ClassDef):
                spans.extend(self._segment_class(stmt, span, text, starts))
            else:
                spans.extend(
                    split_to_fit(span[0], span[1], self.max_chunk_size, starts)
                )
        return spans

    def _segment_class(
        self, stmt: ast.ClassDef, span: Span, text: str, starts: List[int]
    ) -> List[Span]:
        """Split an oversized class into a header span and its member spans."""
        header_end = starts[_first_line(stmt.body[0]) - 1]
        header_start, header_end_pos = span[0], header_end
        if header_end_pos - header_start > self.max_chunk_size:
            header_spans = split_to_fit(
                header_start, header_end_pos, self.max_chunk_size, starts
            )
        else:
            header_spans = [(header_start, header_end_pos)]
        return [*header_spans, *self._segment(stmt.body, text, starts)]
