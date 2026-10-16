"""Tests for src.chunking.spans helpers."""

from src.chunking.spans import (
    hard_split,
    line_starts,
    pack,
    split_to_fit,
    to_chunks,
)


def test_line_starts_includes_zero_and_each_newline() -> None:
    text = "ab\ncd\n\nef"
    assert line_starts(text) == [0, 3, 6, 7]


def test_pack_merges_while_within_bound() -> None:
    spans = [(0, 5), (5, 10), (10, 15)]
    assert pack(spans, max_chunk_size=15) == [(0, 15)]


def test_pack_stops_exactly_at_bound() -> None:
    spans = [(0, 5), (5, 10), (10, 11)]
    assert pack(spans, max_chunk_size=10) == [(0, 10), (10, 11)]


def test_pack_does_not_merge_non_contiguous_spans() -> None:
    spans = [(0, 5), (6, 10)]
    assert pack(spans, max_chunk_size=100) == [(0, 5), (6, 10)]


def test_hard_split_cuts_every_max_chunk_size_characters() -> None:
    expected = [(0, 10), (10, 20), (20, 25)]
    assert hard_split(0, 25, max_chunk_size=10) == expected


def test_split_to_fit_keeps_short_line_whole() -> None:
    text = "short line\n"
    starts = line_starts(text)
    assert split_to_fit(0, len(text), max_chunk_size=100, starts=starts) == [
        (0, len(text))
    ]


def test_split_to_fit_hard_cuts_single_over_wide_line() -> None:
    text = "x" * 25
    starts = line_starts(text)
    spans = split_to_fit(0, len(text), max_chunk_size=10, starts=starts)
    assert spans == [(0, 10), (10, 20), (20, 25)]


def test_to_chunks_trims_whitespace_by_moving_offsets() -> None:
    text = "  hello world  "
    chunks = to_chunks("f.py", text, [(0, len(text))])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "hello world"
    assert chunk.first_character_index == 2
    assert chunk.last_character_index == 13
    span = text[chunk.first_character_index:chunk.last_character_index]
    assert span == chunk.text


def test_to_chunks_drops_blank_spans() -> None:
    text = "   \n\n   "
    assert to_chunks("f.py", text, [(0, len(text))]) == []
