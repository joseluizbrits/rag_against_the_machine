"""Tests for src.chunking.markdown_chunker.MarkdownChunker."""

import pytest

from src.chunking.markdown_chunker import MarkdownChunker


def _section(title: str, body_lines: int) -> str:
    body = "".join(f"line {i} of {title}\n" for i in range(body_lines))
    return f"# {title}\n{body}"


def test_one_chunk_per_section_between_half_and_full_bound() -> None:
    text = _section("One", 4) + _section("Two", 4) + _section("Three", 4)
    chunker = MarkdownChunker(max_chunk_size=100)
    assert all(
        100 / 2 < len(s) <= 100
        for s in (_section("One", 4), _section("Two", 4), _section("Three", 4))
    )
    chunks = chunker.chunk("f.md", text)
    assert len(chunks) == 3
    for chunk, title in zip(chunks, ["One", "Two", "Three"]):
        assert chunk.text.startswith(f"# {title}")


def test_hash_inside_fence_is_not_a_header() -> None:
    text = (
        "# Real Header\n"
        "some text\n"
        "```\n"
        "# not a header\n"
        "```\n"
        "more text\n"
    )
    chunks = MarkdownChunker(max_chunk_size=2000).chunk("f.md", text)
    assert len(chunks) == 1
    assert "# not a header" in chunks[0].text


def test_oversized_section_split_within_bound() -> None:
    text = _section("Big", 60)
    assert len(text) > 500
    chunks = MarkdownChunker(max_chunk_size=200).chunk("f.md", text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.last_character_index - chunk.first_character_index <= 200


def test_header_less_text_is_chunked() -> None:
    text = "".join(f"plain line {i}\n" for i in range(50))
    chunks = MarkdownChunker(max_chunk_size=200).chunk("f.txt", text)
    assert chunks
    for chunk in chunks:
        assert chunk.last_character_index - chunk.first_character_index <= 200


def test_offsets_are_exact() -> None:
    text = _section("One", 3) + _section("Two", 3)
    chunks = MarkdownChunker(max_chunk_size=2000).chunk("f.md", text)
    assert chunks
    for chunk in chunks:
        span = text[chunk.first_character_index:chunk.last_character_index]
        assert span == chunk.text


def test_invalid_max_chunk_size_rejected() -> None:
    with pytest.raises(ValueError):
        MarkdownChunker(max_chunk_size=0)
    with pytest.raises(ValueError):
        MarkdownChunker(max_chunk_size=-5)
