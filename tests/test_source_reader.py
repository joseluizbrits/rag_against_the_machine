"""Tests for src.source_reader.SourceReader."""

from pathlib import Path

import pytest

from src.chunking.corpus import CorpusChunker
from src.models import MinimalSource
from src.source_reader import SourceReader


def test_read_round_trips_chunk_text(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (raw / "b.md").write_bytes(
        "# Café\r\nTexto com acentuação e CRLF.\r\n".encode("utf-8")
    )
    chunks = CorpusChunker(2000).chunk_corpus(str(raw))
    assert chunks

    reader = SourceReader()
    for chunk in chunks:
        source = MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )
        recovered = reader.read(source)
        assert recovered.text == chunk.text
        assert recovered.file_path == chunk.file_path


def test_read_caches_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("Hello world", encoding="utf-8")
    reader = SourceReader()
    source = MinimalSource(
        file_path=str(raw / "a.md"),
        first_character_index=0,
        last_character_index=5,
    )

    reader.read(source)
    read_calls = []
    original_read_bytes = Path.read_bytes

    def tracking_read_bytes(self: Path) -> bytes:
        read_calls.append(self)
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)
    reader.read(source)

    assert read_calls == []


def test_read_missing_file_raises(tmp_path: Path) -> None:
    reader = SourceReader()
    source = MinimalSource(
        file_path=str(tmp_path / "missing.md"),
        first_character_index=0,
        last_character_index=5,
    )
    with pytest.raises(FileNotFoundError):
        reader.read(source)
