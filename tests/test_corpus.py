"""Tests for src.chunking.corpus.CorpusChunker."""

from pathlib import Path

import pytest

from src.chunking.corpus import CorpusChunker


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_only_supported_extensions_are_chunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    raw = Path("raw")
    _write(raw / "a.py", "x = 1\n")
    _write(raw / "b.md", "# Title\ntext\n")
    _write(raw / "c.txt", "plain text\n")
    _write(raw / "d.json", '{"a": 1}\n')

    chunks = CorpusChunker(max_chunk_size=2000).chunk_corpus("raw")
    paths = {chunk.file_path for chunk in chunks}
    assert any(p.endswith("a.py") for p in paths)
    assert any(p.endswith("b.md") for p in paths)
    assert any(p.endswith("c.txt") for p in paths)
    assert not any(p.endswith("d.json") for p in paths)


def test_undecodable_file_is_skipped_others_still_chunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    raw = Path("raw")
    raw.mkdir()
    (raw / "bad.py").write_bytes(b"\xff\xfe not utf-8")
    _write(raw / "good.py", "x = 1\n")

    chunks = CorpusChunker(max_chunk_size=2000).chunk_corpus("raw")
    paths = {chunk.file_path for chunk in chunks}
    assert any(p.endswith("good.py") for p in paths)
    assert not any(p.endswith("bad.py") for p in paths)


def test_file_paths_are_relative_with_forward_slashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(Path("raw") / "nested" / "doc.md", "# Title\nbody text\n")

    chunks = CorpusChunker(max_chunk_size=2000).chunk_corpus("raw")
    assert chunks
    for chunk in chunks:
        assert "\\" not in chunk.file_path
        assert chunk.file_path.startswith("raw/")


def test_invalid_max_chunk_size_rejected_before_reading(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        CorpusChunker(max_chunk_size=0)
