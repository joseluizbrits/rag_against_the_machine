"""Tests for src.cli.RagCli."""

from pathlib import Path

import pytest

from src.cli import RagCli
from src.indexing.lexical_index import LexicalIndex


def _raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (raw / "b.md").write_text("# Title\nSome body text.\n", encoding="utf-8")
    return raw


def test_index_happy_path(tmp_path: Path) -> None:
    raw = _raw_dir(tmp_path)
    processed = tmp_path / "processed"

    RagCli().index(
        max_chunk_size=2000, raw_dir=str(raw), processed_dir=str(processed)
    )

    index = LexicalIndex.load(str(processed))
    assert len(index.chunks) >= 2


@pytest.mark.parametrize("bad_size", [0, -1, "abc", True])
def test_index_rejects_invalid_max_chunk_size(
    tmp_path: Path, bad_size: object
) -> None:
    raw = _raw_dir(tmp_path)
    with pytest.raises(ValueError):
        RagCli().index(
            max_chunk_size=bad_size,  # type: ignore[arg-type]
            raw_dir=str(raw),
            processed_dir=str(tmp_path / "processed"),
        )


def test_index_rejects_chunk_size_above_grading_limit(tmp_path: Path) -> None:
    raw = _raw_dir(tmp_path)
    processed = tmp_path / "processed"
    with pytest.raises(ValueError, match="2000"):
        RagCli().index(
            max_chunk_size=2001, raw_dir=str(raw), processed_dir=str(processed)
        )
    assert not processed.exists()


def test_index_accepts_chunk_size_at_the_limit(tmp_path: Path) -> None:
    raw = _raw_dir(tmp_path)
    processed = tmp_path / "processed"
    RagCli().index(
        max_chunk_size=2000, raw_dir=str(raw), processed_dir=str(processed)
    )
    assert LexicalIndex.load(str(processed)).chunks


def test_index_missing_raw_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RagCli().index(
            raw_dir=str(tmp_path / "does_not_exist"),
            processed_dir=str(tmp_path / "processed"),
        )


def test_index_empty_raw_dir_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError):
        RagCli().index(
            raw_dir=str(raw), processed_dir=str(tmp_path / "processed")
        )
