"""Tests for src.cli.RagCli."""

import json
from pathlib import Path

import pytest

from src.cli import RagCli
from src.indexing.lexical_index import LexicalIndex
from src.models import StudentSearchResults


def _raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (raw / "b.md").write_text(
        "# Title\nThis document explains the lora adapter technique.\n",
        encoding="utf-8",
    )
    (raw / "c.md").write_text(
        "# Quantization\nThis document is about quantization instead.\n",
        encoding="utf-8",
    )
    (raw / "d.md").write_text(
        "# Decoding\nThis document covers speculative decoding topics.\n",
        encoding="utf-8",
    )
    return raw


def _indexed_processed_dir(tmp_path: Path) -> Path:
    raw = _raw_dir(tmp_path)
    processed = tmp_path / "processed"
    RagCli().index(
        max_chunk_size=2000, raw_dir=str(raw), processed_dir=str(processed)
    )
    return processed


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


def test_search_prints_ranked_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    capsys.readouterr()

    RagCli().search("lora adapter", k=3, processed_dir=str(processed))

    out = capsys.readouterr().out
    assert "b.md" in out
    assert "[" in out and "]" in out


def test_search_no_results_message(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    processed = _indexed_processed_dir(tmp_path)

    RagCli().search("qwzxv plorkt", k=3, processed_dir=str(processed))

    assert "No results found" in capsys.readouterr().out


@pytest.mark.parametrize("bad_query", ["", "   "])
def test_search_rejects_blank_query(tmp_path: Path, bad_query: str) -> None:
    processed = _indexed_processed_dir(tmp_path)
    with pytest.raises(ValueError):
        RagCli().search(bad_query, k=3, processed_dir=str(processed))


@pytest.mark.parametrize("bad_k", [0, "abc"])
def test_search_rejects_invalid_k(tmp_path: Path, bad_k: object) -> None:
    processed = _indexed_processed_dir(tmp_path)
    with pytest.raises(ValueError):
        RagCli().search(
            "lora",
            k=bad_k,  # type: ignore[arg-type]
            processed_dir=str(processed),
        )


def test_search_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RagCli().search("lora", k=3, processed_dir=str(tmp_path / "missing"))


def _dataset_path(tmp_path: Path, questions: list) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"rag_questions": questions}), encoding="utf-8")
    return path


def test_search_dataset_writes_results_file(tmp_path: Path) -> None:
    processed = _indexed_processed_dir(tmp_path)
    dataset_path = _dataset_path(
        tmp_path,
        [
            {"question_id": "q1", "question": "lora adapter"},
            {"question_id": "q2", "question": "qwzxv plorkt"},
        ],
    )
    save_directory = tmp_path / "output"

    RagCli().search_dataset(
        dataset_path=str(dataset_path),
        k=5,
        save_directory=str(save_directory),
        processed_dir=str(processed),
    )

    saved_path = save_directory / "dataset.json"
    assert saved_path.exists()
    results = StudentSearchResults.model_validate_json(
        saved_path.read_text(encoding="utf-8")
    )
    assert results.k == 5
    by_id = {entry.question_id: entry for entry in results.search_results}
    assert by_id["q1"].question == "lora adapter"
    assert len(by_id["q1"].retrieved_sources) >= 1
    assert all(
        not hasattr(source, "text") for source in by_id["q1"].retrieved_sources
    )
    assert by_id["q2"].retrieved_sources == []


def test_search_dataset_missing_dataset_raises_without_writing(
    tmp_path: Path,
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    save_directory = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        RagCli().search_dataset(
            dataset_path=str(tmp_path / "missing.json"),
            save_directory=str(save_directory),
            processed_dir=str(processed),
        )
    assert not save_directory.exists()


def test_search_dataset_malformed_dataset_raises_without_writing(
    tmp_path: Path,
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    save_directory = tmp_path / "output"
    bad_dataset = tmp_path / "bad.json"
    bad_dataset.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError):
        RagCli().search_dataset(
            dataset_path=str(bad_dataset),
            save_directory=str(save_directory),
            processed_dir=str(processed),
        )
    assert not save_directory.exists()
