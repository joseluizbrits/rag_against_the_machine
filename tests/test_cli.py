"""Tests for src.cli.RagCli."""

import json
from pathlib import Path
from typing import List

import pytest

from src.cli import RagCli
from src.generation import NO_CONTEXT_ANSWER, Answer, AnswerGenerator
from src.indexing.lexical_index import LexicalIndex
from src.models import (
    Chunk,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class _FakeAnswerGenerator:
    """A generator stub that echoes a fixed answer and records its calls."""

    def __init__(
        self, response_text: str = "This is a generated answer."
    ) -> None:
        """Remember the answer text to return, with no recorded calls yet."""
        self.response_text = response_text
        self.questions: List[str] = []

    def answer(self, question: str, chunks: List[Chunk]) -> Answer:
        """Record question, return response_text or NO_CONTEXT_ANSWER."""
        self.questions.append(question)
        if not chunks:
            return Answer(text=NO_CONTEXT_ANSWER, used_chunks=[])
        return Answer(text=self.response_text, used_chunks=chunks)


def _patch_generator_load(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeAnswerGenerator
) -> None:
    monkeypatch.setattr(
        AnswerGenerator, "load", classmethod(lambda cls, name: fake)
    )


def _patch_generator_load_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(cls: type, name: str) -> AnswerGenerator:
        raise AssertionError("AnswerGenerator.load must not be called")

    monkeypatch.setattr(AnswerGenerator, "load", classmethod(_fail))


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


def test_answer_prints_answer_and_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    fake = _FakeAnswerGenerator(
        "The lora adapter technique is explained here."
    )
    _patch_generator_load(monkeypatch, fake)

    RagCli().answer("lora adapter", k=3, processed_dir=str(processed))

    out = capsys.readouterr().out
    assert "The lora adapter technique is explained here." in out
    assert "Sources:" in out
    assert "b.md" in out


def test_answer_no_match_prints_fixed_answer_without_loading_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    _patch_generator_load_failing(monkeypatch)

    RagCli().answer("qwzxv plorkt", k=3, processed_dir=str(processed))

    assert NO_CONTEXT_ANSWER in capsys.readouterr().out


@pytest.mark.parametrize("bad_query", ["", "   "])
def test_answer_rejects_blank_query_without_loading_model(
    tmp_path: Path, bad_query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    _patch_generator_load_failing(monkeypatch)

    with pytest.raises(ValueError):
        RagCli().answer(bad_query, k=3, processed_dir=str(processed))


def test_answer_rejects_invalid_k_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    _patch_generator_load_failing(monkeypatch)

    with pytest.raises(ValueError):
        RagCli().answer("lora", k=0, processed_dir=str(processed))


def test_answer_dataset_writes_answers_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = _indexed_processed_dir(tmp_path)
    search_results_dir = tmp_path / "search_results"
    RagCli().search_dataset(
        dataset_path=str(_dataset_path(tmp_path, [
            {"question_id": "q1", "question": "lora adapter"},
            {"question_id": "q2", "question": "qwzxv plorkt"},
        ])),
        k=5,
        save_directory=str(search_results_dir),
        processed_dir=str(processed),
    )
    fake = _FakeAnswerGenerator(
        "The lora adapter technique is explained here."
    )
    _patch_generator_load(monkeypatch, fake)
    answers_dir = tmp_path / "answers"

    RagCli().answer_dataset(
        student_search_results_path=str(search_results_dir / "dataset.json"),
        save_directory=str(answers_dir),
    )

    saved_path = answers_dir / "dataset.json"
    assert saved_path.exists()
    results = StudentSearchResultsAndAnswer.model_validate_json(
        saved_path.read_text(encoding="utf-8")
    )
    assert results.k == 5
    by_id = {entry.question_id: entry for entry in results.search_results}
    expected_answer = "The lora adapter technique is explained here."
    assert by_id["q1"].answer == expected_answer
    assert by_id["q1"].question == "lora adapter"
    assert len(by_id["q1"].retrieved_sources) >= 1
    assert by_id["q2"].retrieved_sources == []
    assert by_id["q2"].answer == NO_CONTEXT_ANSWER


def test_answer_dataset_missing_results_file_raises_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeAnswerGenerator()
    _patch_generator_load(monkeypatch, fake)
    answers_dir = tmp_path / "answers"

    with pytest.raises(FileNotFoundError):
        RagCli().answer_dataset(
            student_search_results_path=str(tmp_path / "missing.json"),
            save_directory=str(answers_dir),
        )
    assert not answers_dir.exists()


def test_answer_dataset_malformed_results_file_raises_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeAnswerGenerator()
    _patch_generator_load(monkeypatch, fake)
    answers_dir = tmp_path / "answers"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError):
        RagCli().answer_dataset(
            student_search_results_path=str(bad_path),
            save_directory=str(answers_dir),
        )
    assert not answers_dir.exists()


def test_answer_dataset_unreadable_source_raises_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question": "lora adapter",
                        "retrieved_sources": [
                            {
                                "file_path": str(
                                    tmp_path / "does_not_exist.md"
                                ),
                                "first_character_index": 0,
                                "last_character_index": 5,
                            }
                        ],
                    }
                ],
                "k": 5,
            }
        ),
        encoding="utf-8",
    )
    fake = _FakeAnswerGenerator()
    _patch_generator_load(monkeypatch, fake)
    answers_dir = tmp_path / "answers"

    with pytest.raises(FileNotFoundError):
        RagCli().answer_dataset(
            student_search_results_path=str(results_path),
            save_directory=str(answers_dir),
        )
    assert not answers_dir.exists()


def _results_file(tmp_path: Path, name: str, entries: list, k: int) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps({"search_results": entries, "k": k}), encoding="utf-8"
    )
    return path


def _dataset_file(tmp_path: Path, name: str, questions: list) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"rag_questions": questions}), encoding="utf-8")
    return path


def test_evaluate_prints_recall_and_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    results_path = _results_file(
        tmp_path,
        "results.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "retrieved_sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
        k=5,
    )
    dataset_path = _dataset_file(
        tmp_path,
        "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Because.",
                "sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
    )

    RagCli().evaluate(
        student_search_results_path=str(results_path),
        dataset_path=str(dataset_path),
    )

    out = capsys.readouterr().out
    assert "Recall@1: 1.000" in out
    assert "Recall@5: 1.000" in out
    assert "Evaluated 1 questions (0 missing from the results)" in out


def test_evaluate_small_k_omits_higher_cutoffs(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    results_path = _results_file(
        tmp_path,
        "results.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "retrieved_sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
        k=3,
    )
    dataset_path = _dataset_file(
        tmp_path,
        "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Because.",
                "sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
    )

    RagCli().evaluate(
        student_search_results_path=str(results_path),
        dataset_path=str(dataset_path),
    )

    out = capsys.readouterr().out
    assert "Recall@1" in out
    assert "Recall@3" in out
    assert "Recall@5" not in out
    assert "Recall@10" not in out


def test_evaluate_warns_on_oversized_source(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    results_path = _results_file(
        tmp_path,
        "results.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "retrieved_sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 2500,
                    }
                ],
            }
        ],
        k=5,
    )
    dataset_path = _dataset_file(
        tmp_path,
        "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Because.",
                "sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
    )

    RagCli().evaluate(
        student_search_results_path=str(results_path),
        dataset_path=str(dataset_path),
    )

    assert "Warning: 1 retrieved source(s)" in capsys.readouterr().out


def test_evaluate_missing_dataset_raises(tmp_path: Path) -> None:
    results_path = _results_file(tmp_path, "results.json", [], k=5)
    with pytest.raises(FileNotFoundError):
        RagCli().evaluate(
            student_search_results_path=str(results_path),
            dataset_path=str(tmp_path / "missing.json"),
        )


def test_evaluate_malformed_results_raises(tmp_path: Path) -> None:
    results_path = tmp_path / "bad.json"
    results_path.write_text("not json", encoding="utf-8")
    dataset_path = _dataset_file(
        tmp_path,
        "dataset.json",
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Because.",
                "sources": [
                    {
                        "file_path": "a.md",
                        "first_character_index": 0,
                        "last_character_index": 100,
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError):
        RagCli().evaluate(
            student_search_results_path=str(results_path),
            dataset_path=str(dataset_path),
        )


def test_evaluate_unanswered_dataset_raises(tmp_path: Path) -> None:
    results_path = _results_file(tmp_path, "results.json", [], k=5)
    dataset_path = _dataset_file(
        tmp_path,
        "dataset.json",
        [{"question_id": "q1", "question": "What?"}],
    )
    with pytest.raises(ValueError):
        RagCli().evaluate(
            student_search_results_path=str(results_path),
            dataset_path=str(dataset_path),
        )
