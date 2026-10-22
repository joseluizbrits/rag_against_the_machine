"""Tests for src.evaluation."""

import pytest

from src.evaluation import evaluate, question_recall, span_iou
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def _source(file_path: str, start: int, end: int) -> MinimalSource:
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


def test_span_iou_chunk_covering_reference() -> None:
    reference = _source("a.md", 100, 300)
    retrieved = _source("a.md", 0, 1000)
    assert span_iou(reference, retrieved) == pytest.approx(0.2)


def test_span_iou_overlapping_span_below_threshold() -> None:
    reference = _source("a.md", 100, 300)
    retrieved = _source("a.md", 250, 2000)
    assert span_iou(reference, retrieved) == pytest.approx(50 / 1900)


def test_span_iou_same_span_different_file_does_not_match() -> None:
    reference = _source("a.md", 100, 300)
    retrieved = _source("b.md", 100, 300)
    assert span_iou(reference, retrieved) == 0.0


def test_question_recall_partial_match() -> None:
    references = [_source("a.md", 0, 100), _source("b.md", 0, 100)]
    retrieved = [_source("a.md", 0, 100)]
    assert question_recall(references, retrieved, k=5) == pytest.approx(0.5)


def test_question_recall_match_beyond_cutoff() -> None:
    references = [_source("a.md", 0, 100)]
    retrieved = [
        _source("x.md", 0, 100),
        _source("y.md", 0, 100),
        _source("z.md", 0, 100),
        _source("a.md", 0, 100),
    ]
    assert question_recall(references, retrieved, k=3) == 0.0
    assert question_recall(references, retrieved, k=5) == 1.0


def _answered_question(
    question_id: str, sources: list
) -> AnsweredQuestion:
    return AnsweredQuestion(
        question_id=question_id,
        question="What?",
        sources=sources,
        answer="Because.",
    )


def _results_entry(question_id: str, sources: list) -> MinimalSearchResults:
    return MinimalSearchResults(
        question_id=question_id, question="What?", retrieved_sources=sources
    )


def test_evaluate_missing_question_counts_zero_increments_missing() -> None:
    dataset = RagDataset(
        rag_questions=[
            _answered_question("q1", [_source("a.md", 0, 100)]),
            _answered_question("q2", [_source("b.md", 0, 100)]),
        ]
    )
    results = StudentSearchResults(
        search_results=[_results_entry("q1", [_source("a.md", 0, 100)])], k=5
    )

    report = evaluate(results, dataset)

    assert report.evaluated == 2
    assert report.missing == 1
    assert report.recall_at[5] == pytest.approx(0.5)


def test_evaluate_ignores_unknown_result_ids() -> None:
    dataset = RagDataset(
        rag_questions=[_answered_question("q1", [_source("a.md", 0, 100)])]
    )
    results = StudentSearchResults(
        search_results=[
            _results_entry("q1", [_source("a.md", 0, 100)]),
            _results_entry("unknown", [_source("c.md", 0, 100)]),
        ],
        k=5,
    )

    report = evaluate(results, dataset)

    assert report.evaluated == 1
    assert report.recall_at[5] == 1.0


def test_evaluate_omits_cutoffs_above_k() -> None:
    dataset = RagDataset(
        rag_questions=[_answered_question("q1", [_source("a.md", 0, 100)])]
    )
    results = StudentSearchResults(
        search_results=[_results_entry("q1", [_source("a.md", 0, 100)])], k=3
    )

    report = evaluate(results, dataset)

    assert set(report.recall_at.keys()) == {1, 3}


def test_evaluate_counts_oversized_source_and_overlong_entry() -> None:
    dataset = RagDataset(
        rag_questions=[_answered_question("q1", [_source("a.md", 0, 100)])]
    )
    results = StudentSearchResults(
        search_results=[
            _results_entry(
                "q1",
                [
                    _source("a.md", 0, 2500),
                    _source("b.md", 0, 50),
                    _source("c.md", 0, 50),
                ],
            )
        ],
        k=2,
    )

    report = evaluate(results, dataset)

    assert report.oversized_sources == 1
    assert report.overlong_entries == 1


def test_evaluate_no_reference_sources_raises() -> None:
    dataset = RagDataset(
        rag_questions=[_answered_question("q1", [])]
    )
    results = StudentSearchResults(search_results=[], k=5)

    with pytest.raises(ValueError):
        evaluate(results, dataset, dataset_path="dataset.json")
