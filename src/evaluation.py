"""Local recall@k evaluation of a search-results file against a dataset."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

MATCH_IOU = 0.05
MAX_SOURCE_LENGTH = 2000

_CUTOFFS: Tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class RecallReport:
    """Recall@k and grader-format counts for one evaluation run."""

    recall_at: Dict[int, float]
    evaluated: int
    missing: int
    oversized_sources: int
    overlong_entries: int


def span_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Return the character-range IoU of a and b, or 0.0 if files differ."""
    if a.file_path != b.file_path:
        return 0.0
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, end - start)
    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - intersection
    )
    if union <= 0:
        return 0.0
    return intersection / union


def question_recall(
    references: List[MinimalSource], retrieved: List[MinimalSource], k: int
) -> float:
    """Return the share of references matched by the top-k retrieved."""
    top_k = retrieved[:k]
    matched = sum(
        1
        for reference in references
        if any(span_iou(reference, source) >= MATCH_IOU for source in top_k)
    )
    return matched / len(references)


def evaluate(
    results: StudentSearchResults, dataset: RagDataset, dataset_path: str = ""
) -> RecallReport:
    """Score results against dataset, per the subject's recall@k metric."""
    scored_questions = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion) and question.sources
    ]
    if not scored_questions:
        name = dataset_path or "dataset"
        raise ValueError(f"{name} has no reference sources to evaluate")

    retrieved_by_id = {
        entry.question_id: entry.retrieved_sources
        for entry in results.search_results
    }
    cutoffs = [k for k in _CUTOFFS if k <= results.k]

    recall_at = {k: 0.0 for k in cutoffs}
    missing = 0
    for question in scored_questions:
        retrieved = retrieved_by_id.get(question.question_id)
        if retrieved is None:
            missing += 1
            retrieved = []
        for k in cutoffs:
            recall_at[k] += question_recall(question.sources, retrieved, k)

    evaluated = len(scored_questions)
    for k in cutoffs:
        recall_at[k] /= evaluated

    oversized_sources, overlong_entries = _grader_format_counts(
        results.search_results, results.k
    )
    return RecallReport(
        recall_at=recall_at,
        evaluated=evaluated,
        missing=missing,
        oversized_sources=oversized_sources,
        overlong_entries=overlong_entries,
    )


def _grader_format_counts(
    entries: List[MinimalSearchResults], k: int
) -> Tuple[int, int]:
    """Count sources over MAX_SOURCE_LENGTH and entries with over k sources."""
    oversized_sources = 0
    overlong_entries = 0
    for entry in entries:
        if len(entry.retrieved_sources) > k:
            overlong_entries += 1
        for source in entry.retrieved_sources:
            width = source.last_character_index - source.first_character_index
            if width > MAX_SOURCE_LENGTH:
                oversized_sources += 1
    return oversized_sources, overlong_entries
