"""Tests for src.models pydantic contracts."""

from typing import Any

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def _source(**overrides: Any) -> dict[str, Any]:
    data = {
        "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
        "first_character_index": 0,
        "last_character_index": 100,
    }
    data.update(overrides)
    return data


def test_minimal_source_validates() -> None:
    source = MinimalSource(**_source())
    assert source.file_path == "data/raw/vllm-0.10.1/docs/features/lora.md"
    assert source.first_character_index == 0
    assert source.last_character_index == 100


def test_chunk_is_a_minimal_source() -> None:
    chunk = Chunk(**_source(), text="LoRA adapter loading...")
    assert isinstance(chunk, MinimalSource)
    assert chunk.text == "LoRA adapter loading..."


def test_chunk_missing_text_rejected() -> None:
    with pytest.raises(ValidationError):
        Chunk(**_source())


def test_minimal_source_missing_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MinimalSource(  # type: ignore[call-arg]
            file_path="x", first_character_index=0
        )


def test_unanswered_question_generates_id() -> None:
    question = UnansweredQuestion(question="What is vLLM?")
    assert question.question_id


def test_answered_question_carries_sources_and_answer() -> None:
    answered = AnsweredQuestion(
        question="What is vLLM?",
        sources=[MinimalSource(**_source())],
        answer="vLLM is an inference engine.",
    )
    assert answered.question_id
    assert answered.question == "What is vLLM?"
    assert len(answered.sources) == 1
    assert answered.answer == "vLLM is an inference engine."


def test_rag_dataset_resolves_mixed_question_types() -> None:
    dataset = RagDataset.model_validate(
        {
            "rag_questions": [
                {"question": "unanswered?"},
                {
                    "question": "answered?",
                    "sources": [_source()],
                    "answer": "yes",
                },
            ]
        }
    )
    assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
    assert not isinstance(dataset.rag_questions[0], AnsweredQuestion)
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_minimal_search_results_without_answer() -> None:
    results = MinimalSearchResults(
        question_id="q1",
        question="How to configure OpenAI server?",
        retrieved_sources=[MinimalSource(**_source())],
    )
    assert results.retrieved_sources[0].file_path.endswith("lora.md")


def test_minimal_answer_extends_search_results() -> None:
    answer = MinimalAnswer(
        question_id="q1",
        question="How to configure OpenAI server?",
        retrieved_sources=[MinimalSource(**_source())],
        answer="Set --served-model-name.",
    )
    assert answer.retrieved_sources
    assert answer.answer == "Set --served-model-name."


def test_student_search_results_json_round_trip() -> None:
    original = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="q",
                retrieved_sources=[MinimalSource(**_source())],
            )
        ],
        k=5,
    )
    dumped = original.model_dump_json()
    restored = StudentSearchResults.model_validate_json(dumped)
    assert restored == original


def test_student_search_results_and_answer_json_round_trip() -> None:
    original = StudentSearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id="q1",
                question="q",
                retrieved_sources=[MinimalSource(**_source())],
                answer="a",
            )
        ],
        k=5,
    )
    restored = StudentSearchResultsAndAnswer.model_validate_json(
        original.model_dump_json()
    )
    assert restored == original
