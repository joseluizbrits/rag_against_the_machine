"""Tests for src.datasets."""

import json
from pathlib import Path

import pytest

from src.datasets import load_dataset, save_model
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def test_load_dataset_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {"question_id": "q1", "question": "What is LoRA?"}
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(str(path))

    assert len(dataset.rag_questions) == 1
    assert dataset.rag_questions[0].question == "What is LoRA?"


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / "missing.json"))


def test_load_dataset_invalid_json_raises_single_line_value_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_dataset(str(path))
    assert "\n" not in str(excinfo.value)


def test_load_dataset_without_rag_questions_raises_single_line_value_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong_schema.json"
    path.write_text(json.dumps({"something_else": []}), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_dataset(str(path))
    assert "\n" not in str(excinfo.value)


def test_save_model_creates_nested_directories_and_round_trips(
    tmp_path: Path,
) -> None:
    model = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="What is LoRA?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/lora.md",
                        first_character_index=0,
                        last_character_index=10,
                    )
                ],
            )
        ],
        k=5,
    )
    save_directory = tmp_path / "nested" / "output"

    saved_path = save_model(model, str(save_directory), "results.json")

    assert saved_path == save_directory / "results.json"
    loaded = StudentSearchResults.model_validate_json(
        saved_path.read_text(encoding="utf-8")
    )
    assert loaded == model
