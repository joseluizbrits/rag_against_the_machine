"""Reading RagDataset files and writing pipeline result models."""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.models import RagDataset


def load_dataset(path: str) -> RagDataset:
    """Load and validate a RagDataset from a UTF-8 JSON file at path."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        return RagDataset.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as error:
        message = str(error).replace("\n", " ")
        raise ValueError(
            f"{path} is not a valid dataset: {message}"
        ) from error


def save_model(model: BaseModel, save_directory: str, file_name: str) -> Path:
    """Write model as indented UTF-8 JSON under save_directory, return path."""
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / file_name
    target.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return target
