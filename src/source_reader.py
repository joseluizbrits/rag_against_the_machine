"""Recovers chunk text from a source location by rereading the corpus."""

from pathlib import Path
from typing import Dict

from src.models import Chunk, MinimalSource


class SourceReader:
    """Reads corpus files once per run and slices chunk text from them."""

    def __init__(self) -> None:
        """Start with an empty per-file text cache."""
        self._cache: Dict[str, str] = {}

    def read(self, source: MinimalSource) -> Chunk:
        """Return source as a Chunk, rereading its file's text if needed."""
        text = self._read_file(source.file_path)
        start, end = source.first_character_index, source.last_character_index
        chunk_text = text[start:end]
        return Chunk(
            file_path=source.file_path,
            first_character_index=source.first_character_index,
            last_character_index=source.last_character_index,
            text=chunk_text,
        )

    def _read_file(self, file_path: str) -> str:
        """Return file_path's decoded text, reading it only on first use."""
        if file_path not in self._cache:
            path = Path(file_path)
            if not path.is_file():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            self._cache[file_path] = path.read_bytes().decode("utf-8")
        return self._cache[file_path]
