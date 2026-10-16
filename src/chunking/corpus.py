"""Walks a corpus directory and routes each file to the right chunker."""

from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.chunking.markdown_chunker import MarkdownChunker
from src.chunking.python_chunker import PythonChunker
from src.chunking.spans import Chunker
from src.models import Chunk

_EXTENSION_CHUNKERS = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "markdown",
}


class CorpusChunker:
    """Discovers corpus files under a raw directory and chunks each one."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Raise ValueError if max_chunk_size is not positive."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self.max_chunk_size = max_chunk_size
        self._chunkers: Dict[str, Chunker] = {
            "python": PythonChunker(max_chunk_size),
            "markdown": MarkdownChunker(max_chunk_size),
        }

    def chunk_corpus(self, raw_dir: str) -> List[Chunk]:
        """Walk raw_dir and chunk every supported file, in a stable order."""
        candidates = [
            path
            for path in Path(raw_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in _EXTENSION_CHUNKERS
        ]
        files = sorted(candidates, key=self._relative_path)

        chunks: List[Chunk] = []
        for path in tqdm(files, desc="Chunking"):
            chunks.extend(self._chunk_file(path))
        return chunks

    def _chunk_file(self, path: Path) -> List[Chunk]:
        """Read and chunk one file, skipping it if it can't be decoded."""
        try:
            text = path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            return []
        kind = _EXTENSION_CHUNKERS[path.suffix.lower()]
        return self._chunkers[kind].chunk(self._relative_path(path), text)

    def _relative_path(self, path: Path) -> str:
        """The file's path relative to the cwd, with forward slashes."""
        if path.is_absolute():
            try:
                path = path.relative_to(Path.cwd())
            except ValueError:
                pass
        return path.as_posix()
