"""Fire CLI for the RAG pipeline."""

from pathlib import Path

from src.chunking.corpus import CorpusChunker
from src.indexing.lexical_index import LexicalIndex


def _require_positive_int(value: object, name: str) -> None:
    """Raise ValueError unless value is a real, positive int (no bools)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


class RagCli:
    """Command-line interface for the RAG pipeline."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Chunk raw_dir, build a BM25 index, persist it to processed_dir."""
        _require_positive_int(max_chunk_size, "max_chunk_size")
        if not Path(raw_dir).is_dir():
            raise ValueError(f"raw_dir does not exist: {raw_dir}")

        chunks = CorpusChunker(max_chunk_size).chunk_corpus(raw_dir)
        if not chunks:
            raise ValueError(f"No indexable files found under {raw_dir}")

        index = LexicalIndex.build(chunks)
        index.save(processed_dir)
        print(f"Ingestion complete! Indexed {len(chunks)} chunks under "
              f"{processed_dir}/")
