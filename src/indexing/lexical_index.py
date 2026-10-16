"""Persisted BM25 index over corpus chunks."""

import os
import pickle
import tempfile
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.indexing.tokenizer import tokenize
from src.models import Chunk

_INDEX_FILENAME = "bm25_index.pkl"


class LexicalIndex:
    """A BM25 index over corpus chunks, aligned by position with the chunks."""

    def __init__(self, chunks: List[Chunk], bm25: BM25Okapi) -> None:
        """Store the chunks and their fitted BM25 model, in matching order."""
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "LexicalIndex":
        """Tokenize each chunk's path + text and fit a BM25 model over them."""
        corpus = [
            tokenize(f"{chunk.file_path}\n{chunk.text}")
            for chunk in tqdm(chunks, desc="Tokenizing")
        ]
        return cls(chunks, BM25Okapi(corpus))

    def save(self, processed_dir: str) -> None:
        """Persist this index to processed_dir, replacing any index there."""
        directory = Path(processed_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / _INDEX_FILENAME
        descriptor, tmp_name = tempfile.mkstemp(dir=directory)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, handle)
            tmp_path.replace(target)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, processed_dir: str) -> "LexicalIndex":
        """Load a previously saved index, raising if none exists there."""
        path = Path(processed_dir) / _INDEX_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"No index found under {processed_dir}")
        with path.open("rb") as handle:
            data = pickle.load(handle)
        return cls(data["chunks"], data["bm25"])
