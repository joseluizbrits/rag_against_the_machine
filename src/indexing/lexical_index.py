"""Persisted BM25 index over corpus chunks."""

import os
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.indexing.tokenizer import tokenize
from src.models import Chunk

_INDEX_FILENAME = "bm25_index.pkl"

_Postings = Dict[str, Tuple[NDArray[np.int64], NDArray[np.float64]]]


class LexicalIndex:
    """A BM25 index over corpus chunks, aligned by position with the chunks."""

    def __init__(self, chunks: List[Chunk], bm25: BM25Okapi) -> None:
        """Store chunks and bm25, deriving an inverted index from bm25."""
        self.chunks = chunks
        self.bm25 = bm25
        self._postings = self._build_postings(bm25)
        doc_len = np.array(bm25.doc_len, dtype=np.float64)
        self._norm = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)

    @staticmethod
    def _build_postings(bm25: BM25Okapi) -> _Postings:
        """Invert bm25.doc_freqs into token -> (doc_ids, term_freqs) arrays."""
        raw_postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for doc_id, freqs in enumerate(bm25.doc_freqs):
            for token, term_freq in freqs.items():
                raw_postings[token].append((doc_id, term_freq))
        return {
            token: (
                np.array([doc_id for doc_id, _ in postings], dtype=np.int64),
                np.array([tf for _, tf in postings], dtype=np.float64),
            )
            for token, postings in raw_postings.items()
        }

    def search(self, query: str, k: int) -> List[Chunk]:
        """Return the k highest BM25-scoring chunks for query, best first."""
        scores = np.zeros(len(self.chunks))
        for token in tokenize(query):
            postings = self._postings.get(token)
            idf = self.bm25.idf.get(token)
            if postings is None or not idf:
                continue
            doc_ids, term_freqs = postings
            scores[doc_ids] += (
                idf
                * term_freqs
                * (self.bm25.k1 + 1)
                / (term_freqs + self._norm[doc_ids])
            )
        return self._top_k(scores, k)

    def _top_k(self, scores: NDArray[np.float64], k: int) -> List[Chunk]:
        """Pick the k highest positive scores, sorted descending, as chunks."""
        matched = np.nonzero(scores > 0)[0]
        if matched.size == 0 or k <= 0:
            return []
        top_count = min(k, matched.size)
        partitioned = np.argpartition(scores[matched], -top_count)
        top = matched[partitioned[-top_count:]]
        top = top[np.argsort(scores[top])[::-1]]
        return [self.chunks[i] for i in top]

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
