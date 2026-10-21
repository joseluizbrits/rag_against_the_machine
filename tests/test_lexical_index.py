"""Tests for src.indexing.lexical_index.LexicalIndex."""

from pathlib import Path

import numpy as np
import pytest

from src.indexing.lexical_index import LexicalIndex
from src.indexing.tokenizer import tokenize
from src.models import Chunk


def _chunk(file_path: str, text: str) -> Chunk:
    return Chunk(
        file_path=file_path,
        first_character_index=0,
        last_character_index=len(text),
        text=text,
    )


def test_build_save_load_round_trips_chunks_in_order(tmp_path: Path) -> None:
    chunks = [
        _chunk("data/raw/a.py", "def foo(): pass"),
        _chunk("data/raw/b.md", "# Title\nbody"),
    ]
    index = LexicalIndex.build(chunks)
    index.save(str(tmp_path))

    loaded = LexicalIndex.load(str(tmp_path))
    assert loaded.chunks == chunks


def test_path_term_contributes_to_the_match(tmp_path: Path) -> None:
    lora_chunk = _chunk(
        "data/raw/vllm-0.10.1/docs/features/lora.md",
        "This paragraph never mentions the adapter technique by name.",
    )
    other_chunks = [
        _chunk(
            "data/raw/vllm-0.10.1/docs/features/quantization.md",
            "This paragraph is about a completely different subject.",
        ),
        _chunk(
            "data/raw/vllm-0.10.1/docs/features/speculative_decoding.md",
            "This one covers yet another unrelated topic entirely.",
        ),
    ]
    index = LexicalIndex.build([lora_chunk, *other_chunks])

    scores = index.bm25.get_scores(tokenize("lora"))
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_load_from_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LexicalIndex.load(str(tmp_path))


def test_search_ranks_matching_identifier_first() -> None:
    lora_chunk = _chunk(
        "data/raw/vllm-0.10.1/docs/features/lora.md",
        "This paragraph explains the lora adapter technique in depth.",
    )
    other_chunks = [
        _chunk(
            "data/raw/vllm-0.10.1/docs/features/quantization.md",
            "This paragraph is about a completely different subject.",
        ),
        _chunk(
            "data/raw/vllm-0.10.1/docs/features/speculative_decoding.md",
            "This one covers yet another unrelated topic entirely.",
        ),
    ]
    index = LexicalIndex.build([lora_chunk, *other_chunks])

    results = index.search("lora", k=3)

    assert results[0].file_path == lora_chunk.file_path


def test_search_caps_results_at_k_in_descending_score_order() -> None:
    chunks = [
        _chunk("data/raw/doc_0.md", "lora lora lora adapter technique"),
        _chunk("data/raw/doc_1.md", "lora adapter mentioned briefly"),
        _chunk("data/raw/doc_2.md", "lora appears once here"),
        _chunk("data/raw/doc_3.md", "unrelated content about quantization"),
        _chunk("data/raw/doc_4.md", "another unrelated speculative decoding"),
    ]
    index = LexicalIndex.build(chunks)

    results = index.search("lora", k=2)

    assert len(results) == 2
    scores = index.bm25.get_scores(tokenize("lora"))
    by_path = {chunk.file_path: score for chunk, score in zip(chunks, scores)}
    result_scores = [by_path[chunk.file_path] for chunk in results]
    assert result_scores == sorted(result_scores, reverse=True)


def test_search_nonsensical_query_returns_empty() -> None:
    chunks = [
        _chunk("data/raw/a.py", "def foo(): return 1"),
        _chunk("data/raw/b.md", "# Title\nsome body text"),
    ]
    index = LexicalIndex.build(chunks)

    assert index.search("qwzxv plorkt", k=5) == []


def test_search_stopword_only_query_returns_empty() -> None:
    chunks = [
        _chunk("data/raw/a.py", "def foo(): return 1"),
        _chunk("data/raw/b.md", "# Title\nsome body text"),
    ]
    index = LexicalIndex.build(chunks)

    assert index.search("what is the", k=5) == []


def test_search_scores_match_bm25_get_scores() -> None:
    chunks = [
        _chunk(
            "data/raw/a.py",
            "def load_lora_adapter(path): return read_adapter(path)",
        ),
        _chunk(
            "data/raw/b.md",
            "# LoRA\nThe lora adapter technique fine-tunes efficiently.",
        ),
        _chunk(
            "data/raw/c.md",
            "# Unrelated\nThis talks about something else entirely.",
        ),
    ]
    index = LexicalIndex.build(chunks)
    query = "how does the lora adapter work"

    scores = np.zeros(len(chunks))
    for token in tokenize(query):
        postings = index._postings.get(token)
        idf = index.bm25.idf.get(token)
        if postings is None or not idf:
            continue
        doc_ids, term_freqs = postings
        scores[doc_ids] += (
            idf
            * term_freqs
            * (index.bm25.k1 + 1)
            / (term_freqs + index._norm[doc_ids])
        )

    expected_scores = index.bm25.get_scores(tokenize(query))
    np.testing.assert_allclose(scores, expected_scores, atol=1e-9)
