"""Tests for src.indexing.lexical_index.LexicalIndex."""

from pathlib import Path

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
