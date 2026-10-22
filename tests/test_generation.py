"""Tests for src.generation.AnswerGenerator, using fake model and tokenizer."""

from typing import Any, Dict, List

import pytest
import torch

import src.generation as generation
from src.generation import NO_CONTEXT_ANSWER, AnswerGenerator
from src.models import Chunk


class FakeTokenizer:
    """A whitespace tokenizer that round-trips text through a growing vocab."""

    def __init__(self) -> None:
        """Start with an empty word <-> id vocabulary."""
        self._vocab: Dict[str, int] = {}
        self._reverse: Dict[int, str] = {}

    def _token_id(self, word: str) -> int:
        """Return word's id, assigning it a new one on first use."""
        if word not in self._vocab:
            new_id = len(self._vocab)
            self._vocab[word] = new_id
            self._reverse[new_id] = word
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Split text on whitespace and return each word's token id."""
        return [self._token_id(word) for word in text.split()]

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        """Join the words for token_ids back with single spaces."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return " ".join(self._reverse[int(i)] for i in token_ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_tensors: str,
        return_dict: bool,
    ) -> torch.Tensor:
        """Flatten the chat messages and encode them as a single token row."""
        text = " ".join(message["content"] for message in messages)
        return torch.tensor([self.encode(text)], dtype=torch.long)


class FakeModel:
    """A model whose generate() appends a fixed response to the prompt."""

    def __init__(self, tokenizer: FakeTokenizer, response_text: str) -> None:
        """Remember the tokenizer and the response text to append."""
        self.tokenizer = tokenizer
        self.response_text = response_text
        self.calls = 0
        self.last_input_ids: Any = None

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, do_sample: bool
    ) -> torch.Tensor:
        """Record the call and return input_ids with the response appended."""
        self.calls += 1
        self.last_input_ids = input_ids
        response_ids = torch.tensor(
            self.tokenizer.encode(self.response_text), dtype=torch.long
        )
        return torch.cat([input_ids[0], response_ids]).unsqueeze(0)


def _chunk(file_path: str, text: str) -> Chunk:
    return Chunk(
        file_path=file_path,
        first_character_index=0,
        last_character_index=len(text),
        text=text,
    )


def test_empty_chunks_returns_fixed_answer_without_calling_model() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, "unused")
    generator = AnswerGenerator(model, tokenizer)

    result = generator.answer("What is LoRA?", [])

    assert result.text == NO_CONTEXT_ANSWER
    assert result.used_chunks == []
    assert model.calls == 0


def test_blocks_appear_in_rank_order_with_file_paths() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, "Final answer.")
    generator = AnswerGenerator(model, tokenizer)
    chunk_a = _chunk("a.md", "Alpha content here")
    chunk_b = _chunk("b.md", "Beta content here")

    result = generator.answer("What is this?", [chunk_a, chunk_b])

    assert result.used_chunks == [chunk_a, chunk_b]
    assert model.calls == 1
    prompt_text = tokenizer.decode(model.last_input_ids[0])
    assert prompt_text.find("a.md") < prompt_text.find("b.md")


def test_chunks_past_budget_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "CONTEXT_TOKEN_BUDGET", 5)
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, "Final answer.")
    generator = AnswerGenerator(model, tokenizer)
    chunk_a = _chunk("a.md", "Hi")
    chunk_b = _chunk("b.md", "Hello world")
    chunk_c = _chunk("c.md", "More content")

    result = generator.answer("What is this?", [chunk_a, chunk_b, chunk_c])

    assert result.used_chunks == [chunk_a]
    prompt_text = tokenizer.decode(model.last_input_ids[0])
    assert "b.md" not in prompt_text
    assert "c.md" not in prompt_text


def test_oversized_first_chunk_is_truncated_to_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "CONTEXT_TOKEN_BUDGET", 3)
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, "Final answer.")
    generator = AnswerGenerator(model, tokenizer)
    big_chunk = _chunk("a.md", "Alpha content here is quite long indeed")

    result = generator.answer("What is this?", [big_chunk])

    assert result.used_chunks == [big_chunk]
    prompt_text = tokenizer.decode(model.last_input_ids[0])
    assert "quite" not in prompt_text
    assert "indeed" not in prompt_text


def test_think_block_and_whitespace_are_removed() -> None:
    tokenizer = FakeTokenizer()
    response = "<think> reasoning here </think> Actual answer."
    model = FakeModel(tokenizer, response)
    generator = AnswerGenerator(model, tokenizer)
    chunk = _chunk("a.md", "Some content")

    result = generator.answer("What is this?", [chunk])

    assert result.text == "Actual answer."
