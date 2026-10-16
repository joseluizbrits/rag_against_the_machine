"""Tests for src.indexing.tokenizer.tokenize."""

from src.indexing.tokenizer import tokenize


def test_snake_case_identifier() -> None:
    tokens = tokenize("load_lora_adapter")
    assert "load_lora_adapter" in tokens
    assert "load" in tokens
    assert "lora" in tokens
    assert "adapter" in tokens


def test_camel_case_identifier() -> None:
    tokens = tokenize("SamplingParams")
    assert "samplingparams" in tokens
    assert "sampling" in tokens
    assert "params" in tokens


def test_punctuation_separates_tokens() -> None:
    assert tokenize("--max-model-len=4096") == ["max", "model", "len", "4096"]


def test_question_words_removed() -> None:
    assert tokenize("What is the default port") == ["default", "port"]
