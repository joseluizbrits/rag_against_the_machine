"""Tests for src.chunking.python_chunker.PythonChunker."""

import textwrap

import pytest

from src.chunking.python_chunker import PythonChunker


def test_decorated_function_kept_whole() -> None:
    source = textwrap.dedent(
        '''\
        import os


        @decorator
        def foo():
            """Docstring."""
            return os.getcwd()


        def bar():
            return 1
        '''
    )
    chunks = PythonChunker(max_chunk_size=2000).chunk("f.py", source)
    whole = [
        c
        for c in chunks
        if "@decorator" in c.text and "return os.getcwd()" in c.text
    ]
    assert len(whole) == 1
    assert whole[0].text.startswith("@decorator")


def test_oversized_class_split_without_cutting_a_method() -> None:
    method_body = "        return 1 + 1\n" * 5
    methods = "".join(
        f"    def method_{i}(self):\n{method_body}" for i in range(5)
    )
    source = f"class Big:\n{methods}"
    chunker = PythonChunker(max_chunk_size=300)
    chunks = chunker.chunk("f.py", source)

    assert len(chunks) > 1
    for i in range(5):
        marker = f"def method_{i}"
        method_start = source.index(marker)
        raw_end = source.index(f"def method_{i + 1}") if i < 4 else len(source)
        method_end = len(source[method_start:raw_end].rstrip()) + method_start
        containing = [
            c
            for c in chunks
            if c.first_character_index <= method_start
            and c.last_character_index >= method_end
        ]
        assert containing, f"method_{i} was split across chunks"


def test_syntax_error_falls_back_to_line_splitting() -> None:
    source = "def broken(:\n" + ("x = 1\n" * 200)
    chunks = PythonChunker(max_chunk_size=50).chunk("f.py", source)
    assert chunks
    for chunk in chunks:
        width = chunk.last_character_index - chunk.first_character_index
        assert width <= 50
        span = source[chunk.first_character_index:chunk.last_character_index]
        assert span == chunk.text


def test_offsets_are_exact_for_real_source() -> None:
    source = textwrap.dedent(
        """\
        class Foo:
            def bar(self):
                return 42

        def baz():
            return "hello"
        """
    )
    chunks = PythonChunker(max_chunk_size=2000).chunk("f.py", source)
    assert chunks
    for chunk in chunks:
        span = source[chunk.first_character_index:chunk.last_character_index]
        assert span == chunk.text


def test_invalid_max_chunk_size_rejected() -> None:
    with pytest.raises(ValueError):
        PythonChunker(max_chunk_size=0)
    with pytest.raises(ValueError):
        PythonChunker(max_chunk_size=-1)
