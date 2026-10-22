"""Fire CLI for the RAG pipeline."""

from pathlib import Path
from typing import List

from tqdm import tqdm

from src.chunking.corpus import CorpusChunker
from src.datasets import load_dataset, load_search_results, save_model
from src.generation import NO_CONTEXT_ANSWER, AnswerGenerator
from src.indexing.lexical_index import LexicalIndex
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.source_reader import SourceReader

_DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"

MAX_SOURCE_LENGTH = 2000


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
        if max_chunk_size > MAX_SOURCE_LENGTH:
            raise ValueError(
                f"max_chunk_size must be at most {MAX_SOURCE_LENGTH} "
                f"(the grader rejects longer sources), got {max_chunk_size}"
            )
        if not Path(raw_dir).is_dir():
            raise ValueError(f"raw_dir does not exist: {raw_dir}")

        chunks = CorpusChunker(max_chunk_size).chunk_corpus(raw_dir)
        if not chunks:
            raise ValueError(f"No indexable files found under {raw_dir}")

        index = LexicalIndex.build(chunks)
        index.save(processed_dir)
        print(f"Ingestion complete! Indexed {len(chunks)} chunks under "
              f"{processed_dir}/")

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> None:
        """Print the k best-matching source locations for query, best first."""
        if not query.strip():
            raise ValueError("query must not be empty")
        _require_positive_int(k, "k")

        index = LexicalIndex.load(processed_dir)
        results = index.search(query, k)
        if not results:
            print("No results found")
            return
        for chunk in results:
            print(
                f"{chunk.file_path} "
                f"[{chunk.first_character_index}:{chunk.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
    ) -> None:
        """Search every dataset question, saving a StudentSearchResults."""
        _require_positive_int(k, "k")
        dataset = load_dataset(dataset_path)
        index = LexicalIndex.load(processed_dir)

        results: List[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            chunks = index.search(question.question, k)
            sources = [
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ]
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=results, k=k)
        file_name = Path(dataset_path).name
        saved_path = save_model(output, save_directory, file_name)
        print(f"Search results saved to {saved_path}")

    def answer(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = "data/processed",
        model_name: str = _DEFAULT_MODEL_NAME,
    ) -> None:
        """Print a grounded answer to query and the sources it used."""
        if not query.strip():
            raise ValueError("query must not be empty")
        _require_positive_int(k, "k")

        index = LexicalIndex.load(processed_dir)
        chunks = index.search(query, k)
        if not chunks:
            print(NO_CONTEXT_ANSWER)
            return

        generator = AnswerGenerator.load(model_name)
        result = generator.answer(query, chunks)
        print(result.text)
        print()
        print("Sources:")
        for chunk in result.used_chunks:
            print(
                f"{chunk.file_path} "
                f"[{chunk.first_character_index}:{chunk.last_character_index}]"
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = _DEFAULT_MODEL_NAME,
    ) -> None:
        """Answer every entry of a search-results file, saving the answers."""
        results = load_search_results(student_search_results_path)
        reader = SourceReader()
        chunks_by_entry = [
            [reader.read(source) for source in entry.retrieved_sources]
            for entry in results.search_results
        ]

        generator = AnswerGenerator.load(model_name)
        answers: List[MinimalAnswer] = []
        for entry, chunks in tqdm(
            list(zip(results.search_results, chunks_by_entry)),
            desc="Answering",
        ):
            generated = generator.answer(entry.question, chunks)
            answers.append(
                MinimalAnswer(
                    question_id=entry.question_id,
                    question=entry.question,
                    retrieved_sources=entry.retrieved_sources,
                    answer=generated.text,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k
        )
        file_name = Path(student_search_results_path).name
        saved_path = save_model(output, save_directory, file_name)
        print(f"Answers saved to {saved_path}")
