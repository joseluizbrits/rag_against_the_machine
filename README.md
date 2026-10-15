_This project has been created as part of the 42 curriculum by jbrits-m._

## Description

RAG against the machine is a Retrieval-Augmented Generation system built to answer questions about the vLLM codebase. It ingests and indexes documentation and Python code, retrieves the most relevant snippets for a given query, and generates grounded answers using a local language model.

## Instructions

### Installation

Make sure you have `uv` installed, then run:

```bash
make install
```

### Execution

Run the pipeline or CLI commands using `uv run`:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure OpenAI server?" -k 5
uv run python -m src answer "How to configure OpenAI server?" -k 5
```

## Resources

- vLLM repository documentation
- Python, Pydantic, and Python Fire documentation
- AI was used as a collaborative pair programming assistant to design and implement parsing, indexing, and retrieval logic.

## System Architecture

The system consists of:

- **Indexer:** Chunks Python code and Markdown documents and builds a lexical index (BM25/TF-IDF) persisted under `data/processed/`.
- **Retriever:** Matches queries against the index to return top-k source snippets with file paths and character spans.
- **Generator:** Feeds retrieved context into the Qwen/Qwen3-0.6B language model to produce structured answers in JSON format.

## Chunking Strategy

- Python files are chunked respecting structural boundaries (functions, classes).
- Markdown files are chunked by headers or paragraph limits, ensuring chunks do not exceed `max_chunk_size` (default 2000 characters).

## Retrieval Method

Lexical retrieval (TF-IDF/BM25) scoring code and documentation tokens.

## Performance Analysis

Evaluated using Recall@k metrics on documentation and code datasets.

## Design Decisions

- Utilized Pydantic models for strict data validation between pipeline stages.
- Used Python Fire for clean CLI integration.

## Challenges Faced

Balancing chunk sizes and token budgets for the 0.6B local model while maintaining high recall@k.

## Example Usage

```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```
