"""Generates grounded answers from retrieved chunks using a local causal LM."""

import re
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import Chunk

NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the indexed sources to answer "
    "this question."
)
CONTEXT_TOKEN_BUDGET = 3000

_MAX_NEW_TOKENS = 256
_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a codebase. Use "
    "only the numbered context sources below to answer. Be concise and stay "
    "on point. If the sources do not contain the answer, say so explicitly."
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Answer:
    """A generated answer together with the chunks actually used for it."""

    text: str
    used_chunks: List[Chunk]


class AnswerGenerator:
    """Builds a grounded prompt from chunks and generates an answer."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """Store an already-built model and tokenizer pair."""
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, model_name: str) -> "AnswerGenerator":
        """Download/load model_name and its tokenizer, ready for inference."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")  # type: ignore[arg-type]
        return cls(model, tokenizer)

    def answer(self, question: str, chunks: List[Chunk]) -> Answer:
        """Generate a grounded answer to question from the ranked chunks."""
        if not chunks:
            return Answer(text=NO_CONTEXT_ANSWER, used_chunks=[])

        used_chunks, context = self._pack_context(chunks)
        prompt_ids = self._build_prompt_ids(question, context)
        with torch.inference_mode():
            output_ids = self.model.generate(
                prompt_ids, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False
            )
        generated_ids = output_ids[0][prompt_ids.shape[-1]:]
        raw_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return Answer(text=self._clean(raw_text), used_chunks=used_chunks)

    def _pack_context(self, chunks: List[Chunk]) -> Tuple[List[Chunk], str]:
        """Select the ranked chunks fitting the budget, first one truncated."""
        used_chunks: List[Chunk] = []
        blocks: List[str] = []
        remaining = CONTEXT_TOKEN_BUDGET
        for index, chunk in enumerate(chunks):
            block = self._format_block(index, chunk)
            token_count = len(
                self.tokenizer.encode(block, add_special_tokens=False)
            )
            if token_count > remaining:
                if index == 0:
                    blocks.append(self._truncate(block, remaining))
                    used_chunks.append(chunk)
                break
            blocks.append(block)
            used_chunks.append(chunk)
            remaining -= token_count
        return used_chunks, "\n\n".join(blocks)

    def _truncate(self, block: str, token_budget: int) -> str:
        """Cut block down to at most token_budget tokens."""
        token_ids = self.tokenizer.encode(block, add_special_tokens=False)
        return str(
            self.tokenizer.decode(
                token_ids[:token_budget], skip_special_tokens=True
            )
        )

    @staticmethod
    def _format_block(index: int, chunk: Chunk) -> str:
        """Render chunk as a numbered, path-labelled context block."""
        return f"[{index + 1}] {chunk.file_path}\n{chunk.text}"

    def _build_prompt_ids(self, question: str, context: str) -> Any:
        """Render the chat prompt as token ids, with thinking disabled."""
        user_message = (
            f"{context}\n\nQuestion: {question}" if context else question
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=False,
        )

    @staticmethod
    def _clean(text: str) -> str:
        """Strip any <think> block and surrounding whitespace from text."""
        return _THINK_BLOCK_RE.sub("", text).strip()
