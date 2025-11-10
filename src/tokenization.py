"""Tokenization utilities built around tiktoken."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import tiktoken


@dataclass
class TokenizationResult:
    """Bundle encoded ids with metadata."""

    tokens: List[int]
    encoding_name: str

    def count(self) -> int:
        return len(self.tokens)


class LLMTokenizer:
    """Thin wrapper around tiktoken encoder."""

    def __init__(self, model: Optional[str] = None, encoding_name: Optional[str] = None) -> None:
        if encoding_name:
            self._encoding = tiktoken.get_encoding(encoding_name)
        else:
            target_model = model or "gpt-4o-mini"
            try:
                self._encoding = tiktoken.encoding_for_model(target_model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        self.encoding_name = self._encoding.name

    def encode(self, text: str) -> TokenizationResult:
        tokens = self._encoding.encode(text, allowed_special="all")
        return TokenizationResult(tokens=tokens, encoding_name=self.encoding_name)

    def decode(self, tokens: Iterable[int]) -> str:
        return self._encoding.decode(list(tokens))

