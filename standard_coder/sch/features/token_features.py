from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.interfaces import ChangeRepModel


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|==|!=|<=|>=|\S")


def simple_tokenise(text: str) -> list[str]:
    """Tokenise source diff text.

    This is intentionally simple and language-agnostic. Upgrade A provides
    structure-aware tokenisation for supported languages.
    """
    return _TOKEN_RE.findall(text)


@dataclass
class BagOfTokensChangeRep(ChangeRepModel):
    """Baseline change representation: bag-of-tokens."""

    max_features: int = 256
    min_df: int = 2
    vectoriser: CountVectorizer | None = None

    def fit(self, commits: Sequence[Commit]) -> None:
        texts = [c.diff_text for c in commits]
        self.vectoriser = CountVectorizer(
            tokenizer=simple_tokenise,
            lowercase=False,
            max_features=self.max_features,
            min_df=self.min_df,
            token_pattern=None,
        )
        self.vectoriser.fit(texts)

    def transform(self, commits: Sequence[Commit]) -> np.ndarray:
        if self.vectoriser is None:
            raise RuntimeError("Call fit() before transform().")
        texts = [c.diff_text for c in commits]
        x = self.vectoriser.transform(texts)
        return x.toarray().astype(np.float32)
