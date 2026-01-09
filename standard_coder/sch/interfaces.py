from __future__ import annotations

from typing import Iterable, Protocol, Sequence

import numpy as np

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.domain.value_objects import EffortEstimate, MultiTaskEffortEstimate


class ChangeRepModel(Protocol):
    """Extract features from code change."""

    def fit(self, commits: Sequence[Commit]) -> None:
        ...

    def transform(self, commits: Sequence[Commit]) -> np.ndarray:
        ...


class WorkInferenceModel(Protocol):
    """Infer coding-time labels (probabilistic) from activity events."""

    def fit(self, commit_times_by_author: dict[str, Sequence[int]]) -> None:
        """Fit model per author; timestamps are integer minutes."""
        ...

    def infer_coding_minutes(
        self,
        author_id: str,
        parent_minute: int,
        child_minute: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return Monte Carlo samples of coding minutes in interval."""
        ...


class EffortPredictor(Protocol):
    """Predict SCH distributions given change features."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, x: np.ndarray) -> Sequence[EffortEstimate]:
        ...


class MultiTaskEffortPredictor(Protocol):
    def fit(self, x: np.ndarray, y_coding: np.ndarray, y_delivery: np.ndarray) -> None:
        ...

    def predict(self, x: np.ndarray) -> Sequence[MultiTaskEffortEstimate]:
        ...


class CommitSource(Protocol):
    """Load commits from a source (git, dataset, API)."""

    def iter_commits(self) -> Iterable[Commit]:
        ...
