from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.interfaces import ChangeRepModel


@dataclass
class ConcatenatedChangeRep(ChangeRepModel):
    """Combine multiple ChangeRepModel feature spaces by concatenation."""

    parts: tuple[ChangeRepModel, ...]

    def fit(self, commits: Sequence[Commit]) -> None:
        for p in self.parts:
            p.fit(commits)

    def transform(self, commits: Sequence[Commit]) -> np.ndarray:
        mats = [p.transform(commits) for p in self.parts]
        return np.concatenate(mats, axis=1).astype(np.float32)
