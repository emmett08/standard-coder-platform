from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


class Distribution(Protocol):
    """Minimal distribution contract used across SCH and forecasting."""

    def mean(self) -> float:
        ...

    def quantile(self, q: float) -> float:
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EffortEstimate:
    """Effort distribution returned by SCH models."""

    distribution: Distribution
    unit: str  # "sch" or "hours"
    model_version: str
    metadata: Mapping[str, Any] | None = None

    def mean(self) -> float:
        return float(self.distribution.mean())

    def p50(self) -> float:
        return float(self.distribution.quantile(0.5))

    def p90(self) -> float:
        return float(self.distribution.quantile(0.9))


@dataclass(frozen=True)
class MultiTaskEffortEstimate:
    coding: EffortEstimate
    delivery: EffortEstimate | None = None
