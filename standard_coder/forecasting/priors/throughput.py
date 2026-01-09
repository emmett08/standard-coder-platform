from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.stats as st

from standard_coder.forecasting.domain.models import ThroughputPrior


@dataclass(frozen=True)
class LogNormalThroughputPrior(ThroughputPrior):
    """A simple prior over daily throughput (SCH/day) using a log-normal."""

    sigma: float
    scale: float

    @classmethod
    def fit_from_samples(cls, samples: Sequence[float]) -> "LogNormalThroughputPrior":
        x = np.array([max(1e-9, float(s)) for s in samples], dtype=float)
        shape, loc, scale = st.lognorm.fit(x, floc=0.0)
        return cls(sigma=float(shape), scale=float(scale))

    def sample_daily_throughput_sch(self, rng: np.random.Generator, day_index: int) -> float:
        return float(st.lognorm(s=self.sigma, scale=self.scale).rvs(random_state=rng))


@dataclass(frozen=True)
class EmpiricalThroughputPrior(ThroughputPrior):
    """Empirical bootstrap prior over daily throughput."""

    samples: tuple[float, ...]

    def sample_daily_throughput_sch(self, rng: np.random.Generator, day_index: int) -> float:
        idx = int(rng.integers(0, len(self.samples)))
        return float(self.samples[idx])
