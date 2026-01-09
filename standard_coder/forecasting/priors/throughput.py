from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.stats as st

from standard_coder.forecasting.domain.models import ThroughputPrior

logger = logging.getLogger(__name__)


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


def default_throughput_prior() -> ThroughputPrior:
    # Conservative fallback: 1..10 SCH/day bootstrap.
    samples = tuple(float(i) for i in range(1, 11))
    return EmpiricalThroughputPrior(samples=samples)


def throughput_prior_from_history(path: Path) -> ThroughputPrior:
    # Expected JSON format (example):
    # [
    #   {"sprint_id": "...", "done_effort_hours": 60.0, "workdays": 10},
    #   ...
    # ]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read historical sprint outcomes: %s", exc)
        return default_throughput_prior()

    per_day: list[float] = []
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            done = float(row.get("done_effort_hours") or 0.0)
            days = float(row.get("workdays") or 0.0)
            if done > 0 and days > 0:
                per_day.append(done / days)

    if len(per_day) < 3:
        return default_throughput_prior()

    return EmpiricalThroughputPrior(samples=tuple(per_day))
