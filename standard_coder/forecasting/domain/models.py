from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class SprintConfig:
    sprint_id: str
    start_date: date
    length_days: int
    working_days: frozenset[int]  # 0=Mon ... 6=Sun


@dataclass(frozen=True)
class CapacityProfile:
    day: date
    available_sch: float


@dataclass(frozen=True)
class WorkItemState:
    work_item_id: str
    status: str  # "todo", "in_progress", "done"
    remaining_scope_factor: float = 1.0


@dataclass(frozen=True)
class ForecastResult:
    sprint_id: str
    p_complete_by_end: float
    p_complete_by_day: tuple[float, ...]
    remaining_effort_p50_sch: float
    remaining_effort_p90_sch: float
    completion_day_p50: int | None
    completion_day_p90: int | None
    metadata: Mapping[str, float] | None = None


class EffortEstimateProvider(Protocol):
    def get_effort_samples_sch(
        self,
        work_item_id: str,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return Monte Carlo samples of remaining effort (SCH)."""
        ...


class ThroughputPrior(Protocol):
    def sample_daily_throughput_sch(
        self,
        rng: np.random.Generator,
        day_index: int,
    ) -> float:
        """Sample stochastic throughput per day (SCH/day)."""
        ...


class SprintForecaster(Protocol):
    def forecast(
        self,
        sprint: SprintConfig,
        work_items: Sequence[WorkItemState],
        capacities: Sequence[CapacityProfile],
        effort_provider: EffortEstimateProvider,
        throughput_prior: ThroughputPrior,
        n_sims: int,
        rng_seed: int,
    ) -> ForecastResult:
        ...
