from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from standard_coder.forecasting.domain.models import (
    CapacityProfile,
    SprintConfig,
    WorkItemState,
)
from standard_coder.forecasting.priors.throughput import EmpiricalThroughputPrior
from standard_coder.forecasting.simulator.monte_carlo import MonteCarloSprintForecaster


class _ConstEffortProvider:
    def __init__(self, value: float) -> None:
        self.value = value

    def get_effort_samples_sch(self, work_item_id: str, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value, dtype=float)


def test_monte_carlo_forecaster_shapes() -> None:
    sprint = SprintConfig(
        sprint_id="S1",
        start_date=date(2025, 1, 6),
        length_days=10,
        working_days=frozenset({0, 1, 2, 3, 4}),
    )
    items = [
        WorkItemState(work_item_id="A", status="todo", remaining_scope_factor=1.0),
        WorkItemState(work_item_id="B", status="todo", remaining_scope_factor=1.0),
    ]
    caps = [
        CapacityProfile(day=sprint.start_date + timedelta(days=i), available_sch=16.0)
        for i in range(sprint.length_days)
    ]
    provider = _ConstEffortProvider(value=10.0)
    prior = EmpiricalThroughputPrior(samples=(8.0, 10.0, 12.0))

    forecaster = MonteCarloSprintForecaster()
    res = forecaster.forecast(
        sprint=sprint,
        work_items=items,
        capacities=caps,
        effort_provider=provider,
        throughput_prior=prior,
        n_sims=2000,
        rng_seed=1,
    )

    assert 0.0 <= res.p_complete_by_end <= 1.0
    assert len(res.p_complete_by_day) == sprint.length_days
    assert res.remaining_effort_p50_sch >= 0.0
