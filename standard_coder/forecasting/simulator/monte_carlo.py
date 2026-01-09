from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import numpy as np

from standard_coder.forecasting.domain.models import (
    CapacityProfile,
    EffortEstimateProvider,
    ForecastResult,
    SprintConfig,
    SprintForecaster,
    ThroughputPrior,
    WorkItemState,
)


@dataclass
class MonteCarloSprintForecaster(SprintForecaster):
    """Stochastic sprint completion forecaster using Monte Carlo simulation."""

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
        rng = np.random.default_rng(int(rng_seed))

        # Capacity map
        cap_map = {c.day: float(c.available_sch) for c in capacities}

        # Prepare per-day schedule
        days: list[tuple[int, bool, float]] = []
        for d in range(sprint.length_days):
            day_date = sprint.start_date + timedelta(days=d)
            is_working = day_date.weekday() in sprint.working_days
            cap = cap_map.get(day_date, 0.0 if not is_working else 8.0)  # default
            days.append((d + 1, is_working, float(max(0.0, cap))))

        remaining_items = [w for w in work_items if w.status != "done"]
        if not remaining_items:
            p_by_day = tuple(1.0 for _ in range(sprint.length_days))
            return ForecastResult(
                sprint_id=sprint.sprint_id,
                p_complete_by_end=1.0,
                p_complete_by_day=p_by_day,
                remaining_effort_p50_sch=0.0,
                remaining_effort_p90_sch=0.0,
                completion_day_p50=1,
                completion_day_p90=1,
            )

        # Sample effort for each item once: shape (n_items, n_sims)
        effort_samples = np.zeros((len(remaining_items), n_sims), dtype=float)
        scope = np.zeros(len(remaining_items), dtype=float)
        for i, item in enumerate(remaining_items):
            effort_samples[i] = effort_provider.get_effort_samples_sch(
                item.work_item_id,
                n=n_sims,
                rng=rng,
            )
            scope[i] = float(max(0.0, item.remaining_scope_factor))

        effort_samples *= scope[:, None]
        total_remaining = np.sum(effort_samples, axis=0)  # (n_sims,)

        completion_day = np.full(n_sims, fill_value=-1, dtype=int)

        for day_index, is_working, cap in days:
            if not is_working or cap <= 0.0:
                continue

            daily = np.array(
                [
                    throughput_prior.sample_daily_throughput_sch(rng, day_index)
                    for _ in range(n_sims)
                ],
                dtype=float,
            )
            daily = np.minimum(daily, cap)

            still_open = total_remaining > 0.0
            total_remaining[still_open] -= daily[still_open]

            newly_done = (completion_day < 0) & (total_remaining <= 0.0)
            completion_day[newly_done] = day_index

        completed = completion_day > 0
        p_complete_by_end = float(np.mean(completed))

        p_complete_by_day_list: list[float] = []
        for d in range(1, sprint.length_days + 1):
            p_complete_by_day_list.append(float(np.mean((completion_day > 0) & (completion_day <= d))))

        remaining_end = np.clip(total_remaining, 0.0, None)
        rem_p50 = float(np.quantile(remaining_end, 0.5))
        rem_p90 = float(np.quantile(remaining_end, 0.9))

        comp_days = completion_day[completed]
        if comp_days.size == 0:
            p50_day: int | None = None
            p90_day: int | None = None
        else:
            p50_day = int(np.quantile(comp_days, 0.5))
            p90_day = int(np.quantile(comp_days, 0.9))

        return ForecastResult(
            sprint_id=sprint.sprint_id,
            p_complete_by_end=p_complete_by_end,
            p_complete_by_day=tuple(p_complete_by_day_list),
            remaining_effort_p50_sch=rem_p50,
            remaining_effort_p90_sch=rem_p90,
            completion_day_p50=p50_day,
            completion_day_p90=p90_day,
        )
