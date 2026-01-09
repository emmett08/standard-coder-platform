from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np

from standard_coder.forecasting.domain.models import (
    CapacityProfile,
    EffortEstimateProvider,
    ForecastResult,
    SprintConfig,
    ThroughputPrior,
    WorkItemState,
)
from standard_coder.forecasting.scenarios.scenarios import IdentityScenario, Scenario
from standard_coder.forecasting.simulator.monte_carlo import MonteCarloSprintForecaster
from standard_coder.integration.event_bus import EventBus
from standard_coder.integration.events import (
    AvailabilityChanged,
    ScopeChanged,
    SprintConfigured,
    SprintForecastComputed,
    WorkItemAddedToSprint,
    WorkItemEffortEstimated,
    WorkItemRemovedFromSprint,
    WorkItemStatusChanged,
)
from standard_coder.sch.domain.value_objects import EffortEstimate


logger = logging.getLogger(__name__)


@dataclass
class InMemoryEffortStore:
    """Stores latest effort estimates per work item."""

    _store: dict[str, EffortEstimate] = field(default_factory=dict)

    def upsert(self, work_item_id: str, estimate: EffortEstimate) -> None:
        self._store[work_item_id] = estimate

    def get(self, work_item_id: str) -> EffortEstimate | None:
        return self._store.get(work_item_id)


@dataclass
class StoreBackedEffortProvider(EffortEstimateProvider):
    store: InMemoryEffortStore

    def get_effort_samples_sch(
        self,
        work_item_id: str,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        est = self.store.get(work_item_id)
        if est is None:
            # Conservative fallback: assume large unknown task.
            return rng.lognormal(mean=0.0, sigma=1.0, size=n).astype(float)
        return est.distribution.sample(n=n, rng=rng).astype(float)


@dataclass
class SprintState:
    sprint: SprintConfig
    work_items: dict[str, WorkItemState] = field(default_factory=dict)
    capacities: dict[date, float] = field(default_factory=dict)


@dataclass
class ForecastingService:
    """Event-driven forecasting service with strict interfaces."""

    bus: EventBus
    throughput_prior: ThroughputPrior
    effort_store: InMemoryEffortStore = field(default_factory=InMemoryEffortStore)
    forecaster: MonteCarloSprintForecaster = field(default_factory=MonteCarloSprintForecaster)
    n_sims: int = 10_000
    rng_seed: int = 42
    auto_recompute: bool = False

    _sprints: dict[str, SprintState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bus.subscribe(SprintConfigured, self._on_sprint_configured)
        self.bus.subscribe(WorkItemAddedToSprint, self._on_item_added)
        self.bus.subscribe(WorkItemRemovedFromSprint, self._on_item_removed)
        self.bus.subscribe(WorkItemStatusChanged, self._on_status_changed)
        self.bus.subscribe(ScopeChanged, self._on_scope_changed)
        self.bus.subscribe(AvailabilityChanged, self._on_availability_changed)
        self.bus.subscribe(WorkItemEffortEstimated, self._on_effort_estimated)

    
    def _maybe_recompute(self, sprint_ids: Sequence[str]) -> None:
        if not self.auto_recompute:
            return
        for sid in set(sprint_ids):
            try:
                self.compute_forecast(sprint_id=sid)
            except Exception:
                logger.exception("Auto forecast recompute failed for sprint %s", sid)

# --- Event handlers -----------------------------------------------------

    def _on_sprint_configured(self, e: SprintConfigured) -> None:
        self._sprints[e.sprint_id] = SprintState(
            sprint=SprintConfig(
                sprint_id=e.sprint_id,
                start_date=e.start_date,
                length_days=e.length_days,
                working_days=frozenset(e.working_days),
            )
        )
        self._maybe_recompute([e.sprint_id])

    def _on_item_added(self, e: WorkItemAddedToSprint) -> None:
        st = self._sprints.get(e.sprint_id)
        if st is None:
            return
        st.work_items.setdefault(
            e.work_item_id,
            WorkItemState(work_item_id=e.work_item_id, status="todo"),
        )
        self._maybe_recompute([e.sprint_id])

    def _on_item_removed(self, e: WorkItemRemovedFromSprint) -> None:
        st = self._sprints.get(e.sprint_id)
        if st is None:
            return
        st.work_items.pop(e.work_item_id, None)
        self._maybe_recompute([e.sprint_id])

    def _on_status_changed(self, e: WorkItemStatusChanged) -> None:
        # Find which sprint contains this work item.
        for st in self._sprints.values():
            if e.work_item_id in st.work_items:
                w = st.work_items[e.work_item_id]
                st.work_items[e.work_item_id] = WorkItemState(
                    work_item_id=w.work_item_id,
                    status=e.new_status,
                    remaining_scope_factor=w.remaining_scope_factor,
                )
                self._maybe_recompute([st.sprint.sprint_id])

    def _on_scope_changed(self, e: ScopeChanged) -> None:
        for st in self._sprints.values():
            if e.work_item_id in st.work_items:
                w = st.work_items[e.work_item_id]
                st.work_items[e.work_item_id] = WorkItemState(
                    work_item_id=w.work_item_id,
                    status=w.status,
                    remaining_scope_factor=float(max(0.0, e.scope_factor)),
                )
                self._maybe_recompute([st.sprint.sprint_id])

    def _on_availability_changed(self, e: AvailabilityChanged) -> None:
        # In a real system you would map person -> team -> sprint, etc.
        for st in self._sprints.values():
            st.capacities[e.day] = float(max(0.0, st.capacities.get(e.day, 8.0) + e.delta_available_sch))
        self._maybe_recompute([s.sprint.sprint_id for s in self._sprints.values()])

    def _on_effort_estimated(self, e: WorkItemEffortEstimated) -> None:
        # Expect SCH effort estimate payload in a structured dict.
        payload = e.effort
        dist_type = str(payload.get("type", ""))

        if dist_type == "quantiles":
            from standard_coder.sch.domain.distributions import QuantileDistribution

            q = {float(k): float(v) for k, v in payload["quantiles"].items()}
            dist = QuantileDistribution(quantiles=q)
        elif dist_type == "gmm":
            from standard_coder.sch.domain.distributions import GaussianMixtureDistribution
            import numpy as np

            dist = GaussianMixtureDistribution(
                weights=np.array(payload["weights"], dtype=float),
                means=np.array(payload["means"], dtype=float),
                stds=np.array(payload["stds"], dtype=float),
                low=payload.get("low"),
                high=payload.get("high"),
            )
        else:
            logger.warning("Unknown effort distribution type: %s", dist_type)
            return

        est = EffortEstimate(
            distribution=dist,
            unit="sch",
            model_version=e.model_version,
        )
        self.effort_store.upsert(e.work_item_id, est)
        affected = [s.sprint.sprint_id for s in self._sprints.values() if e.work_item_id in s.work_items]
        self._maybe_recompute(affected)

    # --- Forecast API -------------------------------------------------------

    def compute_forecast(
        self,
        sprint_id: str,
        scenarios: Sequence[Scenario] | None = None,
    ) -> dict[str, ForecastResult]:
        st = self._sprints.get(sprint_id)
        if st is None:
            raise KeyError(f"Unknown sprint: {sprint_id}")

        effort_provider = StoreBackedEffortProvider(store=self.effort_store)

        base_caps = [
            CapacityProfile(day=d, available_sch=cap) for d, cap in sorted(st.capacities.items())
        ]
        base_items = list(st.work_items.values())

        scenarios = list(scenarios or [IdentityScenario()])

        results: dict[str, ForecastResult] = {}
        for sc in scenarios:
            items_sc = sc.apply_work_items(base_items)
            caps_sc = sc.apply_capacities(base_caps)
            res = self.forecaster.forecast(
                sprint=st.sprint,
                work_items=items_sc,
                capacities=caps_sc,
                effort_provider=effort_provider,
                throughput_prior=self.throughput_prior,
                n_sims=self.n_sims,
                rng_seed=self.rng_seed,
            )
            results[sc.scenario_id] = res

            self.bus.publish(
                SprintForecastComputed(
                    occurred_at=datetime.utcnow(),
                    sprint_id=sprint_id,
                    result=res.__dict__,
                    scenario_id=sc.scenario_id,
                )
            )

        return results
