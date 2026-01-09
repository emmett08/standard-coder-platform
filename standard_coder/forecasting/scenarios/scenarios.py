from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence

from standard_coder.forecasting.domain.models import CapacityProfile, WorkItemState


class Scenario(Protocol):
    scenario_id: str

    def apply_work_items(self, items: Sequence[WorkItemState]) -> list[WorkItemState]:
        ...

    def apply_capacities(self, capacities: Sequence[CapacityProfile]) -> list[CapacityProfile]:
        ...


@dataclass(frozen=True)
class IdentityScenario:
    scenario_id: str = "baseline"

    def apply_work_items(self, items: Sequence[WorkItemState]) -> list[WorkItemState]:
        return list(items)

    def apply_capacities(self, capacities: Sequence[CapacityProfile]) -> list[CapacityProfile]:
        return list(capacities)


@dataclass(frozen=True)
class TimeOffScenario:
    """Reduce capacity on specific days (e.g., dev A off for 2 days)."""

    scenario_id: str
    days: tuple[date, ...]
    delta_available_sch: float

    def apply_work_items(self, items: Sequence[WorkItemState]) -> list[WorkItemState]:
        return list(items)

    def apply_capacities(self, capacities: Sequence[CapacityProfile]) -> list[CapacityProfile]:
        day_set = set(self.days)
        out: list[CapacityProfile] = []
        for c in capacities:
            if c.day in day_set:
                out.append(
                    CapacityProfile(
                        day=c.day,
                        available_sch=max(0.0, c.available_sch + self.delta_available_sch),
                    )
                )
            else:
                out.append(c)
        return out


@dataclass(frozen=True)
class ScopeGrowthScenario:
    """Multiply remaining work by a constant factor (e.g., +10% scope)."""

    scenario_id: str
    scope_factor: float

    def apply_work_items(self, items: Sequence[WorkItemState]) -> list[WorkItemState]:
        out: list[WorkItemState] = []
        for w in items:
            if w.status == "done":
                out.append(w)
                continue
            out.append(
                WorkItemState(
                    work_item_id=w.work_item_id,
                    status=w.status,
                    remaining_scope_factor=w.remaining_scope_factor * self.scope_factor,
                )
            )
        return out

    def apply_capacities(self, capacities: Sequence[CapacityProfile]) -> list[CapacityProfile]:
        return list(capacities)
