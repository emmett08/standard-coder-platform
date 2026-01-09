from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainEvent:
    """Base type for all domain events."""

    occurred_at: datetime


# --- SCH domain events -------------------------------------------------------


@dataclass(frozen=True)
class WorkItemEffortEstimated(DomainEvent):
    work_item_id: str
    effort: Mapping[str, Any]
    model_version: str


@dataclass(frozen=True)
class WorkItemEffortInvalidated(DomainEvent):
    work_item_id: str
    reason: str


# --- Forecasting / delivery domain events -----------------------------------


@dataclass(frozen=True)
class SprintConfigured(DomainEvent):
    sprint_id: str
    start_date: date
    length_days: int
    working_days: tuple[int, ...]  # 0=Mon ... 6=Sun


@dataclass(frozen=True)
class WorkItemAddedToSprint(DomainEvent):
    sprint_id: str
    work_item_id: str


@dataclass(frozen=True)
class WorkItemRemovedFromSprint(DomainEvent):
    sprint_id: str
    work_item_id: str


@dataclass(frozen=True)
class WorkItemStatusChanged(DomainEvent):
    work_item_id: str
    old_status: str
    new_status: str


@dataclass(frozen=True)
class ScopeChanged(DomainEvent):
    work_item_id: str
    scope_factor: float


@dataclass(frozen=True)
class AvailabilityChanged(DomainEvent):
    person_id: str
    day: date
    delta_available_sch: float


@dataclass(frozen=True)
class SprintForecastComputed(DomainEvent):
    sprint_id: str
    result: Mapping[str, Any]
    scenario_id: str | None = None
