from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from standard_coder.integration.event_bus import InMemoryEventBus
from standard_coder.integration.events import DomainEvent


@dataclass(frozen=True)
class _Evt(DomainEvent):
    value: int


def test_event_bus_publish_subscribe() -> None:
    bus = InMemoryEventBus()
    seen: list[int] = []

    def handler(e: _Evt) -> None:
        seen.append(e.value)

    bus.subscribe(_Evt, handler)
    bus.publish(_Evt(occurred_at=datetime.now(timezone.utc), value=1))
    bus.publish(_Evt(occurred_at=datetime.now(timezone.utc), value=2))

    assert seen == [1, 2]
