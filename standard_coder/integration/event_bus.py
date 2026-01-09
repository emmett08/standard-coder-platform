from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Protocol, Type, TypeVar

from standard_coder.integration.events import DomainEvent


logger = logging.getLogger(__name__)

E = TypeVar("E", bound=DomainEvent)
Handler = Callable[[DomainEvent], None]


class EventBus(Protocol):
    """Publish/subscribe interface for domain events."""

    def publish(self, event: DomainEvent) -> None:
        ...

    def subscribe(self, event_type: Type[E], handler: Callable[[E], None]) -> None:
        ...


@dataclass
class InMemoryEventBus(EventBus):
    """Simple synchronous in-process pub/sub bus."""

    _handlers: DefaultDict[Type[DomainEvent], list[Handler]]

    def __init__(self) -> None:
        self._handlers = defaultdict(list)

    def publish(self, event: DomainEvent) -> None:
        for event_type, handlers in self._handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception:
                        logger.exception(
                            "Event handler failed: %s for %s",
                            handler,
                            event,
                        )

    def subscribe(self, event_type: Type[E], handler: Callable[[E], None]) -> None:
        self._handlers[event_type].append(handler)  # type: ignore[arg-type]
