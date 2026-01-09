from __future__ import annotations

import logging
from typing import Final


DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure standard library logging for CLI and scripts."""
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
