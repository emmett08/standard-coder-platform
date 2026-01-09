from __future__ import annotations

import logging
from pathlib import Path
from typing import Final


DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, log_dir: str | None = None) -> None:
    """Configure standard library logging for CLI and scripts.

    If log_dir is provided, logs are written to '<log_dir>/run.log' as well
    as stderr. This is useful for long-running, resumable pipelines.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8"))

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT, handlers=handlers)
