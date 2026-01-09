from __future__ import annotations

from datetime import datetime


def parse_iso8601(dt_str: str) -> datetime:
    # GitHub/ZenHub use e.g. 2024-01-01T00:00:00Z
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)
