from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from standard_coder.sch.domain.entities import PullRequest


def _parse_dt(value: str) -> datetime:
    # Accept 'Z' suffix
    v = value.replace("Z", "+00:00")
    return datetime.fromisoformat(v)


def load_pull_requests(path: Path) -> list[PullRequest]:
    """Load PR metadata from a JSON file.

    Expected format: a JSON array of objects, e.g.

    [
      {
        "pr_id": "...",
        "repo": "...",
        "author_id": "...",
        "opened_at": "2026-01-01T09:00:00+00:00",
        "merged_at": "2026-01-02T15:00:00+00:00",
        "commits": ["<sha1>", "<sha2>"],
        "review_rounds": 2,
        "review_comments": 10,
        "ci_runs": 3,
        "ci_failures": 1
      }
    ]
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("PR file must be a JSON list")

    prs: list[PullRequest] = []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        merged = obj.get("merged_at")
        prs.append(
            PullRequest(
                pr_id=str(obj["pr_id"]),
                repo=str(obj.get("repo", "")),
                author_id=str(obj.get("author_id", "")),
                opened_at=_parse_dt(str(obj["opened_at"])),
                merged_at=_parse_dt(str(merged)) if merged else None,
                commits=tuple(str(x) for x in obj.get("commits", [])),
                review_rounds=int(obj.get("review_rounds", 0)),
                review_comments=int(obj.get("review_comments", 0)),
                ci_runs=int(obj.get("ci_runs", 0)),
                ci_failures=int(obj.get("ci_failures", 0)),
                metadata={str(k): str(v) for k, v in obj.get("metadata", {}).items()}
                if isinstance(obj.get("metadata"), dict)
                else None,
            )
        )
    return prs
