from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from standard_coder.adapters.github.storage import init_schema
from standard_coder.common.time_utils import parse_iso8601
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


def load_pull_requests_from_github_db(path: Path) -> list[PullRequest]:
    """Load PR metadata from the GitHub ingestion SQLite DB."""
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    init_schema(conn)

    review_counts: dict[tuple[str, int], int] = {}
    for repo, pr_number, count in conn.execute(
        """
        SELECT repo, pr_number, COUNT(*)
        FROM pr_reviews
        GROUP BY repo, pr_number
        """
    ).fetchall():
        review_counts[(str(repo), int(pr_number))] = int(count)

    commits_by_pr: dict[tuple[str, int], list[str]] = {}
    for repo, pr_number, sha in conn.execute(
        "SELECT repo, pr_number, sha FROM pr_commits"
    ).fetchall():
        commits_by_pr.setdefault((str(repo), int(pr_number)), []).append(str(sha))

    prs: list[PullRequest] = []
    for row in conn.execute(
        """
        SELECT repo, number, pr_id, author_login, created_at, merged_at, review_comments
        FROM pull_requests
        """
    ).fetchall():
        repo, number, pr_id, author_login, created_at, merged_at, review_comments = row
        if not created_at:
            continue

        key = (str(repo), int(number))
        opened_at = parse_iso8601(str(created_at))
        merged_dt = parse_iso8601(str(merged_at)) if merged_at else None
        review_rounds = review_counts.get(key, 0)
        commits = tuple(commits_by_pr.get(key, []))

        prs.append(
            PullRequest(
                pr_id=str(pr_id) if pr_id else f"{repo}#{number}",
                repo=str(repo),
                author_id=str(author_login or ""),
                opened_at=opened_at,
                merged_at=merged_dt,
                commits=commits,
                review_rounds=review_rounds,
                review_comments=int(review_comments or 0),
                ci_runs=0,
                ci_failures=0,
                metadata={"pr_number": str(number)} if number is not None else None,
            )
        )

    conn.close()
    return prs
