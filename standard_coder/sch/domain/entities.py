from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Commit:
    """A commit with minimal information needed for effort modelling."""

    commit_id: str
    author_id: str
    authored_at: datetime
    language: str
    changed_files: tuple[str, ...]
    diff_text: str
    before_blobs: Mapping[str, str] | None = None
    after_blobs: Mapping[str, str] | None = None


@dataclass(frozen=True)
class PullRequest:
    pr_id: str
    repo: str
    author_id: str
    opened_at: datetime
    merged_at: datetime | None
    commits: tuple[str, ...]
    review_rounds: int = 0
    review_comments: int = 0
    ci_runs: int = 0
    ci_failures: int = 0
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True)
class WorkItem:
    """A unit of planning work (e.g., a ticket) linked to code changes."""

    work_item_id: str
    title: str
    language: str
    commit_ids: tuple[str, ...] = ()
    pr_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True)
class FeatureVector:
    """Numeric features for a single example."""

    values: Sequence[float]
