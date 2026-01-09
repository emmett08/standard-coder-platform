from __future__ import annotations

import logging
from typing import Any

from standard_coder.adapters.github.github_client import GitHubClient
from standard_coder.adapters.github.storage import connect, init_schema
from standard_coder.common.time_utils import parse_iso8601

logger = logging.getLogger(__name__)


def ingest_pull_requests(
    *,
    github: GitHubClient,
    repos: list[str],
    db_path,
    pulls_since_iso: str | None,
    ingest_reviews: bool,
    ingest_check_runs: bool,
) -> None:
    conn = connect(db_path)
    init_schema(conn)

    pulls_since = parse_iso8601(pulls_since_iso) if pulls_since_iso else None

    for repo_full in repos:
        owner, name = repo_full.split("/", 1)
        logger.info("Ingesting PRs for %s", repo_full)

        for pr in github.paginate(
            f"/repos/{owner}/{name}/pulls",
            params={"state": "all", "sort": "updated", "direction": "desc"},
            per_page=100,
            max_pages=50,
        ):
            updated_at = pr.get("updated_at")
            if pulls_since and updated_at:
                if parse_iso8601(updated_at) < pulls_since:
                    # List is sorted by updated desc; safe to stop.
                    break

            number = int(pr["number"])
            pr_data = github.get(f"/repos/{owner}/{name}/pulls/{number}")
            _upsert_pr(conn, repo_full, pr_data)

            _ingest_pr_commits(conn, github, owner, name, repo_full, number)

            if ingest_reviews:
                _ingest_pr_reviews(conn, github, owner, name, repo_full, number)

            # Optional (can be expensive / rate-limit heavy). Left as a hook.
            if ingest_check_runs:  # pragma: no cover
                pass

    conn.close()


def _upsert_pr(conn, repo_full: str, pr_data: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO pull_requests (
            repo, number, pr_id, state, title, author_login,
            created_at, updated_at, closed_at, merged_at,
            additions, deletions, changed_files, comments, review_comments, commits_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(repo, number) DO UPDATE SET
            pr_id=excluded.pr_id,
            state=excluded.state,
            title=excluded.title,
            author_login=excluded.author_login,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            closed_at=excluded.closed_at,
            merged_at=excluded.merged_at,
            additions=excluded.additions,
            deletions=excluded.deletions,
            changed_files=excluded.changed_files,
            comments=excluded.comments,
            review_comments=excluded.review_comments,
            commits_count=excluded.commits_count;
        """,
        (
            repo_full,
            int(pr_data["number"]),
            int(pr_data.get("id") or 0),
            pr_data.get("state"),
            pr_data.get("title"),
            (pr_data.get("user") or {}).get("login"),
            pr_data.get("created_at"),
            pr_data.get("updated_at"),
            pr_data.get("closed_at"),
            pr_data.get("merged_at"),
            int(pr_data.get("additions") or 0),
            int(pr_data.get("deletions") or 0),
            int(pr_data.get("changed_files") or 0),
            int(pr_data.get("comments") or 0),
            int(pr_data.get("review_comments") or 0),
            int(pr_data.get("commits") or 0),
        ),
    )
    conn.commit()


def _ingest_pr_commits(
    conn, github: GitHubClient, owner: str, name: str, repo_full: str, number: int
) -> None:
    for item in github.paginate(
        f"/repos/{owner}/{name}/pulls/{number}/commits", per_page=100, max_pages=50
    ):
        sha = (item.get("sha") or "").strip()
        if not sha:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO pr_commits (repo, pr_number, sha)
            VALUES (?, ?, ?)
            """,
            (repo_full, number, sha),
        )
    conn.commit()


def _ingest_pr_reviews(
    conn, github: GitHubClient, owner: str, name: str, repo_full: str, number: int
) -> None:
    for item in github.paginate(
        f"/repos/{owner}/{name}/pulls/{number}/reviews", per_page=100, max_pages=20
    ):
        review_id = int(item.get("id") or 0)
        if review_id == 0:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO pr_reviews (
                repo, pr_number, review_id, author_login, state, submitted_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                repo_full,
                number,
                review_id,
                (item.get("user") or {}).get("login"),
                item.get("state"),
                item.get("submitted_at"),
            ),
        )
    conn.commit()


def build_pr_metrics_map(db_path) -> dict[str, dict[str, Any]]:
    """Return PR metrics keyed by commit SHA for multi-task delivery labels."""
    conn = connect(db_path)
    init_schema(conn)

    rows = conn.execute(
        """
        SELECT
            pr.repo, pr.number, pr.created_at, pr.merged_at,
            pr.comments, pr.review_comments,
            (SELECT COUNT(*) FROM pr_reviews r WHERE r.repo=pr.repo AND r.pr_number=pr.number) AS review_count,
            pc.sha
        FROM pull_requests pr
        JOIN pr_commits pc
          ON pc.repo=pr.repo AND pc.pr_number=pr.number
        WHERE pr.merged_at IS NOT NULL
        """
    ).fetchall()

    metrics_by_sha: dict[str, dict[str, Any]] = {}
    for repo, number, created_at, merged_at, comments, review_comments, review_count, sha in rows:
        key = str(sha)
        entry = {
            "repo": repo,
            "pr_number": int(number),
            "created_at": created_at,
            "merged_at": merged_at,
            "comments": int(comments or 0),
            "review_comments": int(review_comments or 0),
            "review_count": int(review_count or 0),
        }
        prev = metrics_by_sha.get(key)
        if not prev:
            metrics_by_sha[key] = entry
            continue

        # Pick the PR with later merged_at.
        prev_m = parse_iso8601(prev["merged_at"])
        new_m = parse_iso8601(entry["merged_at"])
        if new_m > prev_m:
            metrics_by_sha[key] = entry

    conn.close()
    return metrics_by_sha
