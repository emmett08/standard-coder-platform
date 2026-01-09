from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

from standard_coder.adapters.github.github_client import GitHubClient
from standard_coder.adapters.zenhub.storage import connect, init_schema
from standard_coder.adapters.zenhub.zenhub_client import ZenHubClient
from standard_coder.common.time_utils import parse_iso8601

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SprintBacklog:
    sprint_id: str
    repo: str
    start_date: date
    end_date: date
    issue_numbers: list[int]


def ingest_current_sprints_from_milestones(
    *,
    github: GitHubClient,
    zenhub: ZenHubClient,
    repos: list[str],
    github_repo_ids: dict[str, int],
    db_path,
    milestone_title_prefix: str,
    default_sprint_length_days: int,
) -> list[SprintBacklog]:
    conn = connect(db_path)
    init_schema(conn)

    backlogs: list[SprintBacklog] = []

    for repo_full in repos:
        owner, name = repo_full.split("/", 1)
        repo_id = github_repo_ids[repo_full]

        # Pick the "current" milestone:
        # - open milestones only
        # - title prefix match (optional)
        milestones = list(
            github.paginate(
                f"/repos/{owner}/{name}/milestones",
                params={"state": "open", "sort": "due_on", "direction": "asc"},
                per_page=100,
                max_pages=5,
            )
        )
        if milestone_title_prefix:
            milestones = [
                m for m in milestones if str(m.get("title", "")).startswith(milestone_title_prefix)
            ]

        if not milestones:
            logger.warning("No open sprint milestones found for %s", repo_full)
            continue

        m = milestones[0]
        milestone_number = int(m["number"])
        title = str(m.get("title", f"Milestone {milestone_number}"))
        due_on = m.get("due_on")
        if due_on:
            end_dt = parse_iso8601(due_on).date()
            start_dt = end_dt.fromordinal(end_dt.toordinal() - default_sprint_length_days)
        else:
            # If no due date, use today as end and back-calc start.
            end_dt = date.today()
            start_dt = end_dt.fromordinal(end_dt.toordinal() - default_sprint_length_days)

        sprint_id = f"{repo_full}#milestone-{milestone_number}"

        conn.execute(
            """
            INSERT OR REPLACE INTO zenhub_milestones (
                repo, milestone_id, title, start_date, end_date, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                repo_full,
                milestone_number,
                title,
                start_dt.isoformat(),
                end_dt.isoformat(),
                json.dumps(m),
            ),
        )
        conn.commit()

        # Fetch issues attached to that milestone from GitHub.
        issues = list(
            github.paginate(
                f"/repos/{owner}/{name}/issues",
                params={"milestone": milestone_number, "state": "all"},
                per_page=100,
                max_pages=20,
            )
        )
        issue_numbers: list[int] = []
        for issue in issues:
            if "pull_request" in issue:
                continue
            issue_number = int(issue["number"])
            issue_numbers.append(issue_number)

            # Pull ZenHub metadata (estimate/pipeline).
            try:
                zh_issue = zenhub.get_issue_data(repo_id, issue_number)
            except Exception as exc:  # pragma: no cover
                logger.warning("ZenHub issue fetch failed for %s#%s: %s", repo_full, issue_number, exc)
                zh_issue = {}

            estimate_value = None
            if isinstance(zh_issue, dict):
                estimate = zh_issue.get("estimate") or {}
                if isinstance(estimate, dict):
                    estimate_value = estimate.get("value")

            pipeline_name = None
            if isinstance(zh_issue, dict):
                pipeline = zh_issue.get("pipeline") or {}
                if isinstance(pipeline, dict):
                    pipeline_name = pipeline.get("name")

            is_epic = 1 if (zh_issue.get("is_epic") is True) else 0 if isinstance(zh_issue, dict) else 0

            conn.execute(
                """
                INSERT OR REPLACE INTO zenhub_issues (
                    repo, issue_number, estimate_value, pipeline_name, is_epic, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_full,
                    issue_number,
                    float(estimate_value) if estimate_value is not None else None,
                    pipeline_name,
                    is_epic,
                    json.dumps(zh_issue),
                ),
            )
        conn.commit()

        backlogs.append(
            SprintBacklog(
                sprint_id=sprint_id,
                repo=repo_full,
                start_date=start_dt,
                end_date=end_dt,
                issue_numbers=sorted(set(issue_numbers)),
            )
        )

        logger.info("Sprint %s: %s issues", sprint_id, len(issue_numbers))

    conn.close()
    return backlogs


def load_issue_estimates(db_path, repo: str, issue_numbers: list[int]) -> dict[int, float | None]:
    conn = connect(db_path)
    init_schema(conn)
    placeholders = ",".join("?" for _ in issue_numbers) if issue_numbers else "NULL"
    rows = conn.execute(
        f"""
        SELECT issue_number, estimate_value
        FROM zenhub_issues
        WHERE repo = ? AND issue_number IN ({placeholders})
        """,
        [repo] + issue_numbers,
    ).fetchall()
    conn.close()

    out: dict[int, float | None] = {n: None for n in issue_numbers}
    for issue_number, estimate_value in rows:
        out[int(issue_number)] = float(estimate_value) if estimate_value is not None else None
    return out
