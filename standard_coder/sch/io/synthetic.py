from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import unified_diff
from typing import Sequence

from standard_coder.sch.domain.entities import Commit, PullRequest


@dataclass(frozen=True)
class SyntheticConfig:
    n_authors: int = 10
    days: int = 35
    step_minutes: int = 5
    commit_prob_per_step: float = 0.06
    seed: int = 7


def _make_base_module() -> str:
    return "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "",
            "def sub(a, b):",
            "    return a - b",
            "",
        ]
    ) + "\n"


def _apply_random_edit(code: str, rng: random.Random) -> str:
    lines = code.splitlines()
    op = rng.choice(["add_func", "add_if", "change_op", "add_loop"])
    if op == "add_func":
        n = rng.randint(1, 9999)
        lines.extend(
            [
                "",
                f"def f{n}(x):",
                "    y = x * 2",
                "    return y",
            ]
        )
    elif op == "add_if":
        lines.extend(
            [
                "",
                "def check(x):",
                "    if x >= 0:",
                "        return True",
                "    return False",
            ]
        )
    elif op == "change_op":
        # Swap a comparison or operator if present.
        joined = "\n".join(lines)
        if ">=" in joined:
            joined = joined.replace(">=", ">", 1)
        elif "==" in joined:
            joined = joined.replace("==", "!=", 1)
        else:
            joined = joined.replace("+", "-", 1) if "+" in joined else joined
        lines = joined.splitlines()
    else:  # add_loop
        lines.extend(
            [
                "",
                "def sum_to(n):",
                "    s = 0",
                "    for i in range(n):",
                "        s += i",
                "    return s",
            ]
        )
    return "\n".join(lines) + "\n"


def _diff(before: str, after: str) -> str:
    b = before.splitlines(keepends=True)
    a = after.splitlines(keepends=True)
    d = unified_diff(b, a, fromfile="before.py", tofile="after.py")
    return "".join(d)


def generate_synthetic_commits(cfg: SyntheticConfig) -> tuple[list[Commit], list[PullRequest]]:
    rng = random.Random(cfg.seed)
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=cfg.days)

    commits: list[Commit] = []
    prs: list[PullRequest] = []

    for author_i in range(cfg.n_authors):
        author_id = f"dev{author_i}@example.com"
        module_code = _make_base_module()

        commit_ids: list[str] = []
        commit_times: list[datetime] = []

        t = start
        step = timedelta(minutes=cfg.step_minutes)

        # Simulate bursts in office hours, with weekends mostly idle.
        while t < now:
            is_workday = t.weekday() < 5
            office = 9 <= t.hour < 17
            if is_workday and office and rng.random() < cfg.commit_prob_per_step:
                before = module_code
                after = _apply_random_edit(before, rng)
                module_code = after

                cid = f"{author_i}-{int(t.timestamp())}"
                commit_ids.append(cid)
                commit_times.append(t)

                commits.append(
                    Commit(
                        commit_id=cid,
                        author_id=author_id,
                        authored_at=t,
                        language="python",
                        changed_files=("module.py",),
                        diff_text=_diff(before, after),
                        before_blobs={"module.py": before},
                        after_blobs={"module.py": after},
                    )
                )
            t += step

        # Group commits into PRs (simple).
        for pr_i in range(0, len(commit_ids), 8):
            chunk = commit_ids[pr_i : pr_i + 8]
            if not chunk:
                continue
            opened_at = commit_times[pr_i]
            last_idx = min(pr_i + len(chunk) - 1, len(commit_times) - 1)
            merged_at = commit_times[last_idx] + timedelta(hours=rng.uniform(2, 24))

            prs.append(
                PullRequest(
                    pr_id=f"PR-{author_i}-{pr_i}",
                    repo="synthetic/repo",
                    author_id=author_id,
                    opened_at=opened_at,
                    merged_at=merged_at,
                    commits=tuple(chunk),
                    review_rounds=rng.randint(0, 3),
                    review_comments=rng.randint(0, 20),
                    ci_runs=rng.randint(1, 6),
                    ci_failures=rng.randint(0, 2),
                )
            )

    # Sort for reproducibility
    commits.sort(key=lambda c: c.authored_at)
    prs.sort(key=lambda p: p.opened_at)
    return commits, prs
