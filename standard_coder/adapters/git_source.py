from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.interfaces import CommitSource


def _run_git(repo: Path, args: list[str]) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return res.stdout


def _infer_language(path: str) -> str:
    if path.endswith(".py"):
        return "python"
    if path.endswith(".java"):
        return "java"
    if path.endswith(".js"):
        return "javascript"
    return "unknown"


@dataclass
class GitCommitSource(CommitSource):
    """Read commits from a local git repository using the git CLI."""

    repo_path: Path
    max_commits: int = 500

    def iter_commits(self) -> Iterable[Commit]:
        # Format: hash|author_email|author_time_epoch
        log = _run_git(
            self.repo_path,
            [
                "log",
                f"-n{self.max_commits}",
                "--pretty=format:%H|%ae|%at",
                "--reverse",
            ],
        ).strip()

        if not log:
            return

        lines = log.splitlines()
        for line in lines:
            commit_id, author_email, authored_at = line.split("|", 2)
            authored_dt = datetime.fromtimestamp(int(authored_at), tz=timezone.utc)

            # Get file list
            name_status = _run_git(
                self.repo_path,
                ["show", "--name-only", "--pretty=format:", commit_id],
            ).strip()
            files = [f for f in name_status.splitlines() if f.strip()]

            diff_text = _run_git(
                self.repo_path,
                ["show", "--pretty=format:", "--unified=0", commit_id],
            )

            language = "unknown"
            if files:
                language = _infer_language(files[0])

            before_blobs: dict[str, str] = {}
            after_blobs: dict[str, str] = {}
            if language == "python":
                parent = f"{commit_id}^"
                for f in files:
                    if not f.endswith(".py"):
                        continue
                    try:
                        before_blobs[f] = _run_git(self.repo_path, ["show", f"{parent}:{f}"])
                    except Exception:
                        before_blobs[f] = ""
                    try:
                        after_blobs[f] = _run_git(self.repo_path, ["show", f"{commit_id}:{f}"])
                    except Exception:
                        after_blobs[f] = ""

            yield Commit(
                commit_id=commit_id,
                author_id=author_email,
                authored_at=authored_dt,
                language=language,
                changed_files=tuple(files),
                diff_text=diff_text,
                before_blobs=before_blobs if before_blobs else None,
                after_blobs=after_blobs if after_blobs else None,
            )
