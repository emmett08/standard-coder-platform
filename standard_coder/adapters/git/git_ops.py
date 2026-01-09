from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitRepoSpec:
    name: str  # e.g. owner/repo or local name
    url: str
    local_path: Path


class GitError(RuntimeError):
    pass


def _run_git(repo_path: Path, args: list[str]) -> str:
    cmd = ["git", "-C", str(repo_path)] + args
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        msg = exc.output.decode("utf-8", errors="replace")
        raise GitError(f"git failed: {' '.join(cmd)}\n{msg}") from exc


def clone_or_fetch_repo(
    spec: GitRepoSpec,
    *,
    mirror: bool = True,
    prune: bool = True,
) -> None:
    if spec.local_path.exists():
        logger.info("Fetching %s", spec.name)
        fetch_args = ["fetch", "--all"]
        if prune:
            fetch_args.append("--prune")
        _run_git(spec.local_path, fetch_args)
        return

    spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s", spec.name)
    if mirror:
        subprocess.check_call(["git", "clone", "--mirror", spec.url, str(spec.local_path)])
    else:
        subprocess.check_call(["git", "clone", spec.url, str(spec.local_path)])
