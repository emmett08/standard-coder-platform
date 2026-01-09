from __future__ import annotations

import fnmatch
import glob
import gzip
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from standard_coder.sch.domain.entities import Commit


def _run_git(repo: Path, args: list[str]) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return res.stdout


def discover_repos(include_globs: list[str], exclude_globs: list[str]) -> list[Path]:
    """Discover local git repositories from glob patterns."""
    candidates: list[Path] = []
    for g in include_globs:
        expanded = os.path.expanduser(g)
        for match in glob.glob(expanded):
            p = Path(match)
            if not p.exists():
                continue
            if p.is_file():
                continue
            if (p / ".git").exists():
                candidates.append(p.resolve())

    out: list[Path] = []
    expanded_excludes = [os.path.expanduser(ex) for ex in exclude_globs]
    for p in candidates:
        excluded = any(fnmatch.fnmatch(str(p), ex) for ex in expanded_excludes)
        if not excluded:
            out.append(p)
    # Stable order for reproducibility
    return sorted(set(out))


def _infer_language_from_paths(paths: list[str]) -> str:
    exts = [Path(p).suffix.lower() for p in paths]
    if any(e == ".py" for e in exts):
        return "python"
    if any(e == ".java" for e in exts):
        return "java"
    if any(e in (".js", ".ts") for e in exts):
        return "javascript"
    return "unknown"


@dataclass(frozen=True)
class MinedCommit:
    repo_path: Path
    commit_id: str
    parent_id: str | None
    author_id: str
    authored_at: datetime
    changed_files: tuple[str, ...]
    language: str


def iter_mined_commits(repo: Path, max_commits: int) -> Iterator[MinedCommit]:
    """Yield commit metadata from a local repository.

    This stage is intentionally *lightweight* and does not store diffs
    or blobs. Later stages can cache diffs/blobs on demand.
    """
    # hash|parent|author_email|author_time_epoch
    log = _run_git(
        repo,
        [
            "log",
            f"-n{int(max_commits)}",
            "--pretty=format:%H|%P|%ae|%at",
            "--reverse",
        ],
    ).strip()

    if not log:
        return

    for line in log.splitlines():
        commit_id, parents, author_email, authored_at = line.split("|", 3)
        parent_id = parents.split()[0] if parents.strip() else None
        authored_dt = datetime.fromtimestamp(int(authored_at), tz=timezone.utc)

        name_only = _run_git(repo, ["show", "--name-only", "--pretty=format:", commit_id]).strip()
        files = tuple(f for f in name_only.splitlines() if f.strip())
        language = _infer_language_from_paths(list(files))
        yield MinedCommit(
            repo_path=repo,
            commit_id=commit_id,
            parent_id=parent_id,
            author_id=author_email,
            authored_at=authored_dt,
            changed_files=files,
            language=language,
        )


def load_commit(
    mined: MinedCommit,
    cache_dir: Path,
    include_blobs: bool,
) -> Commit:
    """Load a full Commit object, using cached diff/blobs when possible."""
    repo = mined.repo_path
    repo_slug = repo.name.replace("/", "_")
    diff_dir = cache_dir / "diffs" / repo_slug
    blob_dir = cache_dir / "blobs" / repo_slug
    diff_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    diff_path = diff_dir / f"{mined.commit_id}.patch"
    if diff_path.exists():
        diff_text = diff_path.read_text(errors="ignore")
    else:
        diff_text = _run_git(repo, ["show", "--pretty=format:", "--unified=0", mined.commit_id])
        diff_path.write_text(diff_text)

    before_blobs: dict[str, str] | None = None
    after_blobs: dict[str, str] | None = None

    if include_blobs and mined.language == "python":
        blob_path = blob_dir / f"{mined.commit_id}.json.gz"
        if blob_path.exists():
            payload = json.loads(gzip.decompress(blob_path.read_bytes()).decode("utf-8"))
            before_blobs = payload.get("before") or None
            after_blobs = payload.get("after") or None
        else:
            before_blobs = {}
            after_blobs = {}
            if mined.parent_id is not None:
                parent = mined.parent_id
            else:
                parent = f"{mined.commit_id}^"

            for f in mined.changed_files:
                if not f.endswith(".py"):
                    continue
                try:
                    before_blobs[f] = _run_git(repo, ["show", f"{parent}:{f}"])
                except Exception:
                    before_blobs[f] = ""
                try:
                    after_blobs[f] = _run_git(repo, ["show", f"{mined.commit_id}:{f}"])
                except Exception:
                    after_blobs[f] = ""
            payload = {"before": before_blobs, "after": after_blobs}
            blob_path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))

    return Commit(
        commit_id=mined.commit_id,
        author_id=mined.author_id,
        authored_at=mined.authored_at,
        language=mined.language,
        changed_files=mined.changed_files,
        diff_text=diff_text,
        before_blobs=before_blobs,
        after_blobs=after_blobs,
    )
