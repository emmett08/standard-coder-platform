from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from standard_coder.pipeline.git_io import MinedCommit


@dataclass(frozen=True)
class RawStore:
    db_path: Path

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS commits (
                repo_path TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                parent_id TEXT,
                author_id TEXT NOT NULL,
                authored_at INTEGER NOT NULL,
                language TEXT NOT NULL,
                changed_files TEXT NOT NULL,
                PRIMARY KEY (repo_path, commit_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_author ON commits(author_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lang ON commits(language)")
        conn.commit()
        return conn

    def upsert_mined_commit(self, conn: sqlite3.Connection, c: MinedCommit) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO commits
            (repo_path, commit_id, parent_id, author_id, authored_at, language, changed_files)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(c.repo_path),
                c.commit_id,
                c.parent_id,
                c.author_id,
                int(c.authored_at.timestamp()),
                c.language,
                "\n".join(c.changed_files),
            ),
        )

    def iter_mined(self, conn: sqlite3.Connection) -> Iterator[MinedCommit]:
        cur = conn.execute(
            """
            SELECT repo_path, commit_id, parent_id, author_id, authored_at, language, changed_files
            FROM commits
            ORDER BY repo_path, authored_at, commit_id
            """
        )
        for repo_path, commit_id, parent_id, author_id, authored_at, language, changed_files in cur:
            yield MinedCommit(
                repo_path=Path(repo_path),
                commit_id=commit_id,
                parent_id=parent_id,
                author_id=author_id,
                authored_at=datetime.fromtimestamp(int(authored_at), tz=timezone.utc),
                changed_files=tuple((changed_files or "").splitlines()),
                language=language,
            )
