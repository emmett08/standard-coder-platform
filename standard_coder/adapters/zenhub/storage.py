from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS zenhub_issues (
            repo TEXT NOT NULL,
            issue_number INTEGER NOT NULL,
            estimate_value REAL,
            pipeline_name TEXT,
            is_epic INTEGER,
            raw_json TEXT,
            PRIMARY KEY (repo, issue_number)
        );

        CREATE TABLE IF NOT EXISTS zenhub_milestones (
            repo TEXT NOT NULL,
            milestone_id INTEGER NOT NULL,
            title TEXT,
            start_date TEXT,
            end_date TEXT,
            raw_json TEXT,
            PRIMARY KEY (repo, milestone_id)
        );
        """
    )
    conn.commit()
