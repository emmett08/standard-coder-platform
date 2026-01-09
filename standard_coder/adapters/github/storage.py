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
        CREATE TABLE IF NOT EXISTS pull_requests (
            repo TEXT NOT NULL,
            number INTEGER NOT NULL,
            pr_id INTEGER,
            state TEXT,
            title TEXT,
            author_login TEXT,
            created_at TEXT,
            updated_at TEXT,
            closed_at TEXT,
            merged_at TEXT,
            additions INTEGER,
            deletions INTEGER,
            changed_files INTEGER,
            comments INTEGER,
            review_comments INTEGER,
            commits_count INTEGER,
            PRIMARY KEY (repo, number)
        );

        CREATE TABLE IF NOT EXISTS pr_commits (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            sha TEXT NOT NULL,
            PRIMARY KEY (repo, pr_number, sha)
        );

        CREATE TABLE IF NOT EXISTS pr_reviews (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            review_id INTEGER NOT NULL,
            author_login TEXT,
            state TEXT,
            submitted_at TEXT,
            PRIMARY KEY (repo, pr_number, review_id)
        );

        CREATE TABLE IF NOT EXISTS pr_check_runs (
            repo TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            name TEXT,
            conclusion TEXT,
            started_at TEXT,
            completed_at TEXT,
            PRIMARY KEY (repo, pr_number, run_id)
        );
        """
    )
    conn.commit()
