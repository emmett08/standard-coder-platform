from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.pipelines.dataset_builder import SchDatasetBuilder


@dataclass
class _DummyChangeRep:
    n_features: int = 3

    def fit(self, commits: list[Commit]) -> None:
        return

    def transform(self, commits: list[Commit]) -> np.ndarray:
        return np.ones((len(commits), self.n_features), dtype=np.float32)


@dataclass
class _DummyWorkInference:
    def fit(self, commit_times_by_author: dict[str, list[int]]) -> None:
        return

    def infer_coding_minutes(
        self,
        author_id: str,
        parent_minute: int,
        child_minute: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        minutes = max(0, child_minute - parent_minute)
        return np.full((n_samples,), minutes, dtype=np.float32)


def test_build_coding_dataset_emits_progress() -> None:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    commits: list[Commit] = []
    for author_id in ("a1", "a2"):
        for i in range(3):
            commits.append(
                Commit(
                    commit_id=f"{author_id}-{i}",
                    author_id=author_id,
                    authored_at=base + timedelta(minutes=10 * i),
                    language="python",
                    changed_files=("f.py",),
                    diff_text="",
                )
            )

    events: dict[str, tuple[int, int]] = {}

    def on_progress(phase: str, completed: int, total: int) -> None:
        events[phase] = (completed, total)

    builder = SchDatasetBuilder(
        work_inference=_DummyWorkInference(),
        change_rep=_DummyChangeRep(),
        label_samples_per_commit=2,
        rng_seed=123,
    )
    ds = builder.build_coding_dataset(commits, fit_work_inference=False, on_progress=on_progress)

    # 2 authors * (3 commits -> 2 pairs) = 4 pairs total.
    assert events["label_pairs"] == (4, 4)
    assert events["change_rep_fit"] == (1, 1)
    assert events["change_rep_transform"] == (1, 1)
    assert ds.x.shape == (8, 3)  # 4 pairs * 2 samples each
    assert ds.y_coding_hours.shape == (8,)
    assert ds.origin_commit_ids is not None
    assert len(ds.origin_commit_ids) == 8


@dataclass
class _SelectiveWorkInference(_DummyWorkInference):
    fitted: set[str]

    def fitted_author_ids(self) -> set[str]:
        return set(self.fitted)

    def infer_coding_minutes(
        self,
        author_id: str,
        parent_minute: int,
        child_minute: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if author_id not in self.fitted:
            raise KeyError(f"Author not fitted: {author_id}")
        return super().infer_coding_minutes(author_id, parent_minute, child_minute, n_samples, rng)


def test_build_coding_dataset_skips_unfitted_authors() -> None:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    commits: list[Commit] = []
    for author_id in ("a1", "a2"):
        for i in range(3):
            commits.append(
                Commit(
                    commit_id=f"{author_id}-{i}",
                    author_id=author_id,
                    authored_at=base + timedelta(minutes=10 * i),
                    language="python",
                    changed_files=("f.py",),
                    diff_text="",
                )
            )

    builder = SchDatasetBuilder(
        work_inference=_SelectiveWorkInference(fitted={"a1"}),
        change_rep=_DummyChangeRep(),
        label_samples_per_commit=2,
        rng_seed=123,
    )
    ds = builder.build_coding_dataset(commits, fit_work_inference=False)

    # Only author a1 contributes: (3 commits -> 2 pairs) * 2 samples = 4 rows.
    assert ds.x.shape == (4, 3)
    assert ds.y_coding_hours.shape == (4,)
    assert ds.origin_commit_ids is not None
    assert all(cid.startswith("a1-") for cid in ds.origin_commit_ids)
