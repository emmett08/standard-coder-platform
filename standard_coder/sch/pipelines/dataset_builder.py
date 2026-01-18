from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

import numpy as np

from standard_coder.sch.domain.entities import Commit, PullRequest
from standard_coder.sch.interfaces import ChangeRepModel, WorkInferenceModel


def _to_minute(dt: datetime) -> int:
    return int(dt.timestamp() // 60)


@dataclass
class SchTrainingDataset:
    x: np.ndarray
    y_coding_hours: np.ndarray
    y_delivery_hours: np.ndarray | None = None
    origin_commit_ids: tuple[str, ...] | None = None


@dataclass
class SchDatasetBuilder:
    """Build supervised datasets for SCH predictors."""

    work_inference: WorkInferenceModel
    change_rep: ChangeRepModel
    label_samples_per_commit: int = 50
    rng_seed: int = 123

    def build_coding_dataset(
        self,
        commits: Sequence[Commit],
        *,
        fit_work_inference: bool = True,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> SchTrainingDataset:
        rng = np.random.default_rng(self.rng_seed)

        # Fit change representation.
        if on_progress is not None:
            on_progress("change_rep_fit", 0, 1)
        self.change_rep.fit(commits)
        if on_progress is not None:
            on_progress("change_rep_fit", 1, 1)

        # Fit work inference per author based on commit history.
        times_by_author: dict[str, list[int]] = {}
        commits_by_author: dict[str, list[Commit]] = {}
        for c in commits:
            times_by_author.setdefault(c.author_id, []).append(_to_minute(c.authored_at))
            commits_by_author.setdefault(c.author_id, []).append(c)

        if fit_work_inference:
            self.work_inference.fit(times_by_author)

        fitted_authors: set[str] | None = None
        if hasattr(self.work_inference, "fitted_author_ids"):
            fitted_authors = self.work_inference.fitted_author_ids()  # type: ignore[attr-defined]

        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        origin_commit_ids: list[str] = []

        # Prepare feature vectors per commit for replication.
        if on_progress is not None:
            on_progress("change_rep_transform", 0, 1)
        x_commit = self.change_rep.transform(commits)
        if on_progress is not None:
            on_progress("change_rep_transform", 1, 1)

        # Map commit_id -> index
        idx_by_commit = {c.commit_id: i for i, c in enumerate(commits)}

        total_pairs = sum(
            max(0, len(c_list) - 1)
            for author_id, c_list in commits_by_author.items()
            if fitted_authors is None or author_id in fitted_authors
        )
        if on_progress is not None:
            on_progress("label_pairs", 0, total_pairs)
        pair_i = 0

        for author_id, c_list in commits_by_author.items():
            if fitted_authors is not None and author_id not in fitted_authors:
                continue
            c_sorted = sorted(c_list, key=lambda c: c.authored_at)
            for j in range(1, len(c_sorted)):
                parent = c_sorted[j - 1]
                child = c_sorted[j]
                parent_m = _to_minute(parent.authored_at)
                child_m = _to_minute(child.authored_at)

                try:
                    samples_minutes = self.work_inference.infer_coding_minutes(
                        author_id=author_id,
                        parent_minute=parent_m,
                        child_minute=child_m,
                        n_samples=self.label_samples_per_commit,
                        rng=rng,
                    )
                except KeyError:
                    # Work inference may skip authors (e.g. min_commits threshold).
                    # Silently drop these examples to keep dataset building robust.
                    pair_i += 1
                    if on_progress is not None and (pair_i % 50 == 0 or pair_i == total_pairs):
                        on_progress("label_pairs", pair_i, total_pairs)
                    continue

                y_hours = (samples_minutes / 60.0).astype(np.float32)
                # The paper truncates to [0, 1] hour at prediction time; we
                # optionally apply the same truncation for training stability.
                y_hours = np.clip(y_hours, 0.0, 1.0)

                x0 = x_commit[idx_by_commit[child.commit_id]]
                for yh in y_hours:
                    x_rows.append(x0)
                    y_rows.append(float(yh))
                    origin_commit_ids.append(child.commit_id)

                pair_i += 1
                if on_progress is not None and (pair_i % 50 == 0 or pair_i == total_pairs):
                    on_progress("label_pairs", pair_i, total_pairs)

        x = np.vstack(x_rows).astype(np.float32)
        y = np.array(y_rows, dtype=np.float32)
        return SchTrainingDataset(x=x, y_coding_hours=y, origin_commit_ids=tuple(origin_commit_ids))

    def build_multitask_dataset(
        self,
        commits: Sequence[Commit],
        pull_requests: Sequence[PullRequest],
        *,
        fit_work_inference: bool = True,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> SchTrainingDataset:
        """Build a multi-task dataset: coding SCH + delivery effort.

        Delivery effort is approximated from PR cycle-time components.
        This is intentionally simple; in production you would calibrate this
        to your organisation's definition of delivery effort.
        """
        base = self.build_coding_dataset(commits, fit_work_inference=fit_work_inference, on_progress=on_progress)

        pr_by_commit: dict[str, PullRequest] = {}
        for pr in pull_requests:
            for cid in pr.commits:
                pr_by_commit[cid] = pr

        # Derive a delivery label for each replicated sample based on the
        # PR associated with the commit. If no PR, assume delivery ~= coding.
        y_delivery: list[float] = []
        rng = np.random.default_rng(self.rng_seed + 1)

        # Now compute delivery labels per row.
        if base.origin_commit_ids is None:
            raise RuntimeError("Missing origin_commit_ids; expected build_coding_dataset() to populate it.")
        for cid, y_coding in zip(base.origin_commit_ids, base.y_coding_hours, strict=True):
            pr = pr_by_commit.get(cid)
            if pr is None or pr.merged_at is None:
                # Delivery effort ~= coding effort (baseline).
                y_delivery.append(float(y_coding))
                continue

            cycle_hours = max(0.0, (pr.merged_at - pr.opened_at).total_seconds() / 3600.0)
            review_penalty = 0.05 * float(pr.review_rounds) + 0.01 * float(pr.review_comments)
            ci_penalty = 0.02 * float(pr.ci_failures)

            # Convert cycle time to a bounded "delivery effort" proxy.
            # This is intentionally conservative and should be calibrated.
            delivery = min(1.0, float(y_coding) + 0.2 * min(1.0, cycle_hours / 48.0) + review_penalty + ci_penalty)

            # Add small noise to avoid degeneracy.
            delivery = float(np.clip(delivery + rng.normal(0.0, 0.02), 0.0, 1.0))
            y_delivery.append(delivery)

        return SchTrainingDataset(
            x=base.x,
            y_coding_hours=base.y_coding_hours,
            y_delivery_hours=np.array(y_delivery, dtype=np.float32),
            origin_commit_ids=base.origin_commit_ids,
        )
