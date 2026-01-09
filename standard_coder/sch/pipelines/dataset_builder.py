from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

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


@dataclass
class SchDatasetBuilder:
    """Build supervised datasets for SCH predictors."""

    work_inference: WorkInferenceModel
    change_rep: ChangeRepModel
    label_samples_per_commit: int = 50
    rng_seed: int = 123

    def build_coding_dataset(self, commits: Sequence[Commit]) -> SchTrainingDataset:
        rng = np.random.default_rng(self.rng_seed)

        # Fit change representation.
        self.change_rep.fit(commits)

        # Fit work inference per author based on commit history.
        times_by_author: dict[str, list[int]] = {}
        commits_by_author: dict[str, list[Commit]] = {}
        for c in commits:
            times_by_author.setdefault(c.author_id, []).append(_to_minute(c.authored_at))
            commits_by_author.setdefault(c.author_id, []).append(c)

        self.work_inference.fit(times_by_author)

        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []

        # Prepare feature vectors per commit for replication.
        x_commit = self.change_rep.transform(commits)

        # Map commit_id -> index
        idx_by_commit = {c.commit_id: i for i, c in enumerate(commits)}

        for author_id, c_list in commits_by_author.items():
            c_sorted = sorted(c_list, key=lambda c: c.authored_at)
            for j in range(1, len(c_sorted)):
                parent = c_sorted[j - 1]
                child = c_sorted[j]
                parent_m = _to_minute(parent.authored_at)
                child_m = _to_minute(child.authored_at)

                samples_minutes = self.work_inference.infer_coding_minutes(
                    author_id=author_id,
                    parent_minute=parent_m,
                    child_minute=child_m,
                    n_samples=self.label_samples_per_commit,
                    rng=rng,
                )

                y_hours = (samples_minutes / 60.0).astype(np.float32)
                # The paper truncates to [0, 1] hour at prediction time; we
                # optionally apply the same truncation for training stability.
                y_hours = np.clip(y_hours, 0.0, 1.0)

                x0 = x_commit[idx_by_commit[child.commit_id]]
                for yh in y_hours:
                    x_rows.append(x0)
                    y_rows.append(float(yh))

        x = np.vstack(x_rows).astype(np.float32)
        y = np.array(y_rows, dtype=np.float32)
        return SchTrainingDataset(x=x, y_coding_hours=y)

    def build_multitask_dataset(
        self,
        commits: Sequence[Commit],
        pull_requests: Sequence[PullRequest],
    ) -> SchTrainingDataset:
        """Build a multi-task dataset: coding SCH + delivery effort.

        Delivery effort is approximated from PR cycle-time components.
        This is intentionally simple; in production you would calibrate this
        to your organisation's definition of delivery effort.
        """
        base = self.build_coding_dataset(commits)

        pr_by_commit: dict[str, PullRequest] = {}
        for pr in pull_requests:
            for cid in pr.commits:
                pr_by_commit[cid] = pr

        # Derive a delivery label for each replicated sample based on the
        # PR associated with the commit. If no PR, assume delivery ~= coding.
        y_delivery: list[float] = []
        rng = np.random.default_rng(self.rng_seed + 1)

        # We need to match replicated rows to their originating commit. We
        # do this by rebuilding the replication with a parallel list.
        commit_origin: list[str] = []
        times_by_author: dict[str, list[int]] = {}
        commits_by_author: dict[str, list[Commit]] = {}
        for c in commits:
            times_by_author.setdefault(c.author_id, []).append(_to_minute(c.authored_at))
            commits_by_author.setdefault(c.author_id, []).append(c)

        # This assumes work_inference already fitted in build_coding_dataset().
        x_commit = self.change_rep.transform(commits)
        idx_by_commit = {c.commit_id: i for i, c in enumerate(commits)}

        for author_id, c_list in commits_by_author.items():
            c_sorted = sorted(c_list, key=lambda c: c.authored_at)
            for j in range(1, len(c_sorted)):
                child = c_sorted[j]
                x0 = x_commit[idx_by_commit[child.commit_id]]

                # Recreate the same number of samples.
                for _ in range(self.label_samples_per_commit):
                    commit_origin.append(child.commit_id)

        # Now compute delivery labels per row.
        for cid, y_coding in zip(commit_origin, base.y_coding_hours, strict=True):
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

        return SchTrainingDataset(x=base.x, y_coding_hours=base.y_coding_hours, y_delivery_hours=np.array(y_delivery, dtype=np.float32))
