from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from standard_coder.integration.event_bus import EventBus
from standard_coder.integration.events import WorkItemEffortEstimated
from standard_coder.sch.domain.entities import Commit, PullRequest
from standard_coder.sch.domain.value_objects import EffortEstimate, MultiTaskEffortEstimate
from standard_coder.sch.interfaces import (
    ChangeRepModel,
    EffortPredictor,
    MultiTaskEffortPredictor,
    WorkInferenceModel,
)
from standard_coder.sch.pipelines.dataset_builder import SchDatasetBuilder


@dataclass
class SchTrainingPipeline:
    """Orchestrate training of SCH predictors."""

    work_inference: WorkInferenceModel
    change_rep: ChangeRepModel

    def train_single_task(
        self,
        commits: Sequence[Commit],
        predictor: EffortPredictor,
        label_samples_per_commit: int = 50,
        rng_seed: int = 123,
    ) -> tuple[ChangeRepModel, EffortPredictor]:
        builder = SchDatasetBuilder(
            work_inference=self.work_inference,
            change_rep=self.change_rep,
            label_samples_per_commit=label_samples_per_commit,
            rng_seed=rng_seed,
        )
        ds = builder.build_coding_dataset(commits)
        predictor.fit(ds.x, ds.y_coding_hours)
        return self.change_rep, predictor

    def train_multi_task(
        self,
        commits: Sequence[Commit],
        pull_requests: Sequence[PullRequest],
        predictor: MultiTaskEffortPredictor,
        label_samples_per_commit: int = 50,
        rng_seed: int = 123,
    ) -> tuple[ChangeRepModel, MultiTaskEffortPredictor]:
        builder = SchDatasetBuilder(
            work_inference=self.work_inference,
            change_rep=self.change_rep,
            label_samples_per_commit=label_samples_per_commit,
            rng_seed=rng_seed,
        )
        ds = builder.build_multitask_dataset(commits, pull_requests)
        assert ds.y_delivery_hours is not None
        predictor.fit(ds.x, ds.y_coding_hours, ds.y_delivery_hours)
        return self.change_rep, predictor


@dataclass
class SchScoringService:
    """Score work items (commits/tickets) and publish effort events."""

    change_rep: ChangeRepModel
    predictor: EffortPredictor
    bus: EventBus | None = None

    def score_commits(self, commits: Sequence[Commit]) -> Sequence[EffortEstimate]:
        x = self.change_rep.transform(commits)
        return self.predictor.predict(x)

    def publish_work_item_estimates(
        self,
        work_item_ids: Sequence[str],
        estimates: Sequence[EffortEstimate],
        occurred_at: datetime | None = None,
    ) -> None:
        if self.bus is None:
            return
        occurred_at = occurred_at or datetime.utcnow()
        for wid, est in zip(work_item_ids, estimates, strict=True):
            payload = _serialise_effort(est)
            self.bus.publish(
                WorkItemEffortEstimated(
                    occurred_at=occurred_at,
                    work_item_id=wid,
                    effort=payload,
                    model_version=est.model_version,
                )
            )


def _serialise_effort(est: EffortEstimate) -> dict[str, object]:
    """Serialise to a stable event payload."""
    dist = est.distribution
    cls = type(dist).__name__

    if cls == "QuantileDistribution":
        q = getattr(dist, "quantiles")
        return {
            "type": "quantiles",
            "quantiles": {str(k): float(v) for k, v in q.items()},
        }

    if cls == "GaussianMixtureDistribution":
        return {
            "type": "gmm",
            "weights": [float(x) for x in getattr(dist, "weights")],
            "means": [float(x) for x in getattr(dist, "means")],
            "stds": [float(x) for x in getattr(dist, "stds")],
            "low": getattr(dist, "low"),
            "high": getattr(dist, "high"),
        }

    raise TypeError(f"Unsupported distribution type: {cls}")
