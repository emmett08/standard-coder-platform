from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from standard_coder.forecasting.services.forecasting_service import ForecastingService
from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.pipelines.training import SchScoringService


class ScoreRequest(BaseModel):
    work_item_id: str = Field(..., description="Ticket/PR/commit identifier")
    author_id: str = Field("unknown")
    language: str = Field("unknown")
    diff_text: str
    changed_files: list[str] = Field(default_factory=list)
    before_blobs: dict[str, str] | None = None
    after_blobs: dict[str, str] | None = None


class ScoreResponse(BaseModel):
    work_item_id: str
    unit: str
    model_version: str
    mean: float
    p50: float
    p90: float
    metadata: dict[str, Any] | None = None


def create_app(
    scoring: SchScoringService,
    forecasting: ForecastingService | None = None,
) -> FastAPI:
    app = FastAPI(title="Standard Coder Platform")

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        c = Commit(
            commit_id=req.work_item_id,
            author_id=req.author_id,
            authored_at=__import__("datetime").datetime.utcnow(),
            language=req.language,
            changed_files=tuple(req.changed_files),
            diff_text=req.diff_text,
            before_blobs=req.before_blobs,
            after_blobs=req.after_blobs,
        )
        est = scoring.score_commits([c])[0]
        if scoring.bus is not None:
            scoring.publish_work_item_estimates([req.work_item_id], [est])
        return ScoreResponse(
            work_item_id=req.work_item_id,
            unit=est.unit,
            model_version=est.model_version,
            mean=float(est.mean()),
            p50=float(est.p50()),
            p90=float(est.p90()),
            metadata=dict(est.metadata) if est.metadata is not None else None,
        )

    if forecasting is not None:

        @app.get("/forecast/{sprint_id}")
        def forecast(sprint_id: str) -> dict[str, Any]:
            results = forecasting.compute_forecast(sprint_id=sprint_id)
            return {k: asdict(v) for k, v in results.items()}

    return app
