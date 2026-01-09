from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer

from standard_coder.common.logging_config import configure_logging
from standard_coder.pipeline.config import PipelineConfig
from standard_coder.pipeline.progress_ui import progress_ui
from standard_coder.pipeline.runner import PipelineRunner

from standard_coder.forecasting.priors.throughput import EmpiricalThroughputPrior
from standard_coder.forecasting.scenarios.scenarios import ScopeGrowthScenario, TimeOffScenario
from standard_coder.forecasting.services.forecasting_service import ForecastingService
from standard_coder.integration.event_bus import InMemoryEventBus
from standard_coder.integration.events import (
    AvailabilityChanged,
    ScopeChanged,
    SprintConfigured,
    WorkItemAddedToSprint,
    WorkItemStatusChanged,
)
from standard_coder.sch.features.ast_features import AstEditChangeRep
from standard_coder.sch.features.graph_features import GraphDeltaChangeRep
from standard_coder.sch.features.hybrid import ConcatenatedChangeRep
from standard_coder.sch.features.token_features import BagOfTokensChangeRep
from standard_coder.sch.io.synthetic import SyntheticConfig, generate_synthetic_commits
from standard_coder.sch.pipelines.training import SchScoringService, SchTrainingPipeline
from standard_coder.sch.predictors.mdn import MdnEffortPredictor
from standard_coder.sch.predictors.multitask import HeadConfig, TorchMultiTaskEffortPredictor
from standard_coder.sch.predictors.quantile import QuantileEffortPredictor
from standard_coder.sch.work_inference.hawkes import HawkesWorkInference
from standard_coder.sch.work_inference.neural_hmm import NeuralHmmWorkInference
from standard_coder.sch.work_inference.state_space import VariationalStateSpaceWorkInference


app = typer.Typer(add_completion=False)


@app.command()
def demo(
    seed: int = typer.Option(7, help="Random seed for synthetic data"),
    work_inference: str = typer.Option(
        "hmm",
        help="Work inference model: hmm | hawkes | state_space",
    ),
    predictor: str = typer.Option(
        "mdn",
        help="Predictor head: mdn | quantile | multitask",
    ),
    use_upgrade_a: bool = typer.Option(True, help="Enable AST + graph features (Python)"),
) -> None:
    """Run an end-to-end demo: train SCH model and forecast a sprint."""
    configure_logging(logging.INFO)

    cfg = SyntheticConfig(seed=seed)
    commits, prs = generate_synthetic_commits(cfg)
    typer.echo(f"Generated {len(commits)} commits and {len(prs)} PRs")

    # --- Select components --------------------------------------------------
    if use_upgrade_a:
        change_rep = ConcatenatedChangeRep(
            parts=(
                BagOfTokensChangeRep(max_features=256, min_df=1),
                AstEditChangeRep(),
                GraphDeltaChangeRep(),
            )
        )
    else:
        change_rep = BagOfTokensChangeRep(max_features=256, min_df=1)

    if work_inference == "hmm":
        wi = NeuralHmmWorkInference(step_minutes=cfg.step_minutes, epochs=120, min_commits=60)
    elif work_inference == "hawkes":
        wi = HawkesWorkInference(step_minutes=cfg.step_minutes, min_commits=60)
    elif work_inference == "state_space":
        wi = VariationalStateSpaceWorkInference(step_minutes=cfg.step_minutes, min_commits=60)
    else:
        raise typer.BadParameter("work_inference must be one of: hmm, hawkes, state_space")

    bus = InMemoryEventBus()

    # Throughput prior: synthetic daily throughput values in SCH/day.
    # In production you would fit this from historical sprint outcomes.
    team_size = cfg.n_authors
    throughput_samples = [max(1.0, team_size * 4.0 + (i % 5 - 2) * 1.5) for i in range(60)]
    prior = EmpiricalThroughputPrior(samples=tuple(throughput_samples))

    forecasting_service = ForecastingService(
        bus=bus,
        throughput_prior=prior,
        n_sims=5000,
        rng_seed=seed,
    )

    # --- Train predictor ----------------------------------------------------
    pipeline = SchTrainingPipeline(work_inference=wi, change_rep=change_rep)

    if predictor == "mdn":
        pred = MdnEffortPredictor(epochs=30, truncate_low=0.0, truncate_high=1.0)
        change_rep, pred = pipeline.train_single_task(
            commits=commits,
            predictor=pred,
            label_samples_per_commit=30,
            rng_seed=seed,
        )
        scoring = SchScoringService(change_rep=change_rep, predictor=pred, bus=bus)
    elif predictor == "quantile":
        pred_q = QuantileEffortPredictor(epochs=30, quantiles=(0.5, 0.8, 0.9))
        change_rep, pred_q = pipeline.train_single_task(
            commits=commits,
            predictor=pred_q,
            label_samples_per_commit=30,
            rng_seed=seed,
        )
        scoring = SchScoringService(change_rep=change_rep, predictor=pred_q, bus=bus)
    elif predictor == "multitask":
        mt = TorchMultiTaskEffortPredictor(
            coding_head=HeadConfig(kind="mdn", n_components=8, truncate_low=0.0, truncate_high=1.0),
            delivery_head=HeadConfig(kind="quantile", quantiles=(0.5, 0.8, 0.9)),
            epochs=40,
        )
        change_rep, mt = pipeline.train_multi_task(
            commits=commits,
            pull_requests=prs,
            predictor=mt,
            label_samples_per_commit=30,
            rng_seed=seed,
        )
        # For demo forecasting, publish coding estimates only (SCH).
        pred_single = MdnEffortPredictor(epochs=1)  # placeholder; not used.
        scoring = SchScoringService(change_rep=change_rep, predictor=pred_single, bus=bus)
        # We'll score using mt, then publish mt.coding estimates manually below.
    else:
        raise typer.BadParameter("predictor must be one of: mdn, quantile, multitask")

    # --- Configure a sprint -------------------------------------------------
    now = datetime.now(tz=timezone.utc)
    sprint_start = (now - timedelta(days=7)).date()

    bus.publish(
        SprintConfigured(
            occurred_at=datetime.utcnow(),
            sprint_id="SPRINT-1",
            start_date=sprint_start,
            length_days=14,
            working_days=(0, 1, 2, 3, 4),
        )
    )

    # Set baseline capacity ~ 6h per dev per day.
    for d in range(14):
        day = sprint_start + timedelta(days=d)
        if day.weekday() < 5:
            desired = float(team_size * 6.0)
            bus.publish(
                AvailabilityChanged(
                    occurred_at=datetime.utcnow(),
                    person_id="team",
                    day=day,
                    delta_available_sch=desired - 8.0,
                )
            )

    # Pick a subset of recent commits as sprint work items.
    sprint_commits = commits[-40:]
    work_item_ids = [c.commit_id for c in sprint_commits]

    for wid in work_item_ids:
        bus.publish(
            WorkItemAddedToSprint(
                occurred_at=datetime.utcnow(),
                sprint_id="SPRINT-1",
                work_item_id=wid,
            )
        )

    # Mark some as done and apply some scope changes.
    for wid in work_item_ids[:15]:
        bus.publish(
            WorkItemStatusChanged(
                occurred_at=datetime.utcnow(),
                work_item_id=wid,
                old_status="todo",
                new_status="done",
            )
        )

    # Slight scope growth on a few remaining items.
    for wid in work_item_ids[15:20]:
        bus.publish(
            ScopeChanged(
                occurred_at=datetime.utcnow(),
                work_item_id=wid,
                scope_factor=1.1,
            )
        )

    # --- Score remaining items and publish effort estimates -----------------
    remaining_commits = [c for c in sprint_commits if c.commit_id not in set(work_item_ids[:15])]

    if predictor != "multitask":
        estimates = scoring.score_commits(remaining_commits)
        scoring.publish_work_item_estimates(
            work_item_ids=[c.commit_id for c in remaining_commits],
            estimates=estimates,
        )
    else:
        # Multi-task predictor returns both; publish coding distributions.
        mt_est = mt.predict(change_rep.transform(remaining_commits))
        coding_only = [e.coding for e in mt_est]
        scoring.publish_work_item_estimates(
            work_item_ids=[c.commit_id for c in remaining_commits],
            estimates=coding_only,
        )

    # --- Compute forecasts (baseline + scenarios) ---------------------------
    scenarios = [
        ScopeGrowthScenario(scenario_id="scope+10%", scope_factor=1.1),
        TimeOffScenario(
            scenario_id="team_off_2_days",
            days=(sprint_start + timedelta(days=2), sprint_start + timedelta(days=3)),
            delta_available_sch=-float(team_size * 3.0),
        ),
    ]

    results = forecasting_service.compute_forecast("SPRINT-1", scenarios=scenarios)

    typer.echo("\nForecast results:")
    for sid, res in results.items():
        typer.echo(f"- {sid}: P(complete_by_end)={res.p_complete_by_end:.3f} "
                   f"p50_day={res.completion_day_p50} p90_day={res.completion_day_p90}")

    typer.echo("\nJSON output:")
    typer.echo(json.dumps({k: asdict(v) for k, v in results.items()}, indent=2, default=str))


@app.command()
def init_config(
    path: str = typer.Argument(
        "pipeline_config.toml",
        help="Where to write the pipeline configuration TOML",
    ),
) -> None:
    """Write an example pipeline_config.toml."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    template = project_root / "pipeline_config.example.toml"
    if not template.exists():
        raise RuntimeError(f"Missing template file: {template}")

    out = Path(path).expanduser()
    if out.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing file: {out}")

    out.write_text(template.read_text())
    typer.echo(f"Wrote {out} (edit it, then run: standard-coder pipeline --config {out})")


@app.command()
def pipeline(
    config: str = typer.Option("pipeline_config.toml", help="Path to pipeline_config.toml"),
    stages: str = typer.Option(
        "mine,train_sch,forecast",
        help="Comma-separated stages: mine,train_sch,forecast",
    ),
) -> None:
    """Run the resumable training pipeline with checkpoints and progress UI."""
    config_path = Path(config).expanduser()
    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    with progress_ui() as ui:
        runner = PipelineRunner.from_config_path(config_path, ui=ui)
        runner.run(stage_list)
