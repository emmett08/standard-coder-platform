from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def _read_toml(path: Path) -> dict[str, Any]:
    """Read TOML into a dict, supporting Python 3.10+.

    Uses tomllib when available, falls back to tomli.
    """
    data = path.read_bytes()
    try:
        import tomllib  # type: ignore[attr-defined]

        return tomllib.loads(data.decode("utf-8"))
    except ModuleNotFoundError:
        import tomli  # type: ignore[import-not-found]

        return tomli.loads(data)


class PatternMatcher(BaseModel):
    """Match strings using include/exclude lists.

    Patterns may be:
    - plain substrings (case-insensitive)
    - regular expressions when prefixed with 're:'
    """

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    def matches(self, value: str) -> bool:
        v = value.lower()
        if self.include:
            ok = any(_match_one(v, p) for p in self.include)
            if not ok:
                return False
        if self.exclude:
            bad = any(_match_one(v, p) for p in self.exclude)
            if bad:
                return False
        return True


def _match_one(value_lower: str, pattern: str) -> bool:
    p = pattern.strip()
    if p.startswith("re:"):
        rx = re.compile(p[3:], flags=re.IGNORECASE)
        return rx.search(value_lower) is not None
    return p.lower() in value_lower


class RepoDiscoveryConfig(BaseModel):
    """How to find repositories on disk."""

    include_paths: list[str] = Field(
        default_factory=list,
        description="Glob patterns of local repo paths, e.g. '~/code/*'.",
    )
    exclude_paths: list[str] = Field(
        default_factory=list,
        description="Glob patterns to ignore.",
    )
    max_commits_per_repo: int = Field(
        default=20000,
        description="Safety cap. Set higher if you need deeper history.",
    )


class FiltersConfig(BaseModel):
    """Filters applied to mined commits."""

    authors: PatternMatcher = Field(default_factory=PatternMatcher)
    repos: PatternMatcher = Field(default_factory=PatternMatcher)
    languages: PatternMatcher = Field(default_factory=lambda: PatternMatcher(include=["python"]))


class OutputConfig(BaseModel):
    base_dir: str = Field(default="~/standard_coder_outputs")
    raw_dir: str = Field(default="raw")
    features_dir: str = Field(default="features")
    labels_dir: str = Field(default="labels")
    models_dir: str = Field(default="models")
    forecasts_dir: str = Field(default="forecasts")
    reports_dir: str = Field(default="reports")
    logs_dir: str = Field(default="logs")
    checkpoints_dir: str = Field(default="checkpoints")

    def resolve(self) -> "ResolvedOutputPaths":
        base = _expand(self.base_dir)
        return ResolvedOutputPaths(
            base_dir=base,
            raw_dir=base / self.raw_dir,
            features_dir=base / self.features_dir,
            labels_dir=base / self.labels_dir,
            models_dir=base / self.models_dir,
            forecasts_dir=base / self.forecasts_dir,
            reports_dir=base / self.reports_dir,
            logs_dir=base / self.logs_dir,
            checkpoints_dir=base / self.checkpoints_dir,
        )


class HmmConfig(BaseModel):
    kind: str = Field(default="hmm", description="hmm | hawkes | state_space")
    step_minutes: int = Field(default=5)
    epochs: int = Field(default=250)
    lr: float = Field(default=1e-2)
    hidden_size: int = Field(default=16)
    min_commits_per_author: int = Field(default=50)
    device: str | None = Field(default=None, description="mps recommended on Apple Silicon.")


class ChangeRepConfig(BaseModel):
    use_upgrade_a: bool = Field(default=True)
    tokens_max_features: int = Field(default=512)
    tokens_min_df: int = Field(default=2)


class MdnConfig(BaseModel):
    n_components: int = Field(default=10)
    hidden_sizes: tuple[int, ...] = Field(default=(128, 64, 64))
    epochs: int = Field(default=60)
    lr: float = Field(default=1e-3)
    batch_size: int = Field(default=512)
    truncate_low: float | None = Field(default=0.0)
    truncate_high: float | None = Field(default=1.0)
    device: str | None = Field(default=None)


class QuantileConfig(BaseModel):
    quantiles: tuple[float, ...] = Field(default=(0.5, 0.8, 0.9))
    hidden_sizes: tuple[int, ...] = Field(default=(128, 64, 64))
    epochs: int = Field(default=60)
    lr: float = Field(default=1e-3)
    batch_size: int = Field(default=512)
    device: str | None = Field(default=None)


class MultiTaskConfig(BaseModel):
    enable: bool = Field(default=False, description="Enable multi-task outputs (Upgrade C).")
    pull_requests_path: str = Field(
        default="",
        description="Optional JSON file with PR metadata for delivery-effort labels.",
    )
    epochs: int = Field(default=60)
    lr: float = Field(default=1e-3)
    batch_size: int = Field(default=256)
    device: str | None = Field(default=None)


class TrainingConfig(BaseModel):
    label_samples_per_commit: int = Field(default=50)
    rng_seed: int = Field(default=123)
    hmm: HmmConfig = Field(default_factory=HmmConfig)
    change_rep: ChangeRepConfig = Field(default_factory=ChangeRepConfig)
    mdn: MdnConfig = Field(default_factory=MdnConfig)
    quantile: QuantileConfig = Field(default_factory=QuantileConfig)
    multitask: MultiTaskConfig = Field(default_factory=MultiTaskConfig)
    predictor_kind: str = Field(default="mdn", description="mdn | quantile | multitask")


class CheckpointConfig(BaseModel):
    save_every_epochs: int = Field(default=1)
    resume: bool = Field(default=True)


class ForecastingConfig(BaseModel):
    enable: bool = Field(default=True)
    n_sims: int = Field(default=5000)
    rng_seed: int = Field(default=123)
    sprint_inputs_path: str = Field(
        default="",
        description="Path to sprint inputs JSON (config, work items, capacity).",
    )
    historical_outcomes_path: str = Field(
        default="",
        description="Path to historical throughput samples JSON.",
    )


class PipelineConfig(BaseModel):
    repos: RepoDiscoveryConfig = Field(default_factory=RepoDiscoveryConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        raw = _read_toml(path)
        return cls.model_validate(raw)


class ResolvedOutputPaths(BaseModel):
    base_dir: Path
    raw_dir: Path
    features_dir: Path
    labels_dir: Path
    models_dir: Path
    forecasts_dir: Path
    reports_dir: Path
    logs_dir: Path
    checkpoints_dir: Path

    class Config:
        arbitrary_types_allowed = True

    def ensure_dirs(self) -> None:
        for p in (
            self.base_dir,
            self.raw_dir,
            self.features_dir,
            self.labels_dir,
            self.models_dir,
            self.forecasts_dir,
            self.reports_dir,
            self.logs_dir,
            self.checkpoints_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
