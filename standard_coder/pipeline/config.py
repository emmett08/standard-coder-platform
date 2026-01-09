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


def _merge_paths(primary: list[str], secondary: list[str]) -> list[str]:
    if not primary and not secondary:
        return []
    if not primary:
        return list(secondary)
    if not secondary:
        return list(primary)
    seen: set[str] = set()
    out: list[str] = []
    for p in primary + secondary:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


class RepoDiscoveryConfig(BaseModel):
    """How to find repositories on disk."""

    source: str = Field(default="local", description="local | github")
    include_paths: list[str] = Field(
        default_factory=list,
        description="Glob patterns of local repo paths, e.g. '~/code/*'.",
    )
    exclude_paths: list[str] = Field(
        default_factory=list,
        description="Glob patterns to ignore.",
    )
    include_local_paths: list[str] = Field(
        default_factory=list,
        description="Alias for include_paths (used by GitHub/ZenHub configs).",
    )
    exclude_local_paths: list[str] = Field(
        default_factory=list,
        description="Alias for exclude_paths (used by GitHub/ZenHub configs).",
    )
    include_github_repos: list[str] = Field(
        default_factory=list,
        description="GitHub repositories to ingest, e.g. owner/repo.",
    )
    exclude_github_repos: list[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude GitHub repos.",
    )
    mirror_clone: bool = Field(default=True, description="Use --mirror when cloning GitHub repos.")
    git_fetch_prune: bool = Field(default=True, description="Use --prune when fetching GitHub repos.")
    max_commits_per_repo: int = Field(
        default=20000,
        description="Safety cap. Set higher if you need deeper history.",
    )

    def local_include_paths(self) -> list[str]:
        return _merge_paths(self.include_paths, self.include_local_paths)

    def local_exclude_paths(self) -> list[str]:
        return _merge_paths(self.exclude_paths, self.exclude_local_paths)


class FiltersConfig(BaseModel):
    """Filters applied to mined commits."""

    authors: PatternMatcher = Field(default_factory=PatternMatcher)
    repos: PatternMatcher = Field(default_factory=PatternMatcher)
    languages: PatternMatcher = Field(default_factory=lambda: PatternMatcher(include=["python"]))


class OutputConfig(BaseModel):
    base_dir: str = Field(default="~/standard_coder_outputs")
    repos_dir: str = Field(default="repos")
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
            repos_dir=base / self.repos_dir,
            raw_dir=base / self.raw_dir,
            features_dir=base / self.features_dir,
            labels_dir=base / self.labels_dir,
            models_dir=base / self.models_dir,
            forecasts_dir=base / self.forecasts_dir,
            reports_dir=base / self.reports_dir,
            logs_dir=base / self.logs_dir,
            checkpoints_dir=base / self.checkpoints_dir,
        )


class GithubConfig(BaseModel):
    enabled: bool = Field(default=False)
    token_env_var: str = Field(default="GITHUB_TOKEN")
    api_base_url: str = Field(default="https://api.github.com")
    pulls_since: str | None = Field(default=None, description="ISO8601 cutoff for updated PRs.")
    ingest_reviews: bool = Field(default=True)
    ingest_check_runs: bool = Field(default=False)
    clone_url_template: str = Field(default="https://github.com/{owner}/{repo}.git")


class ZenhubConfig(BaseModel):
    enabled: bool = Field(default=False)
    token_env_var: str = Field(default="ZENHUB_TOKEN")
    api_base_url: str = Field(default="https://api.zenhub.com")
    mode: str = Field(default="milestones", description="milestones | iterations")
    default_sprint_length_days: int = Field(default=10)
    working_days: tuple[int, ...] = Field(default=(0, 1, 2, 3, 4))
    milestone_title_prefix: str = Field(default="")


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
    historical_sprints_path: str = Field(
        default="",
        description="Path to historical sprint outcomes (done effort + workdays).",
    )
    story_points_to_sch: float = Field(
        default=2.0,
        description="Fallback mapping from story points to SCH hours.",
    )
    story_points_cv: float = Field(
        default=0.5,
        description="Coefficient of variation for story-point fallback.",
    )


class PipelineConfig(BaseModel):
    repos: RepoDiscoveryConfig = Field(default_factory=RepoDiscoveryConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    github: GithubConfig = Field(default_factory=GithubConfig)
    zenhub: ZenhubConfig = Field(default_factory=ZenhubConfig)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        raw = _read_toml(path)
        return cls.model_validate(raw)


class ResolvedOutputPaths(BaseModel):
    base_dir: Path
    repos_dir: Path
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
            self.repos_dir,
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

    @property
    def github_db_path(self) -> Path:
        return self.raw_dir / "github.sqlite"

    @property
    def zenhub_db_path(self) -> Path:
        return self.raw_dir / "zenhub.sqlite"
