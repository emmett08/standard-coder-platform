from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from standard_coder.adapters.git.git_ops import GitRepoSpec, clone_or_fetch_repo
from standard_coder.adapters.github.github_client import GitHubClient
from standard_coder.adapters.github.github_ingest import ingest_pull_requests
from standard_coder.adapters.zenhub.zenhub_client import ZenHubClient
from standard_coder.adapters.zenhub.zenhub_ingest import (
    ingest_current_sprints_from_milestones,
    load_issue_estimates,
)
from standard_coder.common.logging_config import configure_logging
from standard_coder.pipeline.checkpointing import RunCheckpoint
from standard_coder.pipeline.config import PipelineConfig, ResolvedOutputPaths
from standard_coder.pipeline.git_io import discover_repos, iter_mined_commits, load_commit
from standard_coder.pipeline.progress_ui import Ui
from standard_coder.pipeline.raw_store import RawStore
from standard_coder.sch.domain.entities import Commit, PullRequest
from standard_coder.sch.features.ast_features import AstEditChangeRep
from standard_coder.sch.features.graph_features import GraphDeltaChangeRep
from standard_coder.sch.features.hybrid import ConcatenatedChangeRep
from standard_coder.sch.features.token_features import BagOfTokensChangeRep
from standard_coder.sch.pipelines.training import SchTrainingPipeline
from standard_coder.sch.predictors.mdn import MdnEffortPredictor
from standard_coder.sch.predictors.quantile import QuantileEffortPredictor
from standard_coder.sch.predictors.multitask import HeadConfig, TorchMultiTaskEffortPredictor
from standard_coder.pipeline.pr_loader import load_pull_requests, load_pull_requests_from_github_db

from standard_coder.sch.work_inference.hawkes import HawkesWorkInference
from standard_coder.sch.work_inference.neural_hmm import NeuralHmmWorkInference
from standard_coder.sch.work_inference.state_space import VariationalStateSpaceWorkInference

logger = logging.getLogger(__name__)

def _stable_fingerprint(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _fit_hmm_author_payload(
    author_id: str,
    times: list[int],
    *,
    step_minutes: int,
    epochs: int,
    lr: float,
    hidden_size: int,
    min_commits: int,
    device: str,
) -> dict[str, object] | None:
    """Fit a single HMM author in an isolated process.

    Kept module-level so it can be pickled by ProcessPoolExecutor.
    """
    wi = NeuralHmmWorkInference(
        step_minutes=step_minutes,
        epochs=epochs,
        lr=lr,
        hidden_size=hidden_size,
        min_commits=min_commits,
        device=device,
    )
    wi.fit_single_author(author_id, times)
    if hasattr(wi, "fitted_author_ids") and author_id not in wi.fitted_author_ids():  # type: ignore[attr-defined]
        return None
    return wi.export_author_state(author_id)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _prefer_mps(device: str | None) -> str | None:
    if device is not None:
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _state_dict_compatible(
    loaded: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> bool:
    if loaded.keys() != expected.keys():
        return False
    for key, value in loaded.items():
        if value.shape != expected[key].shape:
            return False
    return True


def _commit_time_minutes(c: Commit) -> int:
    return int(c.authored_at.timestamp() // 60)


@dataclass
class PipelineRunner:
    config: PipelineConfig
    outputs: ResolvedOutputPaths
    ui: Ui
    checkpoint: RunCheckpoint

    @classmethod
    def from_config_path(cls, config_path: Path, ui: Ui) -> "PipelineRunner":
        cfg = PipelineConfig.load(config_path)
        outs = cfg.outputs.resolve()
        outs.ensure_dirs()

        configure_logging(logging.INFO, log_dir=str(outs.logs_dir))
        ck = RunCheckpoint.load(outs.checkpoints_dir / "run_state.json")
        return cls(config=cfg, outputs=outs, ui=ui, checkpoint=ck)

    def run(self, stages: Sequence[str] | None = None) -> None:
        stages = stages or ("mine", "train_sch", "forecast")
        for stage in stages:
            if stage == "mine":
                self._stage_mine()
            elif stage == "train_sch":
                self._stage_train_sch()
            elif stage == "forecast":
                self._stage_forecast()
            else:
                raise ValueError(f"Unknown stage: {stage}")

    def _resolve_repo_specs(self) -> list[GitRepoSpec]:
        repos_cfg = self.config.repos
        source = str(repos_cfg.source or "local").lower().strip()

        if source == "github":
            include = list(repos_cfg.include_github_repos or [])
            if not include:
                raise ValueError("repos.include_github_repos is required when repos.source='github'")

            exclude = list(repos_cfg.exclude_github_repos or [])
            clone_tmpl = (self.config.github.clone_url_template or "").strip()
            if not clone_tmpl:
                clone_tmpl = "https://github.com/{owner}/{repo}.git"

            mirror = bool(repos_cfg.mirror_clone)
            specs: list[GitRepoSpec] = []
            for repo_full in include:
                if any(fnmatch(repo_full, pat) for pat in exclude):
                    continue
                if "/" not in repo_full:
                    logger.warning("Skipping invalid GitHub repo: %s", repo_full)
                    continue
                owner, repo = repo_full.split("/", 1)
                url = clone_tmpl.format(owner=owner, repo=repo)
                safe = repo_full.replace("/", "__")
                local = self.outputs.repos_dir / (f"{safe}.git" if mirror else safe)
                specs.append(GitRepoSpec(name=repo_full, url=url, local_path=local))
            return specs

        include = repos_cfg.local_include_paths()
        exclude = repos_cfg.local_exclude_paths()
        repos = discover_repos(include, exclude)
        return [GitRepoSpec(name=repo.name, url=str(repo), local_path=repo) for repo in repos]

    def _sync_repos(self, repo_specs: list[GitRepoSpec]) -> None:
        repos_cfg = self.config.repos
        mirror = bool(repos_cfg.mirror_clone)
        prune = bool(repos_cfg.git_fetch_prune)
        for spec in repo_specs:
            if Path(spec.url).exists():
                continue
            clone_or_fetch_repo(spec, mirror=mirror, prune=prune)

    def _ingest_github(self, repo_specs: list[GitRepoSpec]) -> None:
        github_cfg = self.config.github
        if not github_cfg.enabled:
            return

        token = os.environ.get(github_cfg.token_env_var, "")
        if not token:
            raise RuntimeError(f"Missing GitHub token in env var: {github_cfg.token_env_var}")

        client = GitHubClient(token=token, api_base_url=github_cfg.api_base_url)
        repo_names = [spec.name for spec in repo_specs if "/" in spec.name]
        if not repo_names:
            repo_names = list(self.config.repos.include_github_repos or [])
        if not repo_names:
            raise RuntimeError("No GitHub repos configured for ingestion.")

        ingest_pull_requests(
            github=client,
            repos=repo_names,
            db_path=self.outputs.github_db_path,
            pulls_since_iso=github_cfg.pulls_since,
            ingest_reviews=github_cfg.ingest_reviews,
            ingest_check_runs=github_cfg.ingest_check_runs,
        )

    def _ingest_zenhub(self, repo_specs: list[GitRepoSpec]):
        zenhub_cfg = self.config.zenhub
        if not zenhub_cfg.enabled:
            return []

        gh_token = os.environ.get(self.config.github.token_env_var, "")
        if not gh_token:
            raise RuntimeError(f"Missing GitHub token in env var: {self.config.github.token_env_var}")

        zh_token = os.environ.get(zenhub_cfg.token_env_var, "")
        if not zh_token:
            raise RuntimeError(f"Missing ZenHub token in env var: {zenhub_cfg.token_env_var}")

        gh = GitHubClient(token=gh_token, api_base_url=self.config.github.api_base_url)
        zh = ZenHubClient(token=zh_token, api_base_url=zenhub_cfg.api_base_url)

        repos = [spec.name for spec in repo_specs if "/" in spec.name]
        if not repos:
            repos = list(self.config.repos.include_github_repos or [])

        repo_ids: dict[str, int] = {}
        for repo_full in repos:
            owner, name = repo_full.split("/", 1)
            repo_ids[repo_full] = gh.get_repo_id(owner, name)

        mode = str(zenhub_cfg.mode).lower()
        if mode != "milestones":
            logger.warning("ZenHub mode %s not implemented; falling back to milestones", mode)

        return ingest_current_sprints_from_milestones(
            github=gh,
            zenhub=zh,
            repos=repos,
            github_repo_ids=repo_ids,
            db_path=self.outputs.zenhub_db_path,
            milestone_title_prefix=zenhub_cfg.milestone_title_prefix,
            default_sprint_length_days=zenhub_cfg.default_sprint_length_days,
        )

    def _stage_mine(self) -> None:
        stage = "mine"
        fingerprint = _stable_fingerprint(
            {
                "repos": self.config.repos.model_dump(),
                "filters": self.config.filters.model_dump(),
            }
        )
        if self.checkpoint.is_stage_done(stage, fingerprint=fingerprint) and self.config.checkpointing.resume:
            self.ui.log("[green]Skipping mine (already done).[/green]")
            return

        repo_specs = self._resolve_repo_specs()
        if str(self.config.repos.source or "local").lower().strip() == "github":
            self._sync_repos(repo_specs)

        self.ui.log(f"Discovered {len(repo_specs)} repositories")

        store = RawStore(self.outputs.raw_dir / "mined_commits.sqlite")
        conn = store.connect()

        task = self.ui.progress.add_task("Mining commits", total=len(repo_specs))
        for spec in repo_specs:
            repo_label = spec.name if "/" in spec.name else str(spec.local_path)
            # repo-level filter
            if not self.config.filters.repos.matches(repo_label):
                self.ui.progress.advance(task)
                continue

            try:
                for mc in iter_mined_commits(
                    spec.local_path,
                    max_commits=self.config.repos.max_commits_per_repo,
                ):
                    if not self.config.filters.languages.matches(mc.language):
                        continue
                    if not self.config.filters.authors.matches(mc.author_id):
                        continue
                    store.upsert_mined_commit(conn, mc)
                conn.commit()
            except Exception as exc:
                logger.exception("Failed mining repo %s: %s", repo_label, exc)

            self.ui.progress.advance(task)

        conn.close()
        self.checkpoint.mark_stage_done(stage, meta={"repos": len(repo_specs)}, fingerprint=fingerprint)
        self.ui.log("[green]Mining complete.[/green]")

    def _stage_train_sch(self) -> None:
        stage = "train_sch"
        fingerprint = _stable_fingerprint(
            {
                "filters": self.config.filters.model_dump(),
                "training": self.config.training.model_dump(),
            }
        )
        if self.checkpoint.is_stage_done(stage, fingerprint=fingerprint) and self.config.checkpointing.resume:
            self.ui.log("[green]Skipping train_sch (already done).[/green]")
            return

        repo_specs = self._resolve_repo_specs()
        if self.config.github.enabled:
            self._ingest_github(repo_specs)

        # Load mined commit metadata
        store = RawStore(self.outputs.raw_dir / "mined_commits.sqlite")
        conn = store.connect()
        mined = list(store.iter_mined(conn))
        conn.close()

        if not mined:
            raise RuntimeError("No commits found. Run stage 'mine' first or update filters.")

        # Materialise full commit objects (diff + optional blobs) with caching.
        include_blobs = bool(self.config.training.change_rep.use_upgrade_a)
        load_task = self.ui.progress.add_task("Loading diffs/blobs", total=len(mined))
        commits: list[Commit] = []
        for mc in mined:
            try:
                c = load_commit(mc, cache_dir=self.outputs.raw_dir, include_blobs=include_blobs)
                commits.append(c)
            except Exception as exc:
                logger.debug("Skipping commit %s due to load error: %s", mc.commit_id, exc)
            self.ui.progress.advance(load_task)

        commits = [c for c in commits if self.config.filters.languages.matches(c.language)]
        self.ui.log(f"Loaded {len(commits)} commits for training")

        # Choose change representation.
        token_rep = BagOfTokensChangeRep(
            max_features=self.config.training.change_rep.tokens_max_features,
            min_df=self.config.training.change_rep.tokens_min_df,
        )
        if self.config.training.change_rep.use_upgrade_a:
            change_rep = ConcatenatedChangeRep(parts=(token_rep, AstEditChangeRep(), GraphDeltaChangeRep()))
        else:
            change_rep = token_rep

        # Choose work inference model.
        hmm_cfg = self.config.training.hmm
        kind = hmm_cfg.kind.lower().strip()

        if kind == "hmm":
            wi = NeuralHmmWorkInference(
                step_minutes=hmm_cfg.step_minutes,
                epochs=hmm_cfg.epochs,
                lr=hmm_cfg.lr,
                hidden_size=hmm_cfg.hidden_size,
                min_commits=hmm_cfg.min_commits_per_author,
                device=_prefer_mps(hmm_cfg.device),
            )
        elif kind == "hawkes":
            wi = HawkesWorkInference(step_minutes=hmm_cfg.step_minutes, min_commits=hmm_cfg.min_commits_per_author)
        elif kind == "state_space":
            wi = VariationalStateSpaceWorkInference(
                step_minutes=hmm_cfg.step_minutes,
                min_commits=hmm_cfg.min_commits_per_author,
            )
        else:
            raise ValueError("training.hmm.kind must be: hmm | hawkes | state_space")

        # Fit work inference with incremental checkpoints (author-level).
        times_by_author: dict[str, list[int]] = {}
        for c in commits:
            times_by_author.setdefault(c.author_id, []).append(_commit_time_minutes(c))

        wi_dir = self.outputs.models_dir / "work_inference" / kind
        wi_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = wi_dir / "author_map.json"
        author_map: dict[str, str] = {}
        if mapping_path.exists() and self.config.checkpointing.resume:
            author_map = json.loads(mapping_path.read_text())

        authors = sorted(times_by_author.keys())
        wi_task = self.ui.progress.add_task(f"Fitting work inference ({kind})", total=len(authors))

        # Ensure stable keys in author_map (even if we skip/parallelize).
        for author in authors:
            author_map.setdefault(author, _sha1(author))

        # Import cached authors first so downstream dataset building can proceed.
        if self.config.checkpointing.resume and hasattr(wi, "import_author_state"):
            for author in authors:
                out_file = wi_dir / f"{author_map[author]}.json"
                if not out_file.exists():
                    continue
                try:
                    payload = json.loads(out_file.read_text())
                    wi.import_author_state(author, payload)  # type: ignore[attr-defined]
                except Exception:
                    logger.debug("Failed to import cached state for author %s", author)

        # Optional: parallelize author fitting (CPU only).
        n_workers = max(1, int(getattr(hmm_cfg, "parallel_authors", 1)))
        device_str = _prefer_mps(hmm_cfg.device) or "cpu"
        if kind == "hmm" and n_workers > 1 and device_str != "cpu":
            self.ui.log(
                f"[yellow]training.hmm.parallel_authors={n_workers} requested but disabled on device={device_str}; "
                "set training.hmm.device='cpu' to enable CPU parallel fitting.[/yellow]"
            )
            n_workers = 1

        if kind == "hmm" and n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            to_fit: list[str] = []
            for author in authors:
                out_file = wi_dir / f"{author_map[author]}.json"
                if out_file.exists() and self.config.checkpointing.resume:
                    self.ui.progress.advance(wi_task)
                    continue
                to_fit.append(author)

            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {
                    ex.submit(
                        _fit_hmm_author_payload,
                        author,
                        list(times_by_author[author]),
                        step_minutes=hmm_cfg.step_minutes,
                        epochs=hmm_cfg.epochs,
                        lr=hmm_cfg.lr,
                        hidden_size=hmm_cfg.hidden_size,
                        min_commits=hmm_cfg.min_commits_per_author,
                        device=device_str,
                    ): author
                    for author in to_fit
                }

                for fut in as_completed(futs):
                    author = futs[fut]
                    try:
                        payload = fut.result()
                    except Exception as exc:
                        logger.debug("Work inference failed for author %s: %s", author, exc)
                        self.ui.progress.advance(wi_task)
                        continue

                    if payload is not None:
                        out_file = wi_dir / f"{author_map[author]}.json"
                        out_file.write_text(json.dumps(payload))
                        if hasattr(wi, "import_author_state"):
                            wi.import_author_state(author, payload)  # type: ignore[attr-defined]

                    self.ui.progress.advance(wi_task)
        else:
            for author in authors:
                out_file = wi_dir / f"{author_map[author]}.json"
                if out_file.exists() and self.config.checkpointing.resume:
                    self.ui.progress.advance(wi_task)
                    continue

                # Fit one author at a time for checkpointing.
                try:
                    if hasattr(wi, "fit_single_author"):
                        wi.fit_single_author(author, times_by_author[author])  # type: ignore[attr-defined]
                    else:
                        wi.fit({author: times_by_author[author]})
                except Exception as exc:
                    logger.debug("Work inference failed for author %s: %s", author, exc)
                    self.ui.progress.advance(wi_task)
                    continue

                should_export = True
                if hasattr(wi, "fitted_author_ids"):
                    should_export = author in wi.fitted_author_ids()  # type: ignore[attr-defined]
                if hasattr(wi, "export_author_state") and should_export:
                    payload = wi.export_author_state(author)  # type: ignore[attr-defined]
                    out_file.write_text(json.dumps(payload))
                self.ui.progress.advance(wi_task)

        mapping_path.write_text(json.dumps(author_map, indent=2, sort_keys=True))
        self.ui.log(f"Work inference fitted for {len(author_map)} authors (resume ok).")

        # Train predictors.
        pipeline = SchTrainingPipeline(work_inference=wi, change_rep=change_rep)

        pred_kind = self.config.training.predictor_kind.lower().strip()

        # ----------------------- Build dataset (cached) -----------------------
        ds_dir = self.outputs.labels_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)

        cr_cached = self._load_change_rep()

        if pred_kind == "multitask":
            if not self.config.training.multitask.enable:
                raise RuntimeError("predictor_kind=multitask but [training.multitask].enable=false")

            pr_path_raw = self.config.training.multitask.pull_requests_path.strip()
            pull_requests: list[PullRequest]
            if pr_path_raw:
                pr_path = Path(pr_path_raw).expanduser()
                if not pr_path.exists():
                    raise RuntimeError(
                        "Multi-task training requires pull_requests_path to a JSON file "
                        "(see examples/prs.example.json)."
                    )
                pull_requests = load_pull_requests(pr_path)
                self.ui.log(f"Loaded {len(pull_requests)} PRs from {pr_path}")
            elif self.config.github.enabled and self.outputs.github_db_path.exists():
                pull_requests = load_pull_requests_from_github_db(self.outputs.github_db_path)
                if not pull_requests:
                    raise RuntimeError(
                        "GitHub ingestion DB contains no PRs; check [github] settings or ingest again."
                    )
                self.ui.log(f"Loaded {len(pull_requests)} PRs from {self.outputs.github_db_path}")
            else:
                raise RuntimeError(
                    "Multi-task training requires pull_requests_path or enabled GitHub ingestion."
                )
            ds_path = ds_dir / "multitask_train.npz"

            if ds_path.exists() and cr_cached is not None and self.config.checkpointing.resume:
                ds = dict(np.load(ds_path))
                x = ds["x"]
                y_coding = ds["y_coding"]
                y_delivery = ds["y_delivery"]
                change_rep = cr_cached  # type: ignore[assignment]
                self.ui.log("Loaded cached multi-task dataset and change representation.")
            else:
                from standard_coder.sch.pipelines.dataset_builder import SchDatasetBuilder

                progress_tasks: dict[str, int] = {}

                def on_progress(phase: str, completed: int, total: int) -> None:
                    label = {
                        "change_rep_fit": "Dataset: fitting change representation",
                        "change_rep_transform": "Dataset: transforming commits",
                        "label_pairs": "Dataset: sampling labels (commit pairs)",
                    }.get(phase, f"Dataset: {phase}")

                    task_id = progress_tasks.get(phase)
                    if task_id is None:
                        task_id = self.ui.progress.add_task(label, total=total)
                        progress_tasks[phase] = task_id
                    self.ui.progress.update(task_id, total=total, completed=completed)

                builder = SchDatasetBuilder(
                    work_inference=wi,
                    change_rep=change_rep,
                    label_samples_per_commit=self.config.training.label_samples_per_commit,
                    rng_seed=self.config.training.rng_seed,
                )
                ds_built = builder.build_multitask_dataset(
                    commits,
                    pull_requests,
                    fit_work_inference=False,
                    on_progress=on_progress,
                )
                assert ds_built.y_delivery_hours is not None
                x = ds_built.x
                y_coding = ds_built.y_coding_hours
                y_delivery = ds_built.y_delivery_hours
                np.savez_compressed(ds_path, x=x, y_coding=y_coding, y_delivery=y_delivery)
                cr_path = self._save_change_rep(change_rep)
                self.ui.log(f"Saved dataset to {ds_path}")
                self.ui.log(f"Saved change representation to {cr_path}")

        else:
            ds_path = ds_dir / "coding_train.npz"
            if ds_path.exists() and cr_cached is not None and self.config.checkpointing.resume:
                ds = dict(np.load(ds_path))
                x = ds["x"]
                y = ds["y"]
                change_rep = cr_cached  # type: ignore[assignment]
                self.ui.log("Loaded cached coding dataset and change representation.")
            else:
                from standard_coder.sch.pipelines.dataset_builder import SchDatasetBuilder

                progress_tasks: dict[str, int] = {}

                def on_progress(phase: str, completed: int, total: int) -> None:
                    label = {
                        "change_rep_fit": "Dataset: fitting change representation",
                        "change_rep_transform": "Dataset: transforming commits",
                        "label_pairs": "Dataset: sampling labels (commit pairs)",
                    }.get(phase, f"Dataset: {phase}")

                    task_id = progress_tasks.get(phase)
                    if task_id is None:
                        task_id = self.ui.progress.add_task(label, total=total)
                        progress_tasks[phase] = task_id
                    self.ui.progress.update(task_id, total=total, completed=completed)

                builder = SchDatasetBuilder(
                    work_inference=wi,
                    change_rep=change_rep,
                    label_samples_per_commit=self.config.training.label_samples_per_commit,
                    rng_seed=self.config.training.rng_seed,
                )
                ds_built = builder.build_coding_dataset(commits, fit_work_inference=False, on_progress=on_progress)
                x = ds_built.x
                y = ds_built.y_coding_hours
                np.savez_compressed(ds_path, x=x, y=y)
                cr_path = self._save_change_rep(change_rep)
                self.ui.log(f"Saved dataset to {ds_path}")
                self.ui.log(f"Saved change representation to {cr_path}")

        # ----------------------- Train predictor ------------------------------
        if pred_kind == "mdn":
            pred = MdnEffortPredictor(
                n_components=self.config.training.mdn.n_components,
                hidden_sizes=self.config.training.mdn.hidden_sizes,
                epochs=self.config.training.mdn.epochs,
                lr=self.config.training.mdn.lr,
                batch_size=self.config.training.mdn.batch_size,
                truncate_low=self.config.training.mdn.truncate_low,
                truncate_high=self.config.training.mdn.truncate_high,
                device=_prefer_mps(self.config.training.mdn.device),
            )
            self._train_mdn_with_checkpoints(pred, x, y)  # type: ignore[arg-type]
            self._save_mdn(pred)
        elif pred_kind == "quantile":
            pred = QuantileEffortPredictor(
                quantiles=self.config.training.quantile.quantiles,
                hidden_sizes=self.config.training.quantile.hidden_sizes,
                epochs=self.config.training.quantile.epochs,
                lr=self.config.training.quantile.lr,
                batch_size=self.config.training.quantile.batch_size,
                device=_prefer_mps(self.config.training.quantile.device),
            )
            self._train_quantile_with_checkpoints(pred, x, y)  # type: ignore[arg-type]
            self._save_quantile(pred)
        elif pred_kind == "multitask":
            mt_cfg = self.config.training.multitask
            mt = TorchMultiTaskEffortPredictor(
                coding_head=HeadConfig(kind="mdn", n_components=self.config.training.mdn.n_components),
                delivery_head=HeadConfig(kind="quantile", quantiles=self.config.training.quantile.quantiles),
                epochs=mt_cfg.epochs,
                lr=mt_cfg.lr,
                batch_size=mt_cfg.batch_size,
                device=_prefer_mps(mt_cfg.device),
            )
            self._train_multitask_with_checkpoints(mt, x, y_coding, y_delivery)
            self._save_multitask(mt)
        else:
            raise ValueError("training.predictor_kind must be: mdn | quantile | multitask")

        self.checkpoint.mark_stage_done(
            stage,
            meta={"commits": len(commits), "authors": len(authors)},
            fingerprint=fingerprint,
        )
        self.ui.log("[green]SCH training complete.[/green]")

    def _train_mdn_with_checkpoints(self, pred: MdnEffortPredictor, x: np.ndarray, y: np.ndarray) -> None:
        ck_dir = self.outputs.checkpoints_dir / "mdn"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_file = ck_dir / "checkpoint.pt"

        # If the predictor has its own fit, use it when not resuming.
        # For resume capability, implement epoch-level checkpoints here.
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from standard_coder.sch.predictors.mdn import _MdnNet, _mdn_nll  # type: ignore

        device = torch.device(_prefer_mps(pred.device))
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        ds = TensorDataset(x_t, y_t)
        num_workers = int(getattr(self.config.training.mdn, "dataloader_num_workers", 0) or 0)
        dl = DataLoader(
            ds,
            batch_size=pred.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
        )

        net = _MdnNet(in_dim=x.shape[1], n_components=pred.n_components, hidden_sizes=pred.hidden_sizes).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=pred.lr)

        start_epoch = 0
        if ck_file.exists() and self.config.checkpointing.resume:
            payload = torch.load(ck_file, map_location="cpu")
            net_state = payload.get("net")
            if isinstance(net_state, dict) and _state_dict_compatible(net_state, net.state_dict()):
                net.load_state_dict(net_state)
                opt_state = payload.get("opt")
                if isinstance(opt_state, dict):
                    try:
                        opt.load_state_dict(opt_state)
                    except RuntimeError:
                        self.ui.log("MDN optimizer checkpoint incompatible; using fresh optimizer.")
                ck_epoch = payload.get("epoch")
                if isinstance(ck_epoch, (int, float)):
                    start_epoch = int(ck_epoch) + 1
                if start_epoch:
                    self.ui.log(f"Resuming MDN from epoch {start_epoch}.")
                else:
                    self.ui.log("Resuming MDN from checkpoint.")
            else:
                self.ui.log("MDN checkpoint incompatible with current model; training from scratch.")
        else:
            self.ui.log("Training MDN from scratch.")

        # MPS optimisation hints.
        if device.type in ("mps", "cuda"):
            torch.set_float32_matmul_precision("high")

        epoch_task = self.ui.progress.add_task("Training MDN", total=pred.epochs)
        # Advance progress for already-completed epochs.
        if start_epoch:
            self.ui.progress.update(epoch_task, completed=start_epoch)

        net.train()
        for epoch in range(start_epoch, pred.epochs):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pi_logits, mu, sigma = net(xb)
                loss = _mdn_nll(pi_logits, mu, sigma, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                opt.step()
                total += float(loss.item()) * xb.shape[0]

            if (epoch + 1) % 5 == 0:
                self.ui.log(f"MDN epoch {epoch + 1}/{pred.epochs} loss={total / max(1, len(ds)):.4f}")

            if (epoch + 1) % self.config.checkpointing.save_every_epochs == 0:
                torch.save({"epoch": epoch, "net": net.state_dict(), "opt": opt.state_dict()}, ck_file)

            self.ui.progress.advance(epoch_task)

        pred._net = net  # type: ignore[attr-defined]

    def _save_mdn(self, pred: MdnEffortPredictor) -> None:
        out_dir = self.outputs.models_dir / "mdn"
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        import torch

        model_path = out_dir / "model.pt"
        meta_path = out_dir / "metadata.json"

        if pred._net is None:
            raise RuntimeError("MDN net not trained")

        torch.save({"net": pred._net.state_dict(), "config": pred.__dict__}, model_path)
        meta_path.write_text(json.dumps({"model_version": pred.model_version, "unit": pred.unit}, indent=2))

        self.ui.log(f"Saved MDN model to {model_path}")

    def _train_quantile_with_checkpoints(self, pred: QuantileEffortPredictor, x: np.ndarray, y: np.ndarray) -> None:
        ck_dir = self.outputs.checkpoints_dir / "quantile"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_file = ck_dir / "checkpoint.pt"

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from standard_coder.sch.predictors.quantile import _QuantileNet, _pinball_loss  # type: ignore

        device = torch.device(_prefer_mps(pred.device))
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        ds = TensorDataset(x_t, y_t)
        num_workers = int(getattr(self.config.training.quantile, "dataloader_num_workers", 0) or 0)
        dl = DataLoader(
            ds,
            batch_size=pred.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
        )

        net = _QuantileNet(in_dim=x.shape[1], quantiles=pred.quantiles, hidden_sizes=pred.hidden_sizes).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=pred.lr)

        start_epoch = 0
        if ck_file.exists() and self.config.checkpointing.resume:
            payload = torch.load(ck_file, map_location="cpu")
            net_state = payload.get("net")
            if isinstance(net_state, dict) and _state_dict_compatible(net_state, net.state_dict()):
                net.load_state_dict(net_state)
                opt_state = payload.get("opt")
                if isinstance(opt_state, dict):
                    try:
                        opt.load_state_dict(opt_state)
                    except RuntimeError:
                        self.ui.log("Quantile optimizer checkpoint incompatible; using fresh optimizer.")
                ck_epoch = payload.get("epoch")
                if isinstance(ck_epoch, (int, float)):
                    start_epoch = int(ck_epoch) + 1
                if start_epoch:
                    self.ui.log(f"Resuming quantile model from epoch {start_epoch}.")
                else:
                    self.ui.log("Resuming quantile model from checkpoint.")
            else:
                self.ui.log("Quantile checkpoint incompatible with current model; training from scratch.")
        else:
            self.ui.log("Training quantile model from scratch.")

        if device.type in ("mps", "cuda"):
            torch.set_float32_matmul_precision("high")

        epoch_task = self.ui.progress.add_task("Training quantile model", total=pred.epochs)
        if start_epoch:
            self.ui.progress.update(epoch_task, completed=start_epoch)

        net.train()
        for epoch in range(start_epoch, pred.epochs):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                q_pred = net(xb)
                loss = _pinball_loss(q_pred, yb, pred.quantiles)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                opt.step()
                total += float(loss.item()) * xb.shape[0]

            if (epoch + 1) % 5 == 0:
                self.ui.log(f"Quantile epoch {epoch + 1}/{pred.epochs} loss={total / max(1, len(ds)):.4f}")

            if (epoch + 1) % self.config.checkpointing.save_every_epochs == 0:
                torch.save({"epoch": epoch, "net": net.state_dict(), "opt": opt.state_dict()}, ck_file)

            self.ui.progress.advance(epoch_task)

        pred._net = net  # type: ignore[attr-defined]

    def _save_quantile(self, pred: QuantileEffortPredictor) -> None:
        out_dir = self.outputs.models_dir / "quantile"
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        import torch

        model_path = out_dir / "model.pt"
        meta_path = out_dir / "metadata.json"

        if pred._net is None:
            raise RuntimeError("Quantile net not trained")

        torch.save({"net": pred._net.state_dict(), "config": pred.__dict__}, model_path)
        meta_path.write_text(json.dumps({"model_version": pred.model_version, "unit": pred.unit}, indent=2))

        self.ui.log(f"Saved quantile model to {model_path}")



    def _train_multitask_with_checkpoints(
        self,
        pred: TorchMultiTaskEffortPredictor,
        x: np.ndarray,
        y_coding: np.ndarray,
        y_delivery: np.ndarray,
    ) -> None:
        ck_dir = self.outputs.checkpoints_dir / "multitask"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_file = ck_dir / "checkpoint.pt"

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from standard_coder.sch.predictors.multitask import (
            _MultiTaskNet,
            _mdn_nll,
            _pinball_loss,
        )  # type: ignore

        device = torch.device(_prefer_mps(pred.device))
        x_t = torch.tensor(x, dtype=torch.float32)
        y1 = torch.tensor(y_coding.reshape(-1, 1), dtype=torch.float32)
        y2 = torch.tensor(y_delivery.reshape(-1, 1), dtype=torch.float32)

        ds = TensorDataset(x_t, y1, y2)
        num_workers = int(getattr(self.config.training.multitask, "dataloader_num_workers", 0) or 0)
        dl = DataLoader(
            ds,
            batch_size=pred.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
        )

        net = _MultiTaskNet(
            in_dim=x.shape[1],
            trunk_sizes=pred.trunk_sizes,
            coding=pred.coding_head,
            delivery=pred.delivery_head,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=pred.lr)

        start_epoch = 0
        if ck_file.exists() and self.config.checkpointing.resume:
            payload = torch.load(ck_file, map_location="cpu")
            net_state = payload.get("net")
            if isinstance(net_state, dict) and _state_dict_compatible(net_state, net.state_dict()):
                net.load_state_dict(net_state)
                opt_state = payload.get("opt")
                if isinstance(opt_state, dict):
                    try:
                        opt.load_state_dict(opt_state)
                    except RuntimeError:
                        self.ui.log("Multi-task optimizer checkpoint incompatible; using fresh optimizer.")
                ck_epoch = payload.get("epoch")
                if isinstance(ck_epoch, (int, float)):
                    start_epoch = int(ck_epoch) + 1
                if start_epoch:
                    self.ui.log(f"Resuming multi-task model from epoch {start_epoch}.")
                else:
                    self.ui.log("Resuming multi-task model from checkpoint.")
            else:
                self.ui.log("Multi-task checkpoint incompatible with current model; training from scratch.")
        else:
            self.ui.log("Training multi-task model from scratch.")

        if device.type in ("mps", "cuda"):
            torch.set_float32_matmul_precision("high")

        # Prepare quantile tensors if needed.
        q1 = None
        q2 = None
        if pred.coding_head.kind == "quantile":
            q1 = torch.tensor(pred.coding_head.quantiles, dtype=torch.float32, device=device).view(1, -1)
        if pred.delivery_head.kind == "quantile":
            q2 = torch.tensor(pred.delivery_head.quantiles, dtype=torch.float32, device=device).view(1, -1)

        epoch_task = self.ui.progress.add_task("Training multi-task model", total=pred.epochs)
        if start_epoch:
            self.ui.progress.update(epoch_task, completed=start_epoch)

        net.train()
        for epoch in range(start_epoch, pred.epochs):
            total = 0.0
            for xb, yb1, yb2 in dl:
                xb = xb.to(device)
                yb1 = yb1.to(device)
                yb2 = yb2.to(device)

                opt.zero_grad(set_to_none=True)
                c_out, d_out = net(xb)

                loss = torch.tensor(0.0, device=device)
                if pred.coding_head.kind == "mdn":
                    pi, mu, sig = c_out  # type: ignore[misc]
                    loss = loss + _mdn_nll(pi, mu, sig, yb1)
                else:
                    (pred_q,) = c_out  # type: ignore[misc]
                    assert q1 is not None
                    loss = loss + _pinball_loss(pred_q, yb1, q1)

                if pred.delivery_head.kind == "mdn":
                    pi, mu, sig = d_out  # type: ignore[misc]
                    loss = loss + _mdn_nll(pi, mu, sig, yb2)
                else:
                    (pred_q,) = d_out  # type: ignore[misc]
                    assert q2 is not None
                    loss = loss + _pinball_loss(pred_q, yb2, q2)

                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                opt.step()
                total += float(loss.item()) * xb.shape[0]

            if (epoch + 1) % 5 == 0:
                self.ui.log(f"Multi-task epoch {epoch + 1}/{pred.epochs} loss={total / max(1, len(ds)):.4f}")

            if (epoch + 1) % self.config.checkpointing.save_every_epochs == 0:
                torch.save({"epoch": epoch, "net": net.state_dict(), "opt": opt.state_dict()}, ck_file)

            self.ui.progress.advance(epoch_task)

        pred._net = net  # type: ignore[attr-defined]

    def _save_multitask(self, pred: TorchMultiTaskEffortPredictor) -> None:
        out_dir = self.outputs.models_dir / "multitask"
        out_dir.mkdir(parents=True, exist_ok=True)
        import torch

        model_path = out_dir / "model.pt"
        meta_path = out_dir / "metadata.json"

        if pred._net is None:
            raise RuntimeError("Multi-task net not trained")

        torch.save({"net": pred._net.state_dict(), "config": pred.__dict__}, model_path)
        meta_path.write_text(json.dumps({"model_version": pred.model_version, "unit": pred.unit}, indent=2))
        self.ui.log(f"Saved multi-task model to {model_path}")

    def _save_change_rep(self, change_rep) -> Path:
        """Persist the fitted change representation for downstream scoring."""
        import joblib

        cr_dir = self.outputs.features_dir / "change_rep"
        cr_dir.mkdir(parents=True, exist_ok=True)
        path = cr_dir / "change_rep.joblib"
        joblib.dump(change_rep, path)
        return path

    def _load_change_rep(self) -> object | None:
        """Load a previously fitted change representation if available."""
        cr_path = self.outputs.features_dir / "change_rep" / "change_rep.joblib"
        if not cr_path.exists():
            return None
        import joblib

        return joblib.load(cr_path)

    def _stage_forecast(self) -> None:
        stage = "forecast"
        if not self.config.forecasting.enable:
            self.ui.log("Forecasting disabled.")
            return
        fingerprint = _stable_fingerprint(
            {
                "forecasting": self.config.forecasting.model_dump(),
                "zenhub": self.config.zenhub.model_dump(),
                "github": self.config.github.model_dump(),
            }
        )
        if self.checkpoint.is_stage_done(stage, fingerprint=fingerprint) and self.config.checkpointing.resume:
            self.ui.log("[green]Skipping forecast (already done).[/green]")
            return

        from datetime import date, datetime, timezone
        import math

        import scipy.stats as st

        from standard_coder.forecasting.priors.throughput import (
            EmpiricalThroughputPrior,
            default_throughput_prior,
            throughput_prior_from_history,
        )
        from standard_coder.forecasting.services.forecasting_service import ForecastingService
        from standard_coder.integration.event_bus import InMemoryEventBus
        from standard_coder.integration.events import (
            AvailabilityChanged,
            ScopeChanged,
            SprintConfigured,
            WorkItemAddedToSprint,
            WorkItemStatusChanged,
            WorkItemEffortEstimated,
        )

        def _lognormal_quantiles(mean: float, cv: float) -> dict[float, float]:
            mean = max(1e-6, float(mean))
            cv = max(1e-6, float(cv))
            sigma2 = math.log(cv * cv + 1.0)
            sigma = math.sqrt(sigma2)
            mu = math.log(mean) - sigma2 / 2.0
            return {
                0.5: math.exp(mu + sigma * st.norm.ppf(0.5)),
                0.8: math.exp(mu + sigma * st.norm.ppf(0.8)),
                0.9: math.exp(mu + sigma * st.norm.ppf(0.9)),
            }

        def _story_points_effort(points: float | None) -> dict[str, object]:
            sp = float(points) if points is not None else 1.0
            mean = max(1e-6, sp) * self.config.forecasting.story_points_to_sch
            q = _lognormal_quantiles(mean, self.config.forecasting.story_points_cv)
            return {"type": "quantiles", "quantiles": {str(k): float(v) for k, v in q.items()}}

        def _load_throughput_prior() -> object:
            hist_sprints = self.config.forecasting.historical_sprints_path.strip()
            if hist_sprints:
                p = Path(hist_sprints).expanduser()
                if p.exists():
                    return throughput_prior_from_history(p)

            hist_path = Path(self.config.forecasting.historical_outcomes_path).expanduser()
            if hist_path.exists():
                try:
                    hist = json.loads(hist_path.read_text())
                    samples = tuple(float(x) for x in hist.get("throughput_samples_sch_per_day", []))
                except Exception:
                    samples = tuple()
                if samples:
                    return EmpiricalThroughputPrior(samples=samples)

            return default_throughput_prior()

        prior = _load_throughput_prior()

        sprint_inputs = None
        sprint_path_raw = self.config.forecasting.sprint_inputs_path.strip()
        if sprint_path_raw:
            sprint_path = Path(sprint_path_raw).expanduser()
            if sprint_path.exists():
                sprint_inputs = json.loads(sprint_path.read_text())
            else:
                self.ui.log(f"sprint_inputs_path not found: {sprint_path}")

        if sprint_inputs:
            bus = InMemoryEventBus()
            forecasting = ForecastingService(
                bus=bus,
                throughput_prior=prior,
                n_sims=self.config.forecasting.n_sims,
                rng_seed=self.config.forecasting.rng_seed,
            )

            # Publish sprint configuration and state.
            sc = sprint_inputs["sprint"]
            bus.publish(
                SprintConfigured(
                    occurred_at=datetime.now(timezone.utc),
                    sprint_id=sc["sprint_id"],
                    start_date=date.fromisoformat(sc["start_date"]),
                    length_days=int(sc["length_days"]),
                    working_days=tuple(sc["working_days"]),
                )
            )

            for cap in sprint_inputs.get("capacity", []):
                bus.publish(
                    AvailabilityChanged(
                        occurred_at=datetime.now(timezone.utc),
                        person_id=str(cap.get("person_id", "team")),
                        day=date.fromisoformat(cap["day"]),
                        delta_available_sch=float(cap["delta_available_sch"]),
                    )
                )

            for item in sprint_inputs.get("work_items", []):
                wid = str(item["work_item_id"])
                bus.publish(
                    WorkItemAddedToSprint(
                        occurred_at=datetime.now(timezone.utc),
                        sprint_id=sc["sprint_id"],
                        work_item_id=wid,
                    )
                )
                bus.publish(
                    WorkItemStatusChanged(
                        occurred_at=datetime.now(timezone.utc),
                        work_item_id=wid,
                        old_status="todo",
                        new_status=str(item.get("status", "todo")),
                    )
                )
                if "scope_factor" in item:
                    bus.publish(
                        ScopeChanged(
                            occurred_at=datetime.now(timezone.utc),
                            work_item_id=wid,
                            scope_factor=float(item["scope_factor"]),
                        )
                    )
                # Effort estimates are expected to be in event payload format.
                if "effort" in item:
                    bus.publish(
                        WorkItemEffortEstimated(
                            occurred_at=datetime.now(timezone.utc),
                            work_item_id=wid,
                            effort=item["effort"],
                            model_version=str(item.get("model_version", "external")),
                        )
                    )
                elif "story_points" in item:
                    bus.publish(
                        WorkItemEffortEstimated(
                            occurred_at=datetime.now(timezone.utc),
                            work_item_id=wid,
                            effort=_story_points_effort(item.get("story_points")),
                            model_version="story_points",
                        )
                    )

            results = forecasting.compute_forecast(sc["sprint_id"])
            out = next(iter(results.values()))
            out_path = self.outputs.forecasts_dir / f"{sc['sprint_id']}_forecast.json"
            out_path.write_text(json.dumps(out.__dict__, indent=2, default=str))
            self.checkpoint.mark_stage_done(stage, meta={"sprint_id": sc["sprint_id"]}, fingerprint=fingerprint)
            self.ui.log(f"[green]Forecast written to {out_path}[/green]")
            return

        if not self.config.zenhub.enabled:
            self.ui.log("No sprint_inputs_path provided and ZenHub disabled; skipping forecast.")
            return

        repo_specs = self._resolve_repo_specs()
        sprints = self._ingest_zenhub(repo_specs)
        if not sprints:
            self.ui.log("No sprint_inputs_path provided and no ZenHub sprints; skipping forecast.")
            return

        working_days = tuple(self.config.zenhub.working_days)
        sp_cv = self.config.forecasting.story_points_cv
        sp_to_sch = self.config.forecasting.story_points_to_sch

        for sprint in sprints:
            bus = InMemoryEventBus()
            forecasting = ForecastingService(
                bus=bus,
                throughput_prior=prior,
                n_sims=self.config.forecasting.n_sims,
                rng_seed=self.config.forecasting.rng_seed,
            )

            length_days = (sprint.end_date.toordinal() - sprint.start_date.toordinal()) + 1
            sprint_id = sprint.sprint_id

            bus.publish(
                SprintConfigured(
                    occurred_at=datetime.now(timezone.utc),
                    sprint_id=sprint_id,
                    start_date=sprint.start_date,
                    length_days=int(length_days),
                    working_days=tuple(working_days),
                )
            )

            estimates = load_issue_estimates(
                self.outputs.zenhub_db_path,
                sprint.repo,
                sprint.issue_numbers,
            )

            for issue_number in sprint.issue_numbers:
                work_item_id = f"{sprint.repo}#{issue_number}"
                bus.publish(
                    WorkItemAddedToSprint(
                        occurred_at=datetime.now(timezone.utc),
                        sprint_id=sprint_id,
                        work_item_id=work_item_id,
                    )
                )
                sp = estimates.get(issue_number)
                mean = (float(sp) if sp is not None else 1.0) * sp_to_sch
                effort = _lognormal_quantiles(mean, sp_cv)
                bus.publish(
                    WorkItemEffortEstimated(
                        occurred_at=datetime.now(timezone.utc),
                        work_item_id=work_item_id,
                        effort={"type": "quantiles", "quantiles": {str(k): float(v) for k, v in effort.items()}},
                        model_version="story_points",
                    )
                )

            results = forecasting.compute_forecast(sprint_id)
            out = next(iter(results.values()))
            out_path = self.outputs.forecasts_dir / f"{_safe_name(sprint_id)}_forecast.json"
            out_path.write_text(json.dumps(out.__dict__, indent=2, default=str))
            self.ui.log(f"[green]Forecast written to {out_path}[/green]")

        self.checkpoint.mark_stage_done(stage, meta={"sprints": len(sprints)}, fingerprint=fingerprint)


def _safe_name(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("#", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
