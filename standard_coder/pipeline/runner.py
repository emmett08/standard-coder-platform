from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

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
from standard_coder.pipeline.pr_loader import load_pull_requests

from standard_coder.sch.work_inference.hawkes import HawkesWorkInference
from standard_coder.sch.work_inference.neural_hmm import NeuralHmmWorkInference
from standard_coder.sch.work_inference.state_space import VariationalStateSpaceWorkInference

logger = logging.getLogger(__name__)


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

    def _stage_mine(self) -> None:
        stage = "mine"
        if self.checkpoint.is_stage_done(stage) and self.config.checkpointing.resume:
            self.ui.log("[green]Skipping mine (already done).[/green]")
            return

        repos = discover_repos(self.config.repos.include_paths, self.config.repos.exclude_paths)
        self.ui.log(f"Discovered {len(repos)} repositories")

        store = RawStore(self.outputs.raw_dir / "mined_commits.sqlite")
        conn = store.connect()

        task = self.ui.progress.add_task("Mining commits", total=len(repos))
        for repo in repos:
            # repo-level filter
            if not self.config.filters.repos.matches(str(repo)):
                self.ui.progress.advance(task)
                continue

            try:
                for mc in iter_mined_commits(repo, max_commits=self.config.repos.max_commits_per_repo):
                    if not self.config.filters.languages.matches(mc.language):
                        continue
                    if not self.config.filters.authors.matches(mc.author_id):
                        continue
                    store.upsert_mined_commit(conn, mc)
                conn.commit()
            except Exception as exc:
                logger.exception("Failed mining repo %s: %s", repo, exc)

            self.ui.progress.advance(task)

        conn.close()
        self.checkpoint.mark_stage_done(stage, meta={"repos": len(repos)})
        self.ui.log("[green]Mining complete.[/green]")

    def _stage_train_sch(self) -> None:
        stage = "train_sch"
        if self.checkpoint.is_stage_done(stage) and self.config.checkpointing.resume:
            self.ui.log("[green]Skipping train_sch (already done).[/green]")
            return

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

        fitted_before = set(author_map.keys())
        authors = sorted(times_by_author.keys())
        wi_task = self.ui.progress.add_task(f"Fitting work inference ({kind})", total=len(authors))

        for author in authors:
            key = author_map.get(author) or _sha1(author)
            author_map[author] = key
            out_file = wi_dir / f"{key}.json"
            if out_file.exists() and self.config.checkpointing.resume:
                payload = json.loads(out_file.read_text())
                if hasattr(wi, "import_author_state"):
                    wi.import_author_state(author, payload)  # type: ignore[attr-defined]
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

            if hasattr(wi, "export_author_state"):
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

            pr_path = Path(self.config.training.multitask.pull_requests_path).expanduser()
            if not pr_path.exists():
                raise RuntimeError(
                    "Multi-task training requires pull_requests_path to a JSON file "
                    "(see examples/prs.example.json)."
                )

            pull_requests = load_pull_requests(pr_path)
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

                builder = SchDatasetBuilder(
                    work_inference=wi,
                    change_rep=change_rep,
                    label_samples_per_commit=self.config.training.label_samples_per_commit,
                    rng_seed=self.config.training.rng_seed,
                )
                ds_built = builder.build_multitask_dataset(commits, pull_requests)
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

                builder = SchDatasetBuilder(
                    work_inference=wi,
                    change_rep=change_rep,
                    label_samples_per_commit=self.config.training.label_samples_per_commit,
                    rng_seed=self.config.training.rng_seed,
                )
                ds_built = builder.build_coding_dataset(commits)
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

        self.checkpoint.mark_stage_done(stage, meta={"commits": len(commits), "authors": len(authors)})
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
        dl = DataLoader(ds, batch_size=pred.batch_size, shuffle=True)

        net = _MdnNet(in_dim=x.shape[1], n_components=pred.n_components, hidden_sizes=pred.hidden_sizes).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=pred.lr)

        start_epoch = 0
        if ck_file.exists() and self.config.checkpointing.resume:
            payload = torch.load(ck_file, map_location="cpu")
            net.load_state_dict(payload["net"])
            opt.load_state_dict(payload["opt"])
            start_epoch = int(payload["epoch"]) + 1
            self.ui.log(f"Resuming MDN from epoch {start_epoch}.")
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
        dl = DataLoader(ds, batch_size=pred.batch_size, shuffle=True)

        net = _QuantileNet(in_dim=x.shape[1], quantiles=pred.quantiles, hidden_sizes=pred.hidden_sizes).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=pred.lr)

        start_epoch = 0
        if ck_file.exists() and self.config.checkpointing.resume:
            payload = torch.load(ck_file, map_location="cpu")
            net.load_state_dict(payload["net"])
            opt.load_state_dict(payload["opt"])
            start_epoch = int(payload["epoch"]) + 1
            self.ui.log(f"Resuming quantile model from epoch {start_epoch}.")

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
        dl = DataLoader(ds, batch_size=pred.batch_size, shuffle=True)

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
            net.load_state_dict(payload["net"])
            opt.load_state_dict(payload["opt"])
            start_epoch = int(payload["epoch"]) + 1
            self.ui.log(f"Resuming multi-task model from epoch {start_epoch}.")

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
        if self.checkpoint.is_stage_done(stage) and self.config.checkpointing.resume:
            self.ui.log("[green]Skipping forecast (already done).[/green]")
            return

        from datetime import date, datetime

        from standard_coder.forecasting.priors.throughput import EmpiricalThroughputPrior
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

        bus = InMemoryEventBus()

        # Load sprint inputs file if provided; otherwise skip.
        sprint_path = Path(self.config.forecasting.sprint_inputs_path).expanduser()
        if not sprint_path.exists():
            self.ui.log("No sprint_inputs_path provided; skipping forecast.")
            return

        sprint_inputs = json.loads(sprint_path.read_text())

        hist_path = Path(self.config.forecasting.historical_outcomes_path).expanduser()
        if hist_path.exists():
            hist = json.loads(hist_path.read_text())
            samples = tuple(float(x) for x in hist.get("throughput_samples_sch_per_day", []))
        else:
            samples = tuple()

        if not samples:
            # Fallback prior: 1..10 SCH/day
            samples = tuple(float(i) for i in range(1, 11))

        prior = EmpiricalThroughputPrior(samples=samples)
        forecasting = ForecastingService(bus=bus, throughput_prior=prior, n_sims=self.config.forecasting.n_sims, rng_seed=self.config.forecasting.rng_seed)

        # Publish sprint configuration and state.
        sc = sprint_inputs["sprint"]
        bus.publish(
            SprintConfigured(
                occurred_at=datetime.utcnow(),
                sprint_id=sc["sprint_id"],
                start_date=date.fromisoformat(sc["start_date"]),
                length_days=int(sc["length_days"]),
                working_days=tuple(sc["working_days"]),
            )
        )

        for cap in sprint_inputs.get("capacity", []):
            bus.publish(
                AvailabilityChanged(
                    occurred_at=datetime.utcnow(),
                    person_id=str(cap.get("person_id", "team")),
                    day=date.fromisoformat(cap["day"]),
                    delta_available_sch=float(cap["delta_available_sch"]),
                )
            )

        for item in sprint_inputs.get("work_items", []):
            wid = str(item["work_item_id"])
            bus.publish(
                WorkItemAddedToSprint(
                    occurred_at=datetime.utcnow(),
                    sprint_id=sc["sprint_id"],
                    work_item_id=wid,
                )
            )
            bus.publish(
                WorkItemStatusChanged(
                    occurred_at=datetime.utcnow(),
                    work_item_id=wid,
                    old_status="todo",
                    new_status=str(item.get("status", "todo")),
                )
            )
            if "scope_factor" in item:
                bus.publish(
                    ScopeChanged(
                        occurred_at=datetime.utcnow(),
                        work_item_id=wid,
                        scope_factor=float(item["scope_factor"]),
                    )
                )
            # Effort estimates are expected to be in event payload format.
            if "effort" in item:
                bus.publish(
                    WorkItemEffortEstimated(
                        occurred_at=datetime.utcnow(),
                        work_item_id=wid,
                        effort=item["effort"],
                        model_version=str(item.get("model_version", "external")),
                    )
                )

        out = forecasting.get_latest_forecast(sc["sprint_id"])
        out_path = self.outputs.forecasts_dir / f"{sc['sprint_id']}_forecast.json"
        out_path.write_text(json.dumps(out, indent=2, default=str))
        self.checkpoint.mark_stage_done(stage, meta={"sprint_id": sc["sprint_id"]})
        self.ui.log(f"[green]Forecast written to {out_path}[/green]")
