from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import torch
from torch import nn

from standard_coder.sch.interfaces import WorkInferenceModel


logger = logging.getLogger(__name__)


def _select_device(device: str | None) -> torch.device:
    """Prefer Apple Silicon MPS when available, else CUDA, else CPU."""
    if device is not None:
        return torch.device(device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _TransitionNet(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _FittedHmm:
    base_minute: int
    step_minutes: int
    coding_prob: np.ndarray  # (t,)
    model_version: str


def _time_features(
    base_minute: int,
    steps: int,
    step_minutes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build day/week cyclic features for each time step."""
    t = torch.arange(steps, device=device, dtype=torch.float32)
    minutes = base_minute + t * float(step_minutes)
    seconds = minutes * 60.0

    # Convert to datetime-like components using unix time math.
    dt = seconds % (86400.0)
    tod = dt / 1440.0  # days in [0, 60)
    day_angle = 2.0 * math.pi * (dt / 86400.0)
    sin_day = torch.sin(day_angle)
    cos_day = torch.cos(day_angle)

    # Week position: use unix seconds mod week.
    wk = seconds % (86400.0 * 7.0)
    week_angle = 2.0 * math.pi * (wk / (86400.0 * 7.0))
    sin_week = torch.sin(week_angle)
    cos_week = torch.cos(week_angle)

    norm_time = t / max(1.0, float(steps - 1))

    x = torch.stack([sin_day, cos_day, sin_week, cos_week, norm_time], dim=1)
    return x


def _logsumexp(a: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(a, dim=dim)


class NeuralHmmWorkInference(WorkInferenceModel):
    """Baseline work inference: neural time-inhomogeneous HMM.

    This is an engineering-friendly implementation of the model described
    in the paper: 2 hidden states (coding/not-coding) with time-varying
    transition probabilities S(t), E(t) produced by a small neural net.

    The model is fitted **per author**.
    """

    def __init__(
        self,
        step_minutes: int = 5,
        epochs: int = 200,
        lr: float = 1e-2,
        hidden_size: int = 16,
        min_commits: int = 50,
        model_version: str = "neural_hmm_v1",
        device: str | None = None,
    ) -> None:
        self.step_minutes = int(step_minutes)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.hidden_size = int(hidden_size)
        self.min_commits = int(min_commits)
        self.model_version = model_version
        self._device = _select_device(device)
        self._fitted: dict[str, _FittedHmm] = {}

    def fit(self, commit_times_by_author: dict[str, Sequence[int]]) -> None:
        for author_id, times in commit_times_by_author.items():
            times_sorted = sorted(set(int(t) for t in times))
            if len(times_sorted) < self.min_commits:
                logger.info(
                    "Skipping author %s (only %d commits)",
                    author_id,
                    len(times_sorted),
                )
                continue

            base = times_sorted[0]
            last = times_sorted[-1]
            total_minutes = max(1, last - base)
            steps = int(math.ceil(total_minutes / self.step_minutes)) + 1

            y = np.zeros(steps, dtype=np.float32)
            for t in times_sorted:
                idx = int((t - base) / self.step_minutes)
                if 0 <= idx < steps:
                    y[idx] = 1.0

            fitted = self._fit_single_author(base, y)
            self._fitted[author_id] = fitted

    def fitted_author_ids(self) -> set[str]:
        return set(self._fitted.keys())

    def export_author_state(self, author_id: str) -> dict[str, object]:
        """Export a fitted author's state for checkpointing."""
        fitted = self._fitted[author_id]
        return {
            "base_minute": int(fitted.base_minute),
            "step_minutes": int(fitted.step_minutes),
            "coding_prob": fitted.coding_prob.tolist(),
            "model_version": fitted.model_version,
        }

    def import_author_state(self, author_id: str, payload: dict[str, object]) -> None:
        """Import a fitted author's state from a checkpoint."""
        base_minute = int(payload["base_minute"])
        step_minutes = int(payload["step_minutes"])
        coding_prob = np.asarray(payload["coding_prob"], dtype=np.float32)
        model_version = str(payload.get("model_version", self.model_version))
        self._fitted[author_id] = _FittedHmm(
            base_minute=base_minute,
            step_minutes=step_minutes,
            coding_prob=coding_prob,
            model_version=model_version,
        )

    def fit_single_author(self, author_id: str, times: Sequence[int]) -> None:
        """Fit a single author (useful for incremental checkpointing)."""
        times_sorted = sorted(set(int(t) for t in times))
        if len(times_sorted) < self.min_commits:
            logger.info(
                "Skipping author %s (only %d commits)",
                author_id,
                len(times_sorted),
            )
            return

        base = times_sorted[0]
        last = times_sorted[-1]
        total_minutes = max(1, last - base)
        steps = int(math.ceil(total_minutes / self.step_minutes)) + 1

        y = np.zeros(steps, dtype=np.float32)
        for t in times_sorted:
            idx = int((t - base) / self.step_minutes)
            if 0 <= idx < steps:
                y[idx] = 1.0

        fitted = self._fit_single_author(base, y)
        self._fitted[author_id] = fitted

    def _fit_single_author(self, base_minute: int, y: np.ndarray) -> _FittedHmm:
        device = self._device
        steps = int(y.shape[0])

        transition_net = _TransitionNet(hidden_size=self.hidden_size).to(device)
        logit_c = nn.Parameter(torch.tensor(0.0, device=device))
        params = list(transition_net.parameters()) + [logit_c]
        opt = torch.optim.Adam(params, lr=self.lr)

        y_t = torch.tensor(y, device=device, dtype=torch.float32)
        x_t = _time_features(base_minute, steps, self.step_minutes, device=device)

        eps_commit = 1e-6

        for epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)

            logits = transition_net(x_t)
            s = torch.sigmoid(logits[:, 0]).clamp(1e-6, 1.0 - 1e-6)  # start coding
            e = torch.sigmoid(logits[:, 1]).clamp(1e-6, 1.0 - 1e-6)  # end coding
            c = torch.sigmoid(logit_c).clamp(1e-6, 1.0 - 1e-6)

            # log transition probabilities for t -> t+1
            # State 0 = coding, 1 = not-coding
            log_a = torch.zeros((steps - 1, 2, 2), device=device, dtype=torch.float32)
            log_a[:, 0, 0] = torch.log(1.0 - e[:-1])
            log_a[:, 0, 1] = torch.log(e[:-1])
            log_a[:, 1, 0] = torch.log(s[:-1])
            log_a[:, 1, 1] = torch.log(1.0 - s[:-1])

            # log emission probabilities
            log_b = torch.zeros((steps, 2), device=device, dtype=torch.float32)
            # coding state
            log_b[:, 0] = y_t * torch.log(c) + (1.0 - y_t) * torch.log(1.0 - c)
            # not-coding state (allow tiny commit prob for stability)
            log_b[:, 1] = y_t * math.log(eps_commit) + (1.0 - y_t) * math.log(
                1.0 - eps_commit
            )

            # Forward-backward
            log_pi = torch.log(torch.tensor([0.5, 0.5], device=device))

            log_alpha = torch.zeros((steps, 2), device=device, dtype=torch.float32)
            log_alpha[0] = log_pi + log_b[0]

            for t in range(1, steps):
                prev = log_alpha[t - 1].view(2, 1) + log_a[t - 1]
                log_alpha[t] = log_b[t] + _logsumexp(prev, dim=0)

            log_likelihood = _logsumexp(log_alpha[-1], dim=0)
            loss = -log_likelihood

            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "HMM epoch %d/%d loss=%.4f c=%.4f",
                    epoch + 1,
                    self.epochs,
                    float(loss.item()),
                    float(c.item()),
                )

        # Posterior coding probabilities in hindsight mode
        with torch.no_grad():
            logits = transition_net(x_t)
            s = torch.sigmoid(logits[:, 0]).clamp(1e-6, 1.0 - 1e-6)
            e = torch.sigmoid(logits[:, 1]).clamp(1e-6, 1.0 - 1e-6)
            c = torch.sigmoid(logit_c).clamp(1e-6, 1.0 - 1e-6)

            log_a = torch.zeros((steps - 1, 2, 2), device=device, dtype=torch.float32)
            log_a[:, 0, 0] = torch.log(1.0 - e[:-1])
            log_a[:, 0, 1] = torch.log(e[:-1])
            log_a[:, 1, 0] = torch.log(s[:-1])
            log_a[:, 1, 1] = torch.log(1.0 - s[:-1])

            log_b = torch.zeros((steps, 2), device=device, dtype=torch.float32)
            log_b[:, 0] = y_t * torch.log(c) + (1.0 - y_t) * torch.log(1.0 - c)
            log_b[:, 1] = y_t * math.log(eps_commit) + (1.0 - y_t) * math.log(
                1.0 - eps_commit
            )

            log_pi = torch.log(torch.tensor([0.5, 0.5], device=device))

            log_alpha = torch.zeros((steps, 2), device=device, dtype=torch.float32)
            log_alpha[0] = log_pi + log_b[0]
            for t in range(1, steps):
                prev = log_alpha[t - 1].view(2, 1) + log_a[t - 1]
                log_alpha[t] = log_b[t] + _logsumexp(prev, dim=0)

            log_beta = torch.zeros((steps, 2), device=device, dtype=torch.float32)
            log_beta[-1] = 0.0
            for t in range(steps - 2, -1, -1):
                nxt = log_a[t] + log_b[t + 1].view(1, 2) + log_beta[t + 1].view(1, 2)
                log_beta[t] = _logsumexp(nxt, dim=1)

            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - _logsumexp(log_gamma, dim=1).view(steps, 1)
            gamma = torch.exp(log_gamma)

            coding_prob = gamma[:, 0].cpu().numpy()

        return _FittedHmm(
            base_minute=base_minute,
            step_minutes=self.step_minutes,
            coding_prob=coding_prob.astype(np.float32),
            model_version=self.model_version,
        )

    def infer_coding_minutes(
        self,
        author_id: str,
        parent_minute: int,
        child_minute: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if author_id not in self._fitted:
            raise KeyError(f"Author not fitted: {author_id}")

        fitted = self._fitted[author_id]
        a = min(parent_minute, child_minute)
        b = max(parent_minute, child_minute)

        i0 = int((a - fitted.base_minute) / fitted.step_minutes)
        i1 = int((b - fitted.base_minute) / fitted.step_minutes)

        i0 = max(0, min(len(fitted.coding_prob) - 1, i0))
        i1 = max(0, min(len(fitted.coding_prob) - 1, i1))

        probs = fitted.coding_prob[i0:i1]
        steps = probs.shape[0]
        minutes_per_step = fitted.step_minutes

        mean = float(np.sum(probs) * minutes_per_step)
        var = float(np.sum(probs * (1.0 - probs)) * (minutes_per_step ** 2))

        if steps <= 256:
            draws = rng.uniform(0.0, 1.0, size=(n_samples, steps))
            active = (draws < probs[None, :]).astype(np.float32)
            return active.sum(axis=1) * minutes_per_step

        # Normal approximation for large step counts.
        std = math.sqrt(max(1e-9, var))
        samples = rng.normal(loc=mean, scale=std, size=n_samples)
        samples = np.clip(samples, 0.0, float((b - a)))
        return samples.astype(np.float32)
