from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from standard_coder.sch.interfaces import WorkInferenceModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FittedHawkes:
    base_minute: int
    step_minutes: int
    coding_prob: np.ndarray
    mu: float
    alpha: float
    beta: float
    model_version: str


def _neg_log_likelihood(params_raw: np.ndarray, times: np.ndarray, horizon: float) -> float:
    # Transform to positive domain.
    mu = math.exp(float(params_raw[0]))
    alpha = math.exp(float(params_raw[1]))
    beta = math.exp(float(params_raw[2]))

    if times.size == 0:
        return 0.0

    # Compute g_i recursively.
    g = 0.0
    ll = 0.0
    prev_t = float(times[0])
    for i, t in enumerate(times):
        t = float(t)
        if i == 0:
            g = 0.0
        else:
            dt = t - prev_t
            g = math.exp(-beta * dt) * (1.0 + g)
        lam = mu + alpha * g
        ll += math.log(max(1e-12, lam))
        prev_t = t

    # Integral term
    integral = mu * horizon
    # Sum kernel integrals from each event
    tail = np.exp(-beta * (horizon - times))
    integral += (alpha / beta) * float(np.sum(1.0 - tail))

    return -(ll - integral)


class HawkesWorkInference(WorkInferenceModel):
    """Upgrade B: Hawkes-process based work inference.

    This model captures bursty event dynamics. We treat commit events as a
    self-exciting process and map the inferred intensity to an approximate
    coding-activity probability curve.

    This is not identical to the paper's HMM, but provides an alternative
    inductive bias: bursts correspond to coding sessions.
    """

    def __init__(
        self,
        step_minutes: int = 5,
        min_commits: int = 50,
        max_iter: int = 300,
        model_version: str = "hawkes_v1",
    ) -> None:
        self.step_minutes = int(step_minutes)
        self.min_commits = int(min_commits)
        self.max_iter = int(max_iter)
        self.model_version = model_version
        self._fitted: dict[str, _FittedHawkes] = {}

    def fit(self, commit_times_by_author: dict[str, Sequence[int]]) -> None:
        for author_id, times in commit_times_by_author.items():
            t = np.array(sorted(set(int(x) for x in times)), dtype=float)
            if t.size < self.min_commits:
                logger.info(
                    "Skipping author %s (only %d commits)",
                    author_id,
                    int(t.size),
                )
                continue

            base = int(t.min())
            t0 = t - float(base)
            horizon = float(t0.max() + 1.0)

            # Initial guess
            rate = max(1e-6, float(t0.size / horizon))
            x0 = np.log(np.array([rate * 0.5, rate * 0.5, 1.0], dtype=float))

            res = minimize(
                _neg_log_likelihood,
                x0=x0,
                args=(t0, horizon),
                method="L-BFGS-B",
                options={"maxiter": self.max_iter},
            )

            mu = math.exp(float(res.x[0]))
            alpha = math.exp(float(res.x[1]))
            beta = math.exp(float(res.x[2]))

            coding_prob = self._build_coding_prob_curve(
                times=t0,
                horizon=horizon,
                mu=mu,
                alpha=alpha,
                beta=beta,
            )

            self._fitted[author_id] = _FittedHawkes(
                base_minute=base,
                step_minutes=self.step_minutes,
                coding_prob=coding_prob.astype(np.float32),
                mu=mu,
                alpha=alpha,
                beta=beta,
                model_version=self.model_version,
            )

    def fitted_author_ids(self) -> set[str]:
        return set(self._fitted.keys())

    def export_author_state(self, author_id: str) -> dict[str, object]:
        fitted = self._fitted[author_id]
        return {
            "base_minute": int(fitted.base_minute),
            "step_minutes": int(fitted.step_minutes),
            "coding_prob": fitted.coding_prob.tolist(),
            "mu": float(fitted.mu),
            "alpha": float(fitted.alpha),
            "beta": float(fitted.beta),
            "model_version": fitted.model_version,
        }

    def import_author_state(self, author_id: str, payload: dict[str, object]) -> None:
        self._fitted[author_id] = _FittedHawkes(
            base_minute=int(payload["base_minute"]),
            step_minutes=int(payload["step_minutes"]),
            coding_prob=np.asarray(payload["coding_prob"], dtype=np.float32),
            mu=float(payload["mu"]),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            model_version=str(payload.get("model_version", self.model_version)),
        )

    def _build_coding_prob_curve(
        self,
        times: np.ndarray,
        horizon: float,
        mu: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        steps = int(math.ceil(horizon / self.step_minutes)) + 1
        counts = np.zeros(steps, dtype=int)
        for ti in times:
            idx = int(ti / self.step_minutes)
            if 0 <= idx < steps:
                counts[idx] += 1

        g = 0.0
        prob = np.zeros(steps, dtype=float)
        decay = math.exp(-beta * self.step_minutes)
        for i in range(steps):
            g *= decay
            if counts[i] > 0:
                g += float(counts[i])
            lam = mu + alpha * g
            # Map excitation above baseline to a coding probability.
            delta = max(0.0, lam - mu)
            p = 1.0 - math.exp(-delta / max(1e-6, mu))
            prob[i] = min(1.0, max(0.0, p))
        return prob

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

        if steps <= 256:
            draws = rng.uniform(0.0, 1.0, size=(n_samples, steps))
            active = (draws < probs[None, :]).astype(np.float32)
            return active.sum(axis=1) * minutes_per_step

        mean = float(np.sum(probs) * minutes_per_step)
        var = float(np.sum(probs * (1.0 - probs)) * (minutes_per_step ** 2))
        std = math.sqrt(max(1e-9, var))
        samples = rng.normal(loc=mean, scale=std, size=n_samples)
        samples = np.clip(samples, 0.0, float(b - a))
        return samples.astype(np.float32)
