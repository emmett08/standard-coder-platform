from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from standard_coder.sch.interfaces import WorkInferenceModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FittedStateSpace:
    base_minute: int
    step_minutes: int
    coding_prob: np.ndarray
    model_version: str


class VariationalStateSpaceWorkInference(WorkInferenceModel):
    """Upgrade B: state-space style work inference (approximate EKF).

    We model a latent log-intensity random walk and treat commit counts as
    Poisson observations. We then map inferred intensity to a coding
    probability curve.

    This is an approximation intended for engineering scaffolding:
    the EKF here is a simple Gaussian approximation to the non-linear
    Poisson observation model.
    """

    def __init__(
        self,
        step_minutes: int = 5,
        min_commits: int = 50,
        process_var: float = 0.05,
        model_version: str = "state_space_v1",
    ) -> None:
        self.step_minutes = int(step_minutes)
        self.min_commits = int(min_commits)
        self.process_var = float(process_var)
        self.model_version = model_version
        self._fitted: dict[str, _FittedStateSpace] = {}

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
            steps = int(math.ceil(horizon / self.step_minutes)) + 1

            counts = np.zeros(steps, dtype=float)
            for ti in t0:
                idx = int(ti / self.step_minutes)
                if 0 <= idx < steps:
                    counts[idx] += 1.0

            coding_prob = self._infer_prob_curve(counts)
            self._fitted[author_id] = _FittedStateSpace(
                base_minute=base,
                step_minutes=self.step_minutes,
                coding_prob=coding_prob.astype(np.float32),
                model_version=self.model_version,
            )

    def _infer_prob_curve(self, y: np.ndarray) -> np.ndarray:
        dt = float(self.step_minutes)
        steps = int(y.shape[0])

        # EKF parameters
        q = self.process_var
        m = 0.0  # log intensity
        P = 1.0

        m_filt = np.zeros(steps, dtype=float)
        P_filt = np.zeros(steps, dtype=float)

        for t in range(steps):
            # Predict
            m_pred = m
            P_pred = P + q

            # Observation model: h(x) = exp(x) * dt
            lam = math.exp(m_pred) * dt
            H = lam  # derivative wrt x
            R = max(1e-6, lam)  # Poisson variance approx

            S = H * P_pred * H + R
            K = (P_pred * H) / max(1e-12, S)

            m = m_pred + K * (float(y[t]) - lam)
            P = (1.0 - K * H) * P_pred

            m_filt[t] = m
            P_filt[t] = P

        intensity = np.exp(m_filt)  # per minute-ish scale
        baseline = float(np.quantile(intensity, 0.2))
        baseline = max(1e-6, baseline)

        prob = np.zeros(steps, dtype=float)
        for i in range(steps):
            delta = max(0.0, float(intensity[i] - baseline))
            p = 1.0 - math.exp(-delta / baseline)
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
