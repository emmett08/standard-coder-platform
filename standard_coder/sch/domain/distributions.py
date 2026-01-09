from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import scipy.stats as st

from standard_coder.sch.domain.value_objects import Distribution


@dataclass(frozen=True)
class GaussianMixtureDistribution(Distribution):
    """A 1D Gaussian mixture distribution."""

    weights: np.ndarray  # (k,)
    means: np.ndarray  # (k,)
    stds: np.ndarray  # (k,)
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if not (self.means.shape == self.stds.shape == self.weights.shape):
            raise ValueError("weights, means, stds must share shape")
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-4):
            raise ValueError("weights must sum to 1")

    def _component_cdfs(self, x: np.ndarray) -> np.ndarray:
        return st.norm.cdf((x[:, None] - self.means[None, :]) / self.stds[None, :])

    def mean(self) -> float:
        m = float(np.sum(self.weights * self.means))
        if self.low is None and self.high is None:
            return m
        samples = self.sample(20000, np.random.default_rng(0))
        return float(np.mean(samples))

    def quantile(self, q: float) -> float:
        if q <= 0.0:
            return float(self.low if self.low is not None else -np.inf)
        if q >= 1.0:
            return float(self.high if self.high is not None else np.inf)

        # Numeric inversion using binary search on a reasonable bracket.
        lo = self.low if self.low is not None else float(np.min(self.means - 6 * self.stds))
        hi = self.high if self.high is not None else float(np.max(self.means + 6 * self.stds))

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            cdf_mid = float(self.cdf(mid))
            if cdf_mid < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def cdf(self, x: float) -> float:
        x_arr = np.array([x], dtype=float)
        cdfs = self._component_cdfs(x_arr)[0]
        cdf_val = float(np.sum(self.weights * cdfs))

        if self.low is None and self.high is None:
            return cdf_val

        # Truncate by renormalising.
        low = self.low if self.low is not None else -np.inf
        high = self.high if self.high is not None else np.inf

        cdf_low = float(np.sum(self.weights * st.norm.cdf((low - self.means) / self.stds)))
        cdf_high = float(np.sum(self.weights * st.norm.cdf((high - self.means) / self.stds)))
        denom = max(1e-12, cdf_high - cdf_low)

        if x <= low:
            return 0.0
        if x >= high:
            return 1.0
        return (cdf_val - cdf_low) / denom

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.weights.shape[0]
        w = self.weights / max(1e-12, float(np.sum(self.weights)))
        comp = rng.choice(k, size=n, p=w)
        x = rng.normal(loc=self.means[comp], scale=self.stds[comp])

        if self.low is None and self.high is None:
            return x

        low = -np.inf if self.low is None else self.low
        high = np.inf if self.high is None else self.high
        return np.clip(x, low, high)


@dataclass(frozen=True)
class QuantileDistribution(Distribution):
    """A simple distribution defined by monotone quantiles.

    Internally we use a piecewise-linear CDF between anchor quantiles.
    """

    quantiles: dict[float, float]

    def __post_init__(self) -> None:
        qs = sorted(self.quantiles.keys())
        if not qs:
            raise ValueError("At least one quantile required")
        if any(q <= 0.0 or q >= 1.0 for q in qs):
            raise ValueError("Quantile keys must be in (0, 1)")
        vals = [self.quantiles[q] for q in qs]
        if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            raise ValueError("Quantile values must be non-decreasing")

    def mean(self) -> float:
        rng = np.random.default_rng(0)
        return float(np.mean(self.sample(20000, rng)))

    def quantile(self, q: float) -> float:
        qs = sorted(self.quantiles.keys())
        if q <= qs[0]:
            return float(self.quantiles[qs[0]])
        if q >= qs[-1]:
            return float(self.quantiles[qs[-1]])

        for i in range(len(qs) - 1):
            q0, q1 = qs[i], qs[i + 1]
            if q0 <= q <= q1:
                v0, v1 = self.quantiles[q0], self.quantiles[q1]
                t = (q - q0) / max(1e-12, q1 - q0)
                return float(v0 + t * (v1 - v0))
        return float(self.quantiles[qs[-1]])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        return np.array([self.quantile(float(x)) for x in u], dtype=float)


def fit_lognormal_from_quantiles(
    p50: float,
    p90: float,
) -> st.lognorm:
    """Fit a log-normal distribution given two quantiles.

    This is useful when you only have p50/p90 and want continuous sampling.
    """
    p50 = max(1e-9, p50)
    p90 = max(1e-9, p90)

    z50 = st.norm.ppf(0.5)
    z90 = st.norm.ppf(0.9)

    # log(X) ~ N(mu, sigma)
    mu = math.log(p50) - 0.0 * z50
    sigma = (math.log(p90) - math.log(p50)) / max(1e-12, (z90 - z50))
    sigma = max(1e-6, sigma)

    return st.lognorm(s=sigma, scale=math.exp(mu))
