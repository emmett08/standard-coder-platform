from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from standard_coder.sch.domain.distributions import GaussianMixtureDistribution
from standard_coder.sch.domain.value_objects import EffortEstimate
from standard_coder.sch.interfaces import EffortPredictor


logger = logging.getLogger(__name__)


class _MdnNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_components: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.trunk = nn.Sequential(*layers)

        self.pi = nn.Linear(last, n_components)
        self.mu = nn.Linear(last, n_components)
        self.sigma = nn.Linear(last, n_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        pi_logits = self.pi(z)
        mu = self.mu(z)
        sigma = torch.nn.functional.softplus(self.sigma(z)) + 1e-4
        return pi_logits, mu, sigma


def _mdn_nll(
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # y: (batch, 1)
    y = y.expand_as(mu)
    log_pi = torch.log_softmax(pi_logits, dim=1)
    log_norm = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * (
        (y - mu) / sigma
    ) ** 2
    log_prob = torch.logsumexp(log_pi + log_norm, dim=1)
    return -torch.mean(log_prob)


@dataclass
class MdnEffortPredictor(EffortPredictor):
    """Baseline predictor: Mixture Density Network (MDN)."""

    n_components: int = 10
    hidden_sizes: tuple[int, ...] = (128, 64, 64)
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    unit: str = "sch"
    truncate_low: float | None = 0.0
    truncate_high: float | None = 1.0
    model_version: str = "mdn_v1"
    device: str | None = None

    _net: _MdnNet | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net = _MdnNet(
            in_dim=x.shape[1],
            n_components=self.n_components,
            hidden_sizes=self.hidden_sizes,
        ).to(device)

        ds = TensorDataset(x_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self._net.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                pi_logits, mu, sigma = self._net(xb)
                loss = _mdn_nll(pi_logits, mu, sigma, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=5.0)
                opt.step()
                total += float(loss.item()) * xb.shape[0]

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "MDN epoch %d/%d loss=%.4f",
                    epoch + 1,
                    self.epochs,
                    total / max(1, len(ds)),
                )

    def predict(self, x: np.ndarray) -> Sequence[EffortEstimate]:
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")

        device = next(self._net.parameters()).device
        self._net.eval()

        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32, device=device)
            pi_logits, mu, sigma = self._net(xb)
            pi = torch.softmax(pi_logits, dim=1).cpu().numpy()
            mu_n = mu.cpu().numpy()
            sigma_n = sigma.cpu().numpy()

        estimates: list[EffortEstimate] = []
        for i in range(x.shape[0]):
            dist = GaussianMixtureDistribution(
                weights=pi[i].astype(float),
                means=mu_n[i].astype(float),
                stds=sigma_n[i].astype(float),
                low=self.truncate_low,
                high=self.truncate_high,
            )
            estimates.append(
                EffortEstimate(
                    distribution=dist,
                    unit=self.unit,
                    model_version=self.model_version,
                )
            )
        return estimates
