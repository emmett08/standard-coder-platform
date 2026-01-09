from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from standard_coder.sch.domain.distributions import GaussianMixtureDistribution, QuantileDistribution
from standard_coder.sch.domain.value_objects import EffortEstimate, MultiTaskEffortEstimate
from standard_coder.sch.interfaces import MultiTaskEffortPredictor


logger = logging.getLogger(__name__)


HeadKind = Literal["mdn", "quantile"]


@dataclass(frozen=True)
class HeadConfig:
    kind: HeadKind
    n_components: int = 10  # mdn
    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9)  # quantile
    truncate_low: float | None = 0.0  # mdn
    truncate_high: float | None = 1.0  # mdn


class _Trunk(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MdnHead(nn.Module):
    def __init__(self, in_dim: int, n_components: int) -> None:
        super().__init__()
        self.pi = nn.Linear(in_dim, n_components)
        self.mu = nn.Linear(in_dim, n_components)
        self.sigma = nn.Linear(in_dim, n_components)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits = self.pi(z)
        mu = self.mu(z)
        sigma = torch.nn.functional.softplus(self.sigma(z)) + 1e-4
        return pi_logits, mu, sigma


class _QuantileHead(nn.Module):
    def __init__(self, in_dim: int, q: int) -> None:
        super().__init__()
        self.out = nn.Linear(in_dim, q)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.out(z)
        base = torch.nn.functional.softplus(raw[:, :1])
        if raw.shape[1] == 1:
            return base
        inc = torch.nn.functional.softplus(raw[:, 1:])
        rest = base + torch.cumsum(inc, dim=1)
        return torch.cat([base, rest], dim=1)


def _mdn_nll(
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    y = y.expand_as(mu)
    log_pi = torch.log_softmax(pi_logits, dim=1)
    log_norm = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * (
        (y - mu) / sigma
    ) ** 2
    log_prob = torch.logsumexp(log_pi + log_norm, dim=1)
    return -torch.mean(log_prob)


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    diff = target - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff)
    return torch.mean(loss)


class _MultiTaskNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        trunk_sizes: tuple[int, ...],
        coding: HeadConfig,
        delivery: HeadConfig,
    ) -> None:
        super().__init__()
        self.coding_cfg = coding
        self.delivery_cfg = delivery

        self.trunk = _Trunk(in_dim, trunk_sizes)

        self.coding_head_mdn: _MdnHead | None = None
        self.coding_head_q: _QuantileHead | None = None
        if coding.kind == "mdn":
            self.coding_head_mdn = _MdnHead(trunk_sizes[-1] if trunk_sizes else in_dim, coding.n_components)
        else:
            self.coding_head_q = _QuantileHead(trunk_sizes[-1] if trunk_sizes else in_dim, len(coding.quantiles))

        self.delivery_head_mdn: _MdnHead | None = None
        self.delivery_head_q: _QuantileHead | None = None
        if delivery.kind == "mdn":
            self.delivery_head_mdn = _MdnHead(trunk_sizes[-1] if trunk_sizes else in_dim, delivery.n_components)
        else:
            self.delivery_head_q = _QuantileHead(trunk_sizes[-1] if trunk_sizes else in_dim, len(delivery.quantiles))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        z = self.trunk(x)

        if self.coding_cfg.kind == "mdn":
            assert self.coding_head_mdn is not None
            coding_out = self.coding_head_mdn(z)
        else:
            assert self.coding_head_q is not None
            coding_out = (self.coding_head_q(z),)

        if self.delivery_cfg.kind == "mdn":
            assert self.delivery_head_mdn is not None
            delivery_out = self.delivery_head_mdn(z)
        else:
            assert self.delivery_head_q is not None
            delivery_out = (self.delivery_head_q(z),)

        return coding_out, delivery_out


@dataclass
class TorchMultiTaskEffortPredictor(MultiTaskEffortPredictor):
    """Upgrade C: multi-task predictor (coding SCH + delivery effort)."""

    coding_head: HeadConfig = HeadConfig(kind="mdn", n_components=10)
    delivery_head: HeadConfig = HeadConfig(kind="quantile", quantiles=(0.5, 0.8, 0.9))
    trunk_sizes: tuple[int, ...] = (128, 64, 64)
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 256
    unit: str = "sch"
    model_version: str = "multitask_v1"
    device: str | None = None

    _net: _MultiTaskNet | None = None

    def fit(self, x: np.ndarray, y_coding: np.ndarray, y_delivery: np.ndarray) -> None:
        x_t = torch.tensor(x, dtype=torch.float32)
        y1 = torch.tensor(y_coding.reshape(-1, 1), dtype=torch.float32)
        y2 = torch.tensor(y_delivery.reshape(-1, 1), dtype=torch.float32)

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net = _MultiTaskNet(
            in_dim=x.shape[1],
            trunk_sizes=self.trunk_sizes,
            coding=self.coding_head,
            delivery=self.delivery_head,
        ).to(device)

        ds = TensorDataset(x_t, y1, y2)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        if self.coding_head.kind == "quantile":
            q1 = torch.tensor(self.coding_head.quantiles, dtype=torch.float32, device=device).view(1, -1)
        else:
            q1 = None

        if self.delivery_head.kind == "quantile":
            q2 = torch.tensor(self.delivery_head.quantiles, dtype=torch.float32, device=device).view(1, -1)
        else:
            q2 = None

        self._net.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb1, yb2 in dl:
                xb = xb.to(device)
                yb1 = yb1.to(device)
                yb2 = yb2.to(device)

                opt.zero_grad(set_to_none=True)
                (c_out, d_out) = self._net(xb)

                loss = torch.tensor(0.0, device=device)
                if self.coding_head.kind == "mdn":
                    pi, mu, sig = c_out  # type: ignore[misc]
                    loss = loss + _mdn_nll(pi, mu, sig, yb1)
                else:
                    (pred,) = c_out  # type: ignore[misc]
                    assert q1 is not None
                    loss = loss + _pinball_loss(pred, yb1, q1)

                if self.delivery_head.kind == "mdn":
                    pi, mu, sig = d_out  # type: ignore[misc]
                    loss = loss + _mdn_nll(pi, mu, sig, yb2)
                else:
                    (pred,) = d_out  # type: ignore[misc]
                    assert q2 is not None
                    loss = loss + _pinball_loss(pred, yb2, q2)

                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=5.0)
                opt.step()

                total += float(loss.item()) * xb.shape[0]

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "Multi-task epoch %d/%d loss=%.4f",
                    epoch + 1,
                    self.epochs,
                    total / max(1, len(ds)),
                )

    def predict(self, x: np.ndarray) -> Sequence[MultiTaskEffortEstimate]:
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")

        device = next(self._net.parameters()).device
        self._net.eval()

        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32, device=device)
            coding_out, delivery_out = self._net(xb)

        # Convert outputs to distributions.
        results: list[MultiTaskEffortEstimate] = []

        if self.coding_head.kind == "mdn":
            pi_logits, mu, sigma = coding_out  # type: ignore[misc]
            pi = torch.softmax(pi_logits, dim=1).cpu().numpy()
            mu_n = mu.cpu().numpy()
            sig_n = sigma.cpu().numpy()
        else:
            (pred_q,) = coding_out  # type: ignore[misc]
            pred_q_n = pred_q.cpu().numpy()

        if self.delivery_head.kind == "mdn":
            pi_logits2, mu2, sigma2 = delivery_out  # type: ignore[misc]
            pi2 = torch.softmax(pi_logits2, dim=1).cpu().numpy()
            mu2_n = mu2.cpu().numpy()
            sig2_n = sigma2.cpu().numpy()
        else:
            (pred_q2,) = delivery_out  # type: ignore[misc]
            pred_q2_n = pred_q2.cpu().numpy()

        for i in range(x.shape[0]):
            if self.coding_head.kind == "mdn":
                dist1 = GaussianMixtureDistribution(
                    weights=pi[i].astype(float),
                    means=mu_n[i].astype(float),
                    stds=sig_n[i].astype(float),
                    low=self.coding_head.truncate_low,
                    high=self.coding_head.truncate_high,
                )
            else:
                q_map = {
                    float(self.coding_head.quantiles[j]): float(pred_q_n[i, j])
                    for j in range(len(self.coding_head.quantiles))
                }
                dist1 = QuantileDistribution(quantiles=q_map)

            coding_est = EffortEstimate(
                distribution=dist1,
                unit=self.unit,
                model_version=f"{self.model_version}:coding",
            )

            if self.delivery_head.kind == "mdn":
                dist2 = GaussianMixtureDistribution(
                    weights=pi2[i].astype(float),
                    means=mu2_n[i].astype(float),
                    stds=sig2_n[i].astype(float),
                    low=self.delivery_head.truncate_low,
                    high=self.delivery_head.truncate_high,
                )
            else:
                q_map2 = {
                    float(self.delivery_head.quantiles[j]): float(pred_q2_n[i, j])
                    for j in range(len(self.delivery_head.quantiles))
                }
                dist2 = QuantileDistribution(quantiles=q_map2)

            delivery_est = EffortEstimate(
                distribution=dist2,
                unit=self.unit,
                model_version=f"{self.model_version}:delivery",
            )

            results.append(MultiTaskEffortEstimate(coding=coding_est, delivery=delivery_est))

        return results
