from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from standard_coder.sch.domain.distributions import QuantileDistribution
from standard_coder.sch.domain.value_objects import EffortEstimate
from standard_coder.sch.interfaces import EffortPredictor


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


class _QuantileNet(nn.Module):
    def __init__(self, in_dim: int, quantiles: tuple[float, ...], hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.quantiles = quantiles
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(last, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        raw = self.out(z)
        # Enforce monotone quantiles by predicting positive increments.
        base = torch.nn.functional.softplus(raw[:, :1])
        if raw.shape[1] == 1:
            return base
        inc = torch.nn.functional.softplus(raw[:, 1:])
        rest = base + torch.cumsum(inc, dim=1)
        return torch.cat([base, rest], dim=1)


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    # pred: (batch, q), target: (batch, 1)
    diff = target - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff)
    return torch.mean(loss)


@dataclass
class QuantileEffortPredictor(EffortPredictor):
    """Upgrade D: quantile regression for effort distributions."""

    quantiles: tuple[float, ...] = (0.5, 0.8, 0.9)
    hidden_sizes: tuple[int, ...] = (128, 64, 64)
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    unit: str = "sch"
    model_version: str = "quantile_v1"
    device: str | None = None

    _net: _QuantileNet | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        device = _select_device(self.device)
        self._net = _QuantileNet(
            in_dim=x.shape[1],
            quantiles=self.quantiles,
            hidden_sizes=self.hidden_sizes,
        ).to(device)

        ds = TensorDataset(x_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        q_t = torch.tensor(self.quantiles, dtype=torch.float32, device=device).view(1, -1)

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self._net.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                pred = self._net(xb)
                loss = _pinball_loss(pred, yb, q_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=5.0)
                opt.step()
                total += float(loss.item()) * xb.shape[0]

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "Quantile epoch %d/%d loss=%.4f",
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
            pred = self._net(xb).cpu().numpy()

        estimates: list[EffortEstimate] = []
        for i in range(x.shape[0]):
            q_map = {float(self.quantiles[j]): float(pred[i, j]) for j in range(len(self.quantiles))}
            dist = QuantileDistribution(quantiles=q_map)
            estimates.append(
                EffortEstimate(
                    distribution=dist,
                    unit=self.unit,
                    model_version=self.model_version,
                )
            )
        return estimates
