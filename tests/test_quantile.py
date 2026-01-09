from __future__ import annotations

import numpy as np
import torch

from standard_coder.sch.predictors.quantile import QuantileEffortPredictor


def test_quantile_predictor_trains_and_predicts_fast() -> None:
    torch.set_num_threads(1)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    y = (0.3 + 0.1 * x[:, 0] + rng.normal(0.0, 0.05, size=120)).astype(np.float32)
    y = np.clip(y, 0.0, 1.0)

    pred = QuantileEffortPredictor(epochs=2, batch_size=64, quantiles=(0.5, 0.9))
    pred.fit(x, y)

    out = pred.predict(x[:3])
    assert len(out) == 3
    for e in out:
        assert e.p50() <= e.p90()
