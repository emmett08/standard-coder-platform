from __future__ import annotations

import numpy as np
import torch

from standard_coder.sch.predictors.mdn import MdnEffortPredictor


def test_mdn_predictor_trains_and_predicts_fast() -> None:
    torch.set_num_threads(1)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    y = (0.2 + 0.05 * x[:, 0] + rng.normal(0.0, 0.02, size=120)).astype(np.float32)
    y = np.clip(y, 0.0, 1.0)

    pred = MdnEffortPredictor(epochs=2, batch_size=64, n_components=3)
    pred.fit(x, y)

    out = pred.predict(x[:3])
    assert len(out) == 3
    assert all(0.0 <= e.mean() <= 1.0 for e in out)
