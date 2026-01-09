from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class SeedContext:
    seed: int


def set_global_seed(seed: int) -> SeedContext:
    """Set seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    return SeedContext(seed=seed)
