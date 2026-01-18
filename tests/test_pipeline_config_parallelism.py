from __future__ import annotations

from standard_coder.pipeline.config import HmmConfig, MdnConfig, MultiTaskConfig, QuantileConfig


def test_new_parallelism_fields_have_defaults() -> None:
    hmm = HmmConfig()
    assert hmm.parallel_authors == 1

    mdn = MdnConfig()
    assert mdn.dataloader_num_workers == 0

    quantile = QuantileConfig()
    assert quantile.dataloader_num_workers == 0

    multitask = MultiTaskConfig()
    assert multitask.dataloader_num_workers == 0

