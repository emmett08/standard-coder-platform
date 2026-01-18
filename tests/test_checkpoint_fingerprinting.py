from __future__ import annotations

import json
from pathlib import Path

from standard_coder.pipeline.checkpointing import RunCheckpoint


def test_checkpoint_fingerprint_gates_stage_skip(tmp_path: Path) -> None:
    ck_path = tmp_path / "run_state.json"
    ck = RunCheckpoint.load(ck_path)

    assert not ck.is_stage_done("mine", fingerprint="a")

    ck.mark_stage_done("mine", meta={"repos": 1}, fingerprint="fp1")
    assert ck.is_stage_done("mine")
    assert ck.is_stage_done("mine", fingerprint="fp1")
    assert not ck.is_stage_done("mine", fingerprint="fp2")

    raw = json.loads(ck_path.read_text())
    assert raw["stages"]["mine"]["meta"]["fingerprint"] == "fp1"

