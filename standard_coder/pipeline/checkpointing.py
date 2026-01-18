from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


@dataclass
class RunCheckpoint:
    """Persistent run state for resumable pipelines.

    This tracks which stages have completed and stores per-stage metadata,
    e.g. progress counters.

    Design goals:
    - Always write atomically (avoid corrupted state on interruption).
    - Be tolerant to missing keys (forward-compatible).
    """

    path: Path
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RunCheckpoint":
        if path.exists():
            try:
                raw = json.loads(path.read_text())
            except Exception:
                raw = {}
        else:
            raw = {}
        return cls(path=path, state=raw)

    def save(self) -> None:
        self.state.setdefault("updated_at", time.time())
        _atomic_write_json(self.path, self.state)

    def is_stage_done(self, stage: str, *, fingerprint: str | None = None) -> bool:
        stage_state = self.state.get("stages", {}).get(stage, {})
        done = bool(stage_state.get("done", False))
        if not done:
            return False
        if fingerprint is None:
            return True
        meta = stage_state.get("meta", {}) or {}
        return str(meta.get("fingerprint", "")) == str(fingerprint)

    def mark_stage_done(
        self,
        stage: str,
        meta: dict[str, Any] | None = None,
        *,
        fingerprint: str | None = None,
    ) -> None:
        stages = self.state.setdefault("stages", {})
        stages.setdefault(stage, {})
        stages[stage]["done"] = True
        if meta or fingerprint is not None:
            out_meta = stages[stage].setdefault("meta", {})
            if meta:
                out_meta.update(meta)
            if fingerprint is not None:
                out_meta["fingerprint"] = fingerprint
        self.save()

    def stage_meta(self, stage: str) -> dict[str, Any]:
        return dict(self.state.get("stages", {}).get(stage, {}).get("meta", {}))

    def update_stage_meta(self, stage: str, meta: dict[str, Any]) -> None:
        stages = self.state.setdefault("stages", {})
        stages.setdefault(stage, {})
        stages[stage].setdefault("meta", {}).update(meta)
        self.save()
