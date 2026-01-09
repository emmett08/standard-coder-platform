from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ZenHubApiError(RuntimeError):
    pass


@dataclass
class ZenHubClient:
    token: str
    api_base_url: str = "https://api.zenhub.com"

    def _headers(self) -> dict[str, str]:
        return {
            "X-Authentication-Token": self.token,
            "User-Agent": "standard-coder-platform",
            "Accept": "application/json",
        }

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self.api_base_url.rstrip("/") + path
        resp = requests.get(url, headers=self._headers(), params=params, timeout=60)
        if resp.status_code == 429:
            wait_s = int(resp.headers.get("Retry-After", "10"))
            logger.warning("ZenHub rate limit hit. Sleeping %ss", wait_s)
            time.sleep(max(1, wait_s))
            return self.get(path, params=params)
        if resp.status_code >= 400:
            raise ZenHubApiError(f"GET {path} failed: {resp.status_code} {resp.text}")
        return resp.json()

    # --- V1-style endpoints (widely used in ZenHub API docs historically) ---
    def get_issue_data(self, repo_id: int, issue_number: int) -> Any:
        return self.get(f"/p1/repositories/{repo_id}/issues/{issue_number}")

    def get_board(self, repo_id: int) -> Any:
        return self.get(f"/p1/repositories/{repo_id}/board")

    def get_milestones(self, repo_id: int) -> Any:
        return self.get(f"/p1/repositories/{repo_id}/milestones")

    def get_epics(self, repo_id: int) -> Any:
        return self.get(f"/p1/repositories/{repo_id}/epics")
