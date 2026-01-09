from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)


class GitHubApiError(RuntimeError):
    pass


@dataclass
class GitHubClient:
    token: str
    api_base_url: str = "https://api.github.com"
    user_agent: str = "standard-coder-platform"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self.api_base_url.rstrip("/") + path
        while True:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=60)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset:
                    wait_s = max(1, int(reset) - int(time.time()) + 1)
                    logger.warning("GitHub rate limit hit. Sleeping %ss", wait_s)
                    time.sleep(wait_s)
                    continue
            if resp.status_code >= 400:
                raise GitHubApiError(f"GET {path} failed: {resp.status_code} {resp.text}")
            return resp.json()

    def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        per_page: int = 100,
        max_pages: int = 50,
    ) -> Iterator[Any]:
        params = dict(params or {})
        params["per_page"] = per_page
        page = 1
        while page <= max_pages:
            params["page"] = page
            data = self.get(path, params=params)
            if not isinstance(data, list):
                return
            if not data:
                return
            for item in data:
                yield item
            page += 1

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> Any:
        url = self.api_base_url.rstrip("/") + "/graphql"
        payload = {"query": query, "variables": variables or {}}
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        if resp.status_code >= 400:
            raise GitHubApiError(f"GraphQL failed: {resp.status_code} {resp.text}")
        data = resp.json()
        if "errors" in data:
            raise GitHubApiError(f"GraphQL errors: {data['errors']}")
        return data.get("data")

    def get_repo_id(self, owner: str, repo: str) -> int:
        data = self.get(f"/repos/{owner}/{repo}")
        return int(data["id"])
