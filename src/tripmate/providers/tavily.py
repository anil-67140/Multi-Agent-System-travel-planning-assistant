"""Tavily web search (free tier), used to find hotel and attraction sources."""

from __future__ import annotations

from typing import Any

from tripmate.errors import ProviderError
from tripmate.providers.base import HttpClient, WebHit

PROVIDER = "tavily"
SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchClient:
    def __init__(self, http: HttpClient, api_key: str) -> None:
        if not api_key:
            raise ValueError("Tavily API key is required")
        self._http = http
        self._key = api_key

    async def search(self, query: str, max_results: int = 6) -> list[WebHit]:
        data: Any = await self._http.post_json(
            SEARCH_URL,
            provider=PROVIDER,
            # Bearer is the current auth scheme; ``api_key`` in the body is accepted by older API versions.
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "api_key": self._key,
                "query": query[:400],
                "search_depth": "basic",
                "max_results": max(1, min(max_results, 10)),
                "include_answer": False,
            },
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ProviderError(PROVIDER, "unexpected search response", retryable=False)
        hits: list[WebHit] = []
        for item in results:
            try:
                url = str(item["url"])
                if not url.lower().startswith(("http://", "https://")):
                    continue
                hits.append(
                    WebHit(
                        title=str(item.get("title") or url)[:300],
                        url=url[:2000],
                        content=str(item.get("content") or "")[:4000],
                        score=float(item["score"]) if item.get("score") is not None else None,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed entries rather than fail the whole search
        return hits
