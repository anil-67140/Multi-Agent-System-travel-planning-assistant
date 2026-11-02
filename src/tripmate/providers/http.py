"""httpx-backed implementation of the ``HttpClient`` protocol.

Failure classification is the important part: it drives the retry policy.

* timeouts, connection errors, HTTP 5xx and 429 -> retryable
* other 4xx and malformed JSON -> NOT retryable (retrying cannot help)
"""

from __future__ import annotations

from typing import Any

from tripmate.errors import ProviderError


def _retry_after(headers: Any) -> float | None:
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None  # an HTTP-date form; fall back to our own backoff


class HttpxClient:
    def __init__(
        self, *, timeout: float = 10.0, user_agent: str = "tripmate/3.0", client: Any = None
    ) -> None:
        import httpx  # imported lazily so the rest of the package works without it

        self._httpx = httpx
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=False,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

    async def get_json(
        self,
        url: str,
        *,
        provider: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return await self._request("GET", url, provider, params=params, headers=headers)

    async def post_json(
        self,
        url: str,
        *,
        provider: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        return await self._request("POST", url, provider, json=json, headers=headers)

    async def _request(self, method: str, url: str, provider: str, **kwargs: Any) -> Any:
        httpx = self._httpx
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderError(provider, "request timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                provider, f"network error ({type(exc).__name__})", retryable=True
            ) from exc

        status = response.status_code
        if status == 429:
            raise ProviderError(
                provider,
                "rate limited",
                retryable=True,
                retry_after=_retry_after(response.headers),
                status_code=status,
            )
        if status >= 500:
            raise ProviderError(
                provider, f"upstream error {status}", retryable=True, status_code=status
            )
        if status >= 400:
            raise ProviderError(
                provider, f"request rejected ({status})", retryable=False, status_code=status
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(provider, "response was not valid JSON", retryable=False) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
