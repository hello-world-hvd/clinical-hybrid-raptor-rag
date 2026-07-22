from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_OPENROUTER_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
DEFAULT_MIN_INTERVAL_SECONDS = 3.0
DEFAULT_MAX_REQUESTS_PER_RUN = 40

LOGGER = logging.getLogger("openrouter")
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenRouterRequestBudgetExceeded(RuntimeError):
    """Raised before sending a request when the configured budget is exhausted."""


class OpenRouterRateLimiter:
    """Thread-safe request pacing and per-process request budget."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        max_requests: int | None = DEFAULT_MAX_REQUESTS_PER_RUN,
    ) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.max_requests = max_requests if max_requests is None else max(0, max_requests)
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._request_count = 0

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def acquire(self) -> None:
        with self._lock:
            if (
                self.max_requests is not None
                and self._request_count >= self.max_requests
            ):
                raise OpenRouterRequestBudgetExceeded(
                    "OpenRouter request budget exhausted "
                    f"({self._request_count}/{self.max_requests})"
                )

            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_at - now)
            reserved_at = max(now, self._next_request_at)
            self._next_request_at = reserved_at + self.min_interval_seconds
            self._request_count += 1

        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def defer(self, delay_seconds: float) -> None:
        """Delay the next request after a provider rate-limit/server response."""
        with self._lock:
            self._next_request_at = max(
                self._next_request_at,
                time.monotonic() + max(0.0, delay_seconds),
            )


_LIMITERS: dict[tuple[str, float, int | None], OpenRouterRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"none", "unlimited", "off", "0"}:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def shared_rate_limiter(
    *,
    api_key: str,
    min_interval_seconds: float,
    max_requests: int | None,
) -> OpenRouterRateLimiter:
    key = (api_key, min_interval_seconds, max_requests)
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = OpenRouterRateLimiter(
                min_interval_seconds=min_interval_seconds,
                max_requests=max_requests,
            )
            _LIMITERS[key] = limiter
        return limiter


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


class _OpenRouterBaseClient:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        timeout: float,
        max_retries: int,
        session: Optional[requests.Session],
        min_interval_seconds: float | None,
        max_requests_per_run: int | None,
        rate_limiter: OpenRouterRateLimiter | None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured in the environment or .env"
            )

        interval = (
            _env_float(
                "OPENROUTER_MIN_INTERVAL_SECONDS",
                DEFAULT_MIN_INTERVAL_SECONDS,
            )
            if min_interval_seconds is None
            else max(0.0, min_interval_seconds)
        )
        request_budget = (
            _env_optional_int(
                "OPENROUTER_MAX_REQUESTS_PER_RUN",
                DEFAULT_MAX_REQUESTS_PER_RUN,
            )
            if max_requests_per_run is None
            else (None if max_requests_per_run <= 0 else max_requests_per_run)
        )

        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.session = session or requests.Session()
        self.rate_limiter = rate_limiter or shared_rate_limiter(
            api_key=self.api_key,
            min_interval_seconds=interval,
            max_requests=request_budget,
        )

    @property
    def request_count(self) -> int:
        return self.rate_limiter.request_count

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        base = min(2**attempt, 30)
        return base + random.uniform(0.0, min(1.0, base * 0.25))

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.acquire()
                response = self.session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    delay = _retry_after_seconds(response)
                    if delay is None:
                        delay = self._backoff_seconds(attempt)
                    last_error = requests.HTTPError(
                        f"OpenRouter returned HTTP {response.status_code}",
                        response=response,
                    )
                    if attempt + 1 < self.max_retries:
                        LOGGER.warning(
                            "OpenRouter HTTP %s; retrying in %.1fs (%s/%s)",
                            response.status_code,
                            delay,
                            attempt + 1,
                            self.max_retries,
                        )
                        self.rate_limiter.defer(delay)
                    continue

                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("OpenRouter returned a non-object response")
                if data.get("error"):
                    raise ValueError(f"OpenRouter API error: {data['error']}")
                return data
            except OpenRouterRequestBudgetExceeded:
                raise
            except (requests.RequestException, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    delay = self._backoff_seconds(attempt)
                    LOGGER.warning(
                        "OpenRouter request failed; retrying in %.1fs (%s/%s): %s",
                        delay,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    self.rate_limiter.defer(delay)

        raise RuntimeError(
            f"OpenRouter request failed after {self.max_retries} attempts"
        ) from last_error


class OpenRouterClient(_OpenRouterBaseClient):
    """OpenRouter chat client with shared pacing, budget, and retry handling."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 90.0,
        max_retries: int = 4,
        session: Optional[requests.Session] = None,
        min_interval_seconds: float | None = None,
        max_requests_per_run: int | None = None,
        rate_limiter: OpenRouterRateLimiter | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            min_interval_seconds=min_interval_seconds,
            max_requests_per_run=max_requests_per_run,
            rate_limiter=rate_limiter,
        )
        self.model = model

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data = self._post_json(OPENROUTER_URL, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("OpenRouter chat response has no message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter returned an empty chat response")
        return content.strip()


class OpenRouterRerankClient(_OpenRouterBaseClient):
    """OpenRouter rerank client preserving candidate indices."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_RERANK_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 90.0,
        max_retries: int = 4,
        session: Optional[requests.Session] = None,
        min_interval_seconds: float | None = None,
        max_requests_per_run: int | None = None,
        rate_limiter: OpenRouterRateLimiter | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            session=session,
            min_interval_seconds=min_interval_seconds,
            max_requests_per_run=max_requests_per_run,
            rate_limiter=rate_limiter,
        )
        self.model = model

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        if not documents:
            return []

        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = min(top_n, len(documents))

        data = self._post_json(OPENROUTER_RERANK_URL, payload)
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError("OpenRouter rerank response has no results")
        return results
