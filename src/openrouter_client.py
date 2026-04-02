"""OpenRouter client: OpenAI SDK wrapper with retry logic, cost tracking, timing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import requests
from openai import OpenAI

from src.config import (
    API_CALL_TIMEOUT,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
    get_openrouter_base_url,
    get_reasoning_effort,
    ModelPricing,
)

log = logging.getLogger(__name__)

OPENROUTER_PUBLIC_MODELS_URL = "https://openrouter.ai/api/v1/models"

MAX_RETRY_WAIT = 30.0


@dataclass
class UsageInfo:
    """Token usage and cost for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class CompletionResult:
    """Result of a chat completion call."""
    content: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)
    model: str = ""
    finish_reason: str = ""
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None


def _extract_cost(usage_obj: Any) -> float | None:
    """Try to extract the actual cost from an OpenRouter usage object.

    Returns USD cost as float, or None if the field is absent / unparseable.
    """
    raw = getattr(usage_obj, "cost", None)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _usage_from_response(
    *,
    response: Any,
    elapsed: float,
) -> UsageInfo:
    """Build UsageInfo from an OpenRouter / OpenAI chat completion response.

    Uses the per-request ``usage.cost`` field returned by OpenRouter (and by
    proxies that forward it) as the authoritative cost.  Falls back to zero
    when the field is absent — we never pre-fetch the pricing catalog just
    for this.
    """
    usage = UsageInfo(elapsed_seconds=elapsed)
    if not response.usage:
        return usage

    ru = response.usage
    usage.prompt_tokens = int(ru.prompt_tokens or 0)
    usage.completion_tokens = int(ru.completion_tokens or 0)

    details = getattr(ru, "prompt_tokens_details", None)
    if details:
        usage.cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        usage.cache_write_tokens = int(getattr(details, "cache_write_tokens", 0) or 0)

    cost = _extract_cost(ru)
    usage.cost_usd = cost if cost is not None else 0.0
    return usage


class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0
    RETRYABLE_STATUS_CODES = {402, 429, 500, 502, 503}
    EMPTY_CONTENT_RETRIES = 1

    def __init__(self, api_key: str, timeout: float = API_CALL_TIMEOUT) -> None:
        self.api_key = api_key
        self._base_url = get_openrouter_base_url()
        self._timeout = timeout
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            default_headers={
                "HTTP-Referer": OPENROUTER_APP_URL,
                "X-Title": OPENROUTER_APP_NAME,
            },
        )
        self._known_models: set[str] | None = None

    # ------------------------------------------------------------------
    # Model discovery (public OpenRouter API, no auth needed)
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_public_models() -> list[dict[str, Any]]:
        """Fetch the full model list from the real OpenRouter API (no key needed).

        Returns raw model dicts with id, pricing, supported_parameters, etc.
        """
        resp = requests.get(OPENROUTER_PUBLIC_MODELS_URL, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", [])

    @staticmethod
    def fetch_public_pricing() -> dict[str, ModelPricing]:
        """Fetch per-model pricing from the real OpenRouter API (no key needed)."""
        models = OpenRouterClient.fetch_public_models()
        result: dict[str, ModelPricing] = {}
        for m in models:
            mid = m.get("id", "")
            p = m.get("pricing", {})
            result[mid] = ModelPricing(
                prompt_price=float(p.get("prompt", "0")),
                completion_price=float(p.get("completion", "0")),
            )
        return result

    def fetch_available_models(self) -> set[str]:
        """Fetch model IDs available through the configured endpoint (proxy or direct)."""
        if self._known_models is not None:
            return self._known_models
        url = self._base_url.rstrip("/") + "/models"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        self._known_models = {
            m.get("id", "") for m in resp.json().get("data", [])
        }
        return self._known_models

    def validate_model(self, model_id: str) -> bool:
        """Check if a model is available on the configured endpoint."""
        try:
            return model_id in self.fetch_available_models()
        except Exception:
            return True  # optimistic: assume available on network errors

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        *,
        reasoning_effort: str | None = None,
        provider: str | None = None,
        cache_control: bool = False,
    ) -> CompletionResult:
        """Send a chat completion request with retry logic for empty responses.

        When ``cache_control`` is True, a request-level ``cache_control``
        directive is added so that OpenRouter activates prompt caching for
        providers that require an explicit opt-in (e.g. Gemini 3.x).
        """
        use_reasoning = self._resolve_reasoning_effort(reasoning_effort)
        accumulated = UsageInfo()

        for attempt in range(1, self.EMPTY_CONTENT_RETRIES + 2):
            result = self._chat_single(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=use_reasoning,
                provider=provider,
                cache_control=cache_control,
            )

            accumulated.prompt_tokens += result.usage.prompt_tokens
            accumulated.completion_tokens += result.usage.completion_tokens
            accumulated.cached_tokens += result.usage.cached_tokens
            accumulated.cache_write_tokens += result.usage.cache_write_tokens
            accumulated.cost_usd += result.usage.cost_usd
            accumulated.elapsed_seconds += result.usage.elapsed_seconds

            if result.content:
                result.usage = accumulated
                return result

            if attempt <= self.EMPTY_CONTENT_RETRIES:
                reason = (
                    "reasoning_only"
                    if result.usage.completion_tokens > 0
                    else f"error_or_empty (finish_reason={result.finish_reason})"
                )
                log.warning(
                    "%s: empty response (%s, %d tokens), retry %d/%d",
                    model, reason,
                    result.usage.completion_tokens,
                    attempt, self.EMPTY_CONTENT_RETRIES,
                )
                if result.usage.completion_tokens == 0:
                    time.sleep(2.0 * attempt)
                continue

            result.usage = accumulated
            return result

        return result  # type: ignore[possibly-undefined]

    @staticmethod
    def _resolve_reasoning_effort(override: str | None) -> str | None:
        """Resolve reasoning effort from the caller-provided override.

        The explicit value from the model config is authoritative — no need
        to probe the models endpoint.
        """
        if override is None or override in ("off", "none"):
            return None
        if override == "auto":
            return None
        return override

    def _chat_single(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
        provider: str | None = None,
        cache_control: bool = False,
    ) -> CompletionResult:
        """Execute a single chat completion with error retry logic and timing."""
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                extra_body: dict[str, Any] | None = None
                if reasoning_effort:
                    extra_body = {"reasoning": {"effort": reasoning_effort}}

                if provider:
                    extra_body = extra_body or {}
                    extra_body["provider"] = {
                        "order": [provider],
                        "allow_fallbacks": False,
                    }

                if cache_control:
                    extra_body = extra_body or {}
                    extra_body["cache_control"] = {"type": "ephemeral"}

                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                t0 = time.monotonic()
                response = self._client.chat.completions.create(**kwargs)
                elapsed = time.monotonic() - t0

                if elapsed > 30:
                    log.info(
                        "%s: slow response (%.1fs)", model, elapsed,
                    )

                finish_reason = ""
                content = ""
                if response.choices:
                    finish_reason = response.choices[0].finish_reason or ""
                    if response.choices[0].message.content:
                        content = response.choices[0].message.content.strip()

                reasoning_content = None
                reasoning_details = None
                if response.choices:
                    msg = response.choices[0].message
                    raw_reasoning = getattr(msg, "reasoning", None)
                    if raw_reasoning and isinstance(raw_reasoning, str):
                        reasoning_content = raw_reasoning.strip()
                    if not reasoning_content:
                        raw_rc = getattr(msg, "reasoning_content", None)
                        if raw_rc and isinstance(raw_rc, str):
                            reasoning_content = raw_rc.strip()

                    raw_details = getattr(msg, "reasoning_details", None)
                    if raw_details and isinstance(raw_details, list):
                        reasoning_details = [
                            d if isinstance(d, dict) else (d.__dict__ if hasattr(d, "__dict__") else {"type": "unknown"})
                            for d in raw_details
                        ]

                usage = _usage_from_response(
                    response=response,
                    elapsed=elapsed,
                )

                if usage.cached_tokens > 0:
                    pct = (usage.cached_tokens / usage.prompt_tokens * 100) if usage.prompt_tokens else 0
                    log.info(
                        "%s: cache READ — %d/%d prompt tokens (%.0f%%)",
                        model, usage.cached_tokens, usage.prompt_tokens, pct,
                    )
                if usage.cache_write_tokens > 0:
                    log.info(
                        "%s: cache WRITE — %d tokens written to cache",
                        model, usage.cache_write_tokens,
                    )
                if usage.cached_tokens == 0 and usage.cache_write_tokens == 0 and usage.prompt_tokens > 0:
                    log.debug(
                        "%s: no cache activity — %d prompt tokens",
                        model, usage.prompt_tokens,
                    )

                return CompletionResult(
                    content=content,
                    usage=usage,
                    model=model,
                    finish_reason=finish_reason,
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details,
                )

            except Exception as e:
                last_error = e
                status_code = getattr(e, "status_code", None)
                if status_code and status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self.MAX_RETRIES:
                        wait = min(
                            self.RETRY_BACKOFF_BASE ** (attempt + 1),
                            MAX_RETRY_WAIT,
                        )
                        log.warning(
                            "%s: HTTP %s, retry %d/%d in %.0fs",
                            model, status_code, attempt + 1,
                            self.MAX_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                raise

        raise last_error or RuntimeError("Chat completion failed after retries")
