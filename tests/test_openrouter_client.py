"""Tests for openrouter_client.py: retry logic, cost tracking, response parsing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelPricing
from src.openrouter_client import (
    CompletionResult,
    OpenRouterClient,
    UsageInfo,
    _usage_from_openrouter_response,
)


def _make_mock_response(
    content: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cost: float | None = None,
    finish_reason: str = "stop",
    reasoning: str | None = None,
):
    """Build a mock OpenAI chat completion response."""
    msg = SimpleNamespace(content=content, reasoning=reasoning, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage_attrs = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cost is not None:
        usage_attrs["cost"] = cost
    usage = SimpleNamespace(**usage_attrs)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestUsageInfo:
    def test_defaults(self):
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.cost_usd == 0.0

    def test_custom_values(self):
        u = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
        assert u.prompt_tokens == 100
        assert u.cost_usd == 0.01


class TestCompletionResult:
    def test_defaults(self):
        r = CompletionResult()
        assert r.content == ""
        assert r.model == ""
        assert r.reasoning_content is None

    def test_with_content(self):
        r = CompletionResult(content="Hi", model="test/model")
        assert r.content == "Hi"


class TestUsageFromResponse:
    def test_with_api_cost(self):
        response = _make_mock_response(cost=0.05)
        usage = _usage_from_openrouter_response(
            model="test/model",
            response=response,
            elapsed=1.0,
            get_model_pricing=lambda m: ModelPricing(),
        )
        assert usage.cost_usd == 0.05
        assert usage.prompt_tokens == 10

    def test_with_string_cost(self):
        response = _make_mock_response(cost="0.03")
        # Override cost attr
        response.usage.cost = "0.03"
        usage = _usage_from_openrouter_response(
            model="test/model",
            response=response,
            elapsed=1.0,
            get_model_pricing=lambda m: ModelPricing(),
        )
        assert usage.cost_usd == 0.03

    def test_fallback_to_pricing(self):
        response = _make_mock_response(cost=None)
        # Remove cost attr
        if hasattr(response.usage, "cost"):
            delattr(response.usage, "cost")
        pricing = ModelPricing(prompt_price=0.001, completion_price=0.002)
        usage = _usage_from_openrouter_response(
            model="test/model",
            response=response,
            elapsed=2.0,
            get_model_pricing=lambda m: pricing,
        )
        expected = 10 * 0.001 + 20 * 0.002
        assert abs(usage.cost_usd - expected) < 1e-9

    def test_no_usage(self):
        response = SimpleNamespace(choices=[], usage=None)
        usage = _usage_from_openrouter_response(
            model="test/model",
            response=response,
            elapsed=0.5,
            get_model_pricing=lambda m: ModelPricing(),
        )
        assert usage.prompt_tokens == 0
        assert usage.elapsed_seconds == 0.5


class TestOpenRouterClient:
    def setup_method(self):
        self.client = OpenRouterClient.__new__(OpenRouterClient)
        self.client.api_key = "test-key"
        self.client._pricing_cache = {
            "test/model": ModelPricing(prompt_price=0.001, completion_price=0.002),
        }
        self.client._reasoning_models = {"test/reasoning-model"}
        self.client._client = MagicMock()

    def test_get_model_pricing(self):
        pricing = self.client.get_model_pricing("test/model")
        assert pricing.prompt_price == 0.001

    def test_get_model_pricing_default(self):
        pricing = self.client.get_model_pricing("unknown/model")
        assert pricing.prompt_price == 0.0

    def test_validate_model_true(self):
        assert self.client.validate_model("test/model") is True

    def test_validate_model_false(self):
        assert self.client.validate_model("unknown/model") is False

    def test_supports_reasoning(self):
        assert self.client.supports_reasoning("test/reasoning-model") is True
        assert self.client.supports_reasoning("test/model") is False

    def test_chat_success(self):
        mock_response = _make_mock_response(content="Test response", cost=0.01)
        self.client._client.chat.completions.create.return_value = mock_response

        result = self.client.chat("test/model", [{"role": "user", "content": "hi"}])
        assert result.content == "Test response"
        assert result.usage.cost_usd == 0.01

    def test_chat_empty_content_retries(self):
        empty_response = _make_mock_response(content="", completion_tokens=50, cost=0.005)
        good_response = _make_mock_response(content="Got it", cost=0.01)

        self.client._client.chat.completions.create.side_effect = [
            empty_response, good_response,
        ]

        result = self.client.chat("test/model", [{"role": "user", "content": "hi"}])
        assert result.content == "Got it"
        assert self.client._client.chat.completions.create.call_count == 2

    def test_chat_with_reasoning_content(self):
        response = _make_mock_response(content="Answer", reasoning="Thinking...")
        self.client._client.chat.completions.create.return_value = response

        result = self.client.chat("test/model", [{"role": "user", "content": "hi"}])
        assert result.reasoning_content == "Thinking..."

    def test_chat_with_provider(self):
        mock_response = _make_mock_response(content="Response", cost=0.01)
        self.client._client.chat.completions.create.return_value = mock_response

        result = self.client.chat(
            "test/model",
            [{"role": "user", "content": "hi"}],
            provider="test/provider",
        )
        assert result.content == "Response"
        call_kwargs = self.client._client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["provider"]["order"] == ["test/provider"]

    def test_resolve_reasoning_off(self):
        assert self.client._resolve_reasoning_effort("test/model", "off") is None

    def test_resolve_reasoning_explicit(self):
        assert self.client._resolve_reasoning_effort("test/model", "high") == "high"

    def test_resolve_reasoning_auto_no_support(self):
        assert self.client._resolve_reasoning_effort("test/model", "auto") is None

    def test_resolve_reasoning_auto_with_support(self):
        result = self.client._resolve_reasoning_effort("test/reasoning-model", "auto")
        assert result is not None


class TestFetchPricing:
    def test_fetch_pricing_caches(self):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.api_key = "test"
        client._pricing_cache = {"cached/model": ModelPricing(0.1, 0.2)}
        client._reasoning_models = set()

        result = client.fetch_pricing()
        assert "cached/model" in result

    @patch("src.openrouter_client.requests.get")
    def test_fetch_pricing_from_api(self, mock_get):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.api_key = "test"
        client._pricing_cache = {}
        client._reasoning_models = set()

        mock_get.return_value.json.return_value = {
            "data": [
                {
                    "id": "api/model",
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                    "supported_parameters": ["reasoning"],
                },
            ],
        }
        mock_get.return_value.raise_for_status = MagicMock()

        result = client.fetch_pricing()
        assert "api/model" in result
        assert result["api/model"].prompt_price == 0.001
        assert client.supports_reasoning("api/model")
