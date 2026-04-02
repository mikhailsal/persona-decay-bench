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
    _extract_cost,
    _usage_from_response,
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


class TestExtractCost:
    def test_float_cost(self):
        assert _extract_cost(SimpleNamespace(cost=0.05)) == 0.05

    def test_int_cost(self):
        assert _extract_cost(SimpleNamespace(cost=1)) == 1.0

    def test_string_cost(self):
        assert _extract_cost(SimpleNamespace(cost="0.03")) == 0.03

    def test_none_cost(self):
        assert _extract_cost(SimpleNamespace(cost=None)) is None

    def test_missing_cost(self):
        assert _extract_cost(SimpleNamespace()) is None

    def test_bool_cost_rejected(self):
        assert _extract_cost(SimpleNamespace(cost=True)) is None

    def test_bad_string(self):
        assert _extract_cost(SimpleNamespace(cost="n/a")) is None


class TestUsageFromResponse:
    def test_with_api_cost(self):
        response = _make_mock_response(cost=0.05)
        usage = _usage_from_response(response=response, elapsed=1.0)
        assert usage.cost_usd == 0.05
        assert usage.prompt_tokens == 10

    def test_with_string_cost(self):
        response = _make_mock_response(cost="0.03")
        response.usage.cost = "0.03"
        usage = _usage_from_response(response=response, elapsed=1.0)
        assert usage.cost_usd == 0.03

    def test_no_cost_falls_back_to_zero(self):
        response = _make_mock_response(cost=None)
        if hasattr(response.usage, "cost"):
            delattr(response.usage, "cost")
        usage = _usage_from_response(response=response, elapsed=2.0)
        assert usage.cost_usd == 0.0
        assert usage.prompt_tokens == 10

    def test_no_usage(self):
        response = SimpleNamespace(choices=[], usage=None)
        usage = _usage_from_response(response=response, elapsed=0.5)
        assert usage.prompt_tokens == 0
        assert usage.elapsed_seconds == 0.5


class TestOpenRouterClient:
    def setup_method(self):
        self.client = OpenRouterClient.__new__(OpenRouterClient)
        self.client.api_key = "test-key"
        self.client._base_url = "https://openrouter.ai/api/v1"
        self.client._timeout = 60
        self.client._known_models = {"test/model", "test/reasoning-model"}
        self.client._client = MagicMock()

    def test_validate_model_true(self):
        assert self.client.validate_model("test/model") is True

    def test_validate_model_false(self):
        assert self.client.validate_model("unknown/model") is False

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
        assert OpenRouterClient._resolve_reasoning_effort("off") is None

    def test_resolve_reasoning_none_string(self):
        assert OpenRouterClient._resolve_reasoning_effort("none") is None

    def test_resolve_reasoning_explicit(self):
        assert OpenRouterClient._resolve_reasoning_effort("high") == "high"

    def test_resolve_reasoning_auto(self):
        assert OpenRouterClient._resolve_reasoning_effort("auto") is None

    def test_resolve_reasoning_low(self):
        assert OpenRouterClient._resolve_reasoning_effort("low") == "low"

    def test_resolve_reasoning_none_value(self):
        assert OpenRouterClient._resolve_reasoning_effort(None) is None

    def test_chat_with_cache_control(self):
        mock_response = _make_mock_response(content="Cached response", cost=0.01)
        self.client._client.chat.completions.create.return_value = mock_response

        result = self.client.chat(
            "test/model",
            [{"role": "user", "content": "hi"}],
            cache_control=True,
        )
        assert result.content == "Cached response"
        call_kwargs = self.client._client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["cache_control"] == {"type": "ephemeral"}


class TestFetchPublicPricing:
    @patch("src.openrouter_client.requests.get")
    def test_fetch_public_models(self, mock_get):
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

        models = OpenRouterClient.fetch_public_models()
        assert len(models) == 1
        assert models[0]["id"] == "api/model"

    @patch("src.openrouter_client.requests.get")
    def test_fetch_public_pricing(self, mock_get):
        mock_get.return_value.json.return_value = {
            "data": [
                {
                    "id": "api/model",
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                },
            ],
        }
        mock_get.return_value.raise_for_status = MagicMock()

        pricing = OpenRouterClient.fetch_public_pricing()
        assert "api/model" in pricing
        assert pricing["api/model"].prompt_price == 0.001


class TestFetchAvailableModels:
    @patch("src.openrouter_client.requests.get")
    def test_fetches_and_caches(self, mock_get):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.api_key = "test"
        client._base_url = "https://openrouter.ai/api/v1"
        client._known_models = None

        mock_get.return_value.json.return_value = {
            "data": [{"id": "model/a"}, {"id": "model/b"}],
        }
        mock_get.return_value.raise_for_status = MagicMock()

        result = client.fetch_available_models()
        assert "model/a" in result
        assert "model/b" in result
        assert mock_get.call_count == 1

        # Second call uses cache
        client.fetch_available_models()
        assert mock_get.call_count == 1
