"""Additional tests for coverage gaps in openrouter_client, cache, runner, scorer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelConfig, ModelPricing
from src.openrouter_client import CompletionResult, OpenRouterClient, UsageInfo


class TestOpenRouterClientRetry:
    def setup_method(self):
        self.client = OpenRouterClient.__new__(OpenRouterClient)
        self.client.api_key = "test-key"
        self.client._base_url = "https://openrouter.ai/api/v1"
        self.client._timeout = 60
        self.client._known_models = {"test/model"}
        self.client._client = MagicMock()

    def test_retryable_error(self):
        error = Exception("Server error")
        error.status_code = 500
        self.client._client.chat.completions.create.side_effect = error
        self.client.RETRY_BACKOFF_BASE = 0.01
        self.client.MAX_RETRIES = 1

        with pytest.raises(Exception, match="Server error"):
            self.client._chat_single(
                model="test/model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0.7,
            )

    def test_non_retryable_error(self):
        error = Exception("Bad request")
        error.status_code = 400
        self.client._client.chat.completions.create.side_effect = error

        with pytest.raises(Exception, match="Bad request"):
            self.client._chat_single(
                model="test/model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0.7,
            )

    def test_chat_all_empty_retries_exhausted(self):
        msg = SimpleNamespace(content=None, reasoning=None, reasoning_content=None, tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=0)
        response = SimpleNamespace(choices=[choice], usage=usage)

        self.client._client.chat.completions.create.return_value = response
        self.client.EMPTY_CONTENT_RETRIES = 1

        with patch("src.openrouter_client.time.sleep"), \
             patch("src.openrouter_client.time.monotonic", side_effect=[0.0, 0.1, 0.2, 0.3]):
            result = self.client.chat("test/model", [{"role": "user", "content": "hi"}])
            assert result.content == ""

    def test_chat_with_reasoning_effort(self):
        msg = SimpleNamespace(content="answer", reasoning=None, reasoning_content=None, tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        response = SimpleNamespace(choices=[choice], usage=usage)
        self.client._client.chat.completions.create.return_value = response

        result = self.client.chat(
            "test/model",
            [{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
        assert result.content == "answer"
        call_kwargs = self.client._client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["reasoning"]["effort"] == "high"

    def test_no_choices(self):
        response = SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))
        self.client._client.chat.completions.create.return_value = response
        self.client.EMPTY_CONTENT_RETRIES = 0

        with patch("src.openrouter_client.time.sleep"), \
             patch("src.openrouter_client.time.monotonic", side_effect=[0.0, 0.1]):
            result = self.client.chat("test/model", [{"role": "user", "content": "hi"}])
            assert result.content == ""

    def test_reasoning_content_field(self):
        msg = SimpleNamespace(content="answer", reasoning=None, reasoning_content="thinking...")
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, cost=0.01)
        response = SimpleNamespace(choices=[choice], usage=usage)
        self.client._client.chat.completions.create.return_value = response

        result = self.client.chat("test/model", [{"role": "user", "content": "hi"}])
        assert result.reasoning_content == "thinking..."


class TestOpenRouterInit:
    @patch("src.openrouter_client.OpenAI")
    def test_constructor(self, mock_openai):
        client = OpenRouterClient("test-key", timeout=60.0)
        assert client.api_key == "test-key"
        mock_openai.assert_called_once()


class TestCacheEdgeCases:
    def test_load_conversation_corrupt_line(self, tmp_path):
        with patch("src.cache.CACHE_DIR", tmp_path):
            from src.cache import _conversation_path, load_conversation
            path = _conversation_path("model@t0.7", 1, "conv001")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"turn": 1}\nnot-json\n{"turn": 2}\n')
            turns = load_conversation("model@t0.7", 1, "conv001")
            assert len(turns) == 2

    def test_load_checkpoint_corrupt(self, tmp_path):
        with patch("src.cache.CACHE_DIR", tmp_path):
            from src.cache import _checkpoint_path, load_checkpoint
            path = _checkpoint_path("model@t0.7", 1, "conv001", 6)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not json")
            result = load_checkpoint("model@t0.7", 1, "conv001", 6)
            assert result is None


class TestRunnerRunAllConversations:
    @patch("src.runner.run_conversation")
    def test_runs_n_conversations(self, mock_run_conv):
        mock_run_conv.return_value = {"status": "completed", "conversation_id": "test"}
        from src.runner import run_all_conversations
        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        results = run_all_conversations(client, cfg, n_runs=3, max_turns=6)
        assert len(results) == 3
        assert mock_run_conv.call_count == 3


class TestScorerEdgeCases:
    @patch("src.scorer.list_conversations", return_value=["conv001"])
    @patch("src.scorer.list_available_runs", return_value=[1])
    @patch("src.scorer.list_checkpoints", return_value=[])
    @patch("src.scorer.load_checkpoint", return_value=None)
    def test_no_checkpoints(self, mock_cp, mock_cps, mock_runs, mock_convs):
        from src.scorer import score_model
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        ms = score_model("test/model", config=cfg)
        assert ms.stability_index == 0.0

    def test_dimension_scores_single_extended(self):
        from src.scorer import compute_dimension_scores
        data = [
            {"turn": 6, "observer_mean": 17.0, "self_report_total": 20.0},
            {"turn": 12, "observer_mean": 16.0, "self_report_total": 20.0},
            {"turn": 18, "observer_mean": 15.0, "self_report_total": 20.0},
            {"turn": 24, "observer_mean": 14.0, "self_report_total": 20.0},
        ]
        dims = compute_dimension_scores(data)
        assert dims.extended_stability == 7.0  # single extended checkpoint


class TestMainModule:
    def test_main_module_exists(self):
        """The __main__.py module should import cli from src.cli."""
        from pathlib import Path
        main_path = Path(__file__).resolve().parent.parent / "src" / "__main__.py"
        assert main_path.exists()
        content = main_path.read_text()
        assert "from src.cli import cli" in content
