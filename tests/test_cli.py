"""Tests for cli.py: command parsing, argument validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import cli, _resolve_models
from src.config import MODEL_CONFIGS, ModelConfig


@pytest.fixture
def runner():
    return CliRunner()


class TestResolveModels:
    def setup_method(self):
        self._orig = dict(MODEL_CONFIGS)

    def teardown_method(self):
        MODEL_CONFIGS.clear()
        MODEL_CONFIGS.update(self._orig)

    def test_explicit_models(self):
        configs = _resolve_models("test/model-a,test/model-b")
        assert len(configs) == 2
        assert configs[0].model_id == "test/model-a"

    def test_strips_whitespace(self):
        configs = _resolve_models(" test/model-a , test/model-b ")
        assert len(configs) == 2

    def test_none_returns_active(self):
        from src.config import register_config
        cfg = ModelConfig(model_id="test/active-model", display_label="test-active", active=True)
        register_config(cfg)
        configs = _resolve_models(None)
        assert any(c.model_id == "test/active-model" for c in configs)


class TestCliHelp:
    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Persona Decay Benchmark" in result.output

    def test_run_help(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.output

    def test_evaluate_help(self, runner):
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_leaderboard_help(self, runner):
        result = runner.invoke(cli, ["leaderboard", "--help"])
        assert result.exit_code == 0
        assert "--detailed" in result.output

    def test_generate_report_help(self, runner):
        result = runner.invoke(cli, ["generate-report", "--help"])
        assert result.exit_code == 0

    def test_estimate_cost_help(self, runner):
        result = runner.invoke(cli, ["estimate-cost", "--help"])
        assert result.exit_code == 0
        assert "--runs" in result.output

    def test_clear_cache_help(self, runner):
        result = runner.invoke(cli, ["clear-cache", "--help"])
        assert result.exit_code == 0
        assert "--scores-only" in result.output


class TestLeaderboardCommand:
    @patch("src.cache.list_all_cached_models", return_value=[])
    def test_no_cached_data(self, mock_list, runner):
        result = runner.invoke(cli, ["leaderboard"])
        assert result.exit_code == 0
        assert "No cached results" in result.output


class TestClearCacheCommand:
    @patch("src.cache.clear_all_cache", return_value=5)
    def test_clear_all(self, mock_clear, runner):
        result = runner.invoke(cli, ["clear-cache", "--yes"])
        assert result.exit_code == 0
        assert "5" in result.output

    @patch("src.cache.clear_observer_scores", return_value=3)
    def test_clear_scores_only(self, mock_clear, runner):
        result = runner.invoke(cli, ["clear-cache", "--scores-only", "--yes"])
        assert result.exit_code == 0
        assert "3" in result.output


class TestEstimateCostCommand:
    @patch("src.cli.load_api_key", return_value="")
    def test_without_api_key(self, mock_key, runner):
        result = runner.invoke(cli, ["estimate-cost", "--models", "test/model"])
        assert result.exit_code == 0
        assert "Cost Estimate" in result.output


class TestRunCommand:
    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_run_with_models(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(cli, ["run", "--models", "test/model", "--runs", "1"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_run_handles_error(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.side_effect = RuntimeError("API error")
        result = runner.invoke(cli, ["run", "--models", "test/model", "--runs", "1"])
        assert result.exit_code == 0
        assert "Error" in result.output


class TestEvaluateCommand:
    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_evaluate_with_models(self, mock_key, mock_client_cls, mock_eval, runner):
        mock_eval.return_value = [{"checkpoints": {6: {}, 12: {}}}]
        result = runner.invoke(cli, ["evaluate", "--models", "test/model"])
        assert result.exit_code == 0
        mock_eval.assert_called_once()
