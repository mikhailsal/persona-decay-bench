"""Tests for cli.py: command parsing, argument validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli import _eval_single_model, _ModelEvalResult, _ModelRunResult, _resolve_models, _run_single_model, cli
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
        assert "ERROR" in result.output

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_run_verbose_flag(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(cli, ["run", "--models", "test/model", "--runs", "1", "--verbose"])
        assert result.exit_code == 0
        assert "Verbose mode: ON" in result.output
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["verbose"] is True

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_run_parallel(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(
            cli,
            ["run", "--models", "test/a,test/b", "--runs", "1", "--parallel", "2"],
        )
        assert result.exit_code == 0
        assert "Parallel model workers: 2" in result.output
        assert mock_run.call_count == 2

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_run_parallel_with_error(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.side_effect = RuntimeError("boom")
        result = runner.invoke(
            cli,
            ["run", "--models", "test/a,test/b", "--runs", "1", "--parallel", "2"],
        )
        assert result.exit_code == 0
        assert "failed" in result.output.lower()


class TestEvaluateCommand:
    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_evaluate_with_models(self, mock_key, mock_client_cls, mock_eval, runner):
        mock_eval.return_value = [{"checkpoints": {6: {}, 12: {}}}]
        result = runner.invoke(cli, ["evaluate", "--models", "test/model"])
        assert result.exit_code == 0
        mock_eval.assert_called_once()

    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_evaluate_verbose_flag(self, mock_key, mock_client_cls, mock_eval, runner):
        mock_eval.return_value = [{"checkpoints": {6: {}}}]
        result = runner.invoke(cli, ["evaluate", "--models", "test/model", "--verbose"])
        assert result.exit_code == 0
        assert "Verbose mode: ON" in result.output
        call_kwargs = mock_eval.call_args
        assert call_kwargs[1]["verbose"] is True

    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_evaluate_parallel(self, mock_key, mock_client_cls, mock_eval, runner):
        mock_eval.return_value = [{"checkpoints": {6: {}, 12: {}}}]
        result = runner.invoke(
            cli,
            ["evaluate", "--models", "test/a,test/b", "--parallel", "2"],
        )
        assert result.exit_code == 0
        assert "Parallel workers: 2" in result.output
        assert mock_eval.call_count == 2

    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_evaluate_parallel_with_error(self, mock_key, mock_client_cls, mock_eval, runner):
        mock_eval.side_effect = RuntimeError("eval boom")
        result = runner.invoke(
            cli,
            ["evaluate", "--models", "test/a,test/b", "--parallel", "2"],
        )
        assert result.exit_code == 0
        assert "failed" in result.output.lower()


class TestRunSingleModel:
    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    def test_success(self, mock_client_cls, mock_run):
        mock_run.return_value = [{"status": "completed"}, {"status": "cached"}]
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        result = _run_single_model("sk-test", cfg, 2, 10, 60.0, False)
        assert isinstance(result, _ModelRunResult)
        assert result.completed == 2
        assert result.error is None

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    def test_error(self, mock_client_cls, mock_run):
        mock_run.side_effect = RuntimeError("boom")
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        result = _run_single_model("sk-test", cfg, 1, 10, 60.0, False)
        assert result.error is not None

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    def test_passes_parallel_runs(self, mock_client_cls, mock_run):
        mock_run.return_value = [{"status": "completed"}]
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        _run_single_model("sk-test", cfg, 1, 10, 60.0, False, parallel_runs=5)
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["parallel_runs"] == 5


class TestRunCommandParallelRuns:
    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_parallel_runs_option(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(
            cli,
            ["run", "--models", "test/model", "--runs", "3", "--parallel-runs", "3"],
        )
        assert result.exit_code == 0
        assert "Parallel runs per model: 3" in result.output
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["parallel_runs"] == 3

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_parallel_runs_disables_verbose(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(
            cli,
            ["run", "--models", "test/model", "--runs", "2", "--parallel-runs", "2", "--verbose"],
        )
        assert result.exit_code == 0
        assert "Verbose mode disabled" in result.output

    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_both_parallel_options(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(
            cli,
            ["run", "--models", "test/a,test/b", "--runs", "3", "--parallel", "2", "--parallel-runs", "3"],
        )
        assert result.exit_code == 0
        assert "Parallel model workers: 2" in result.output
        assert "Parallel runs per model: 3" in result.output


class TestEvalSingleModel:
    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    def test_success(self, mock_client_cls, mock_eval):
        mock_eval.return_value = [{"checkpoints": {6: {}, 12: {}}}]
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        result = _eval_single_model("sk-test", cfg, 60.0, False)
        assert isinstance(result, _ModelEvalResult)
        assert result.n_checkpoints == 2
        assert result.error is None

    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    def test_error(self, mock_client_cls, mock_eval):
        mock_eval.side_effect = RuntimeError("eval fail")
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        result = _eval_single_model("sk-test", cfg, 60.0, False)
        assert result.error is not None
