"""Extended CLI tests to improve coverage of cli.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.config import MODEL_CONFIGS, ModelConfig
from src.scorer import DecayCurve, DimensionScores, ModelScore


@pytest.fixture
def runner():
    return CliRunner()


class TestLeaderboardWithData:
    @patch("src.leaderboard.display_leaderboard")
    @patch("src.scorer.score_model")
    @patch("src.config.get_config_by_dir_name")
    @patch("src.cache.list_all_cached_models", return_value=["test--model@none-t0.7"])
    def test_leaderboard_with_cached(self, mock_list, mock_get_cfg, mock_score, mock_display, runner):
        mock_get_cfg.return_value = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        mock_score.return_value = ModelScore(
            model_id="test/model",
            stability_index=85.0,
            n_conversations=5,
        )
        result = runner.invoke(cli, ["leaderboard"])
        assert result.exit_code == 0
        mock_display.assert_called_once()

    @patch("src.leaderboard.display_leaderboard")
    @patch("src.scorer.score_model")
    @patch("src.config.get_config_by_dir_name", return_value=None)
    @patch("src.cache.list_all_cached_models", return_value=["test--model@none-t0.7"])
    def test_leaderboard_no_config(self, mock_list, mock_get_cfg, mock_score, mock_display, runner):
        mock_score.return_value = ModelScore(
            model_id="test/model",
            stability_index=85.0,
            n_conversations=5,
        )
        result = runner.invoke(cli, ["leaderboard"])
        assert result.exit_code == 0

    @patch("src.leaderboard.display_leaderboard")
    @patch("src.scorer.score_model")
    @patch("src.config.get_config_by_dir_name")
    @patch("src.cache.list_all_cached_models", return_value=["test--model@none-t0.7"])
    def test_leaderboard_detailed(self, mock_list, mock_get_cfg, mock_score, mock_display, runner):
        mock_get_cfg.return_value = ModelConfig(model_id="test/model")
        mock_score.return_value = ModelScore(model_id="test/model", stability_index=85.0, n_conversations=5)
        result = runner.invoke(cli, ["leaderboard", "--detailed"])
        assert result.exit_code == 0
        call_kwargs = mock_display.call_args
        assert call_kwargs[1]["detailed"] is True


class TestGenerateReportCommand:
    @patch("src.leaderboard.export_json", return_value={})
    @patch("src.leaderboard.generate_markdown_report", return_value="# Report")
    @patch("src.scorer.score_model")
    @patch("src.config.get_config_by_dir_name")
    @patch("src.cache.list_all_cached_models", return_value=["test--model@none-t0.7"])
    def test_generates_report(self, mock_list, mock_get_cfg, mock_score, mock_md, mock_json, runner):
        mock_get_cfg.return_value = ModelConfig(model_id="test/model")
        mock_score.return_value = ModelScore(model_id="test/model", stability_index=85.0, n_conversations=5)
        result = runner.invoke(cli, ["generate-report"])
        assert result.exit_code == 0
        assert "Generated" in result.output
        mock_md.assert_called_once()
        mock_json.assert_called_once()

    @patch("src.cache.list_all_cached_models", return_value=[])
    def test_no_data(self, mock_list, runner):
        result = runner.invoke(cli, ["generate-report"])
        assert result.exit_code == 0
        assert "No cached results" in result.output


class TestEstimateCostWithPricing:
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_with_api_key(self, mock_key, runner):
        from src.config import ModelPricing

        with patch("src.openrouter_client.OpenRouterClient") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            instance.fetch_pricing.return_value = {
                "test/model": ModelPricing(0.001, 0.002)
            }
            instance.get_model_pricing.return_value = ModelPricing(0.001, 0.002)

            result = runner.invoke(cli, ["estimate-cost", "--models", "test/model"])
            assert result.exit_code == 0
            assert "Cost Estimate" in result.output


class TestRunWithMultipleModels:
    @patch("src.runner.run_all_conversations")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_multiple_models(self, mock_key, mock_client_cls, mock_run, runner):
        mock_run.return_value = [{"status": "completed"}]
        result = runner.invoke(cli, ["run", "--models", "test/a,test/b", "--runs", "1"])
        assert result.exit_code == 0
        assert mock_run.call_count == 2


class TestEvaluateWithError:
    @patch("src.evaluator.evaluate_model")
    @patch("src.openrouter_client.OpenRouterClient")
    @patch("src.cli.load_api_key", return_value="sk-test")
    def test_evaluate_handles_error(self, mock_key, mock_client_cls, mock_eval, runner):
        mock_eval.side_effect = RuntimeError("Eval error")
        result = runner.invoke(cli, ["evaluate", "--models", "test/model"])
        assert result.exit_code == 0
        assert "Error" in result.output
