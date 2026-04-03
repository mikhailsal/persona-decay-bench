"""Extended evaluator tests to improve coverage."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.config import ModelConfig
from src.evaluator import (
    _validate_scores,
    evaluate_model,
    run_observer_assessment,
)
from src.openrouter_client import CompletionResult, UsageInfo

_ALL_ITEM_IDS = ["IN-1", "IN-2", "IN-3", "IN-4", "HY-1", "HY-2", "HY-3", "HY-4", "IM-1", "IM-2", "IM-3", "IM-4"]


def _make_all_scores(val: int) -> str:
    return json.dumps(dict.fromkeys(_ALL_ITEM_IDS, val))


class TestValidateScores:
    def test_valid_full(self):
        data = {
            "IN-1": 3,
            "IN-2": 2,
            "IN-3": 1,
            "IN-4": 0,
            "HY-1": 3,
            "HY-2": 2,
            "HY-3": 1,
            "HY-4": 0,
            "IM-1": 3,
            "IM-2": 2,
            "IM-3": 1,
            "IM-4": 0,
        }
        result = _validate_scores(data)
        assert result is not None
        assert len(result) == 12

    def test_partial_scores(self):
        data = {
            "IN-1": 3,
            "IN-2": 2,
            "IN-3": 1,
            "IN-4": 0,
            "HY-1": 3,
            "HY-2": 2,
            "HY-3": 1,
        }
        result = _validate_scores(data)
        assert result is not None
        assert len(result) == 7

    def test_too_few_returns_none(self):
        data = {"IN-1": 3, "IN-2": 2}
        result = _validate_scores(data)
        assert result is None

    def test_non_numeric_values_skipped(self):
        data = {
            "IN-1": "high",
            "IN-2": 2,
            "IN-3": 1,
            "IN-4": 0,
            "HY-1": 3,
            "HY-2": 2,
            "HY-3": 1,
            "HY-4": 0,
            "IM-1": 3,
            "IM-2": 2,
            "IM-3": 1,
            "IM-4": 0,
        }
        result = _validate_scores(data)
        assert result is not None
        assert "IN-1" not in result


class TestRunObserverMixed:
    def test_mixed_valid_and_invalid(self):
        client = MagicMock()
        valid = _make_all_scores(2)
        invalid = "I cannot parse this"

        client.chat.side_effect = [
            CompletionResult(content=valid, usage=UsageInfo(cost_usd=0.01), model="obs", finish_reason="stop"),
            CompletionResult(content=invalid, usage=UsageInfo(cost_usd=0.01), model="obs", finish_reason="stop"),
            CompletionResult(content=valid, usage=UsageInfo(cost_usd=0.01), model="obs", finish_reason="stop"),
        ]

        turns = [{"turn": 1, "role": "participant", "content": "Test"}]
        result = run_observer_assessment(client, turns, up_to_turn=6, n_calls=3)
        assert len(result["observer_ratings"]) == 3
        assert result["observer_mean"] > 0
        assert result["observer_sd"] == 0.0


class TestEvaluateModel:
    @patch("src.evaluator.evaluate_checkpoint")
    @patch("src.evaluator.list_checkpoints", return_value=[6, 12])
    @patch("src.evaluator.list_conversations", return_value=["conv001"])
    def test_evaluates_all(self, mock_convs, mock_cps, mock_eval_cp):
        mock_eval_cp.return_value = {"turn": 6, "observer_mean": 15.0}
        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        # Pass runs explicitly to avoid needing to mock list_available_runs
        results = evaluate_model(client, cfg, runs=[1])
        assert len(results) == 1
        assert 6 in results[0]["checkpoints"]
        assert 12 in results[0]["checkpoints"]
        assert mock_eval_cp.call_count == 2

    def test_no_runs(self):
        client = MagicMock()
        cfg = ModelConfig(model_id="test/model")
        # Pass empty runs list explicitly
        results = evaluate_model(client, cfg, runs=[])
        assert results == []

    @patch("src.evaluator.evaluate_checkpoint")
    @patch("src.evaluator.list_checkpoints", return_value=[6])
    @patch("src.evaluator.list_conversations", return_value=["conv1", "conv2"])
    def test_multiple_runs_and_convs(self, mock_convs, mock_cps, mock_eval_cp):
        mock_eval_cp.return_value = {"turn": 6, "observer_mean": 15.0}
        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        results = evaluate_model(client, cfg, runs=[1, 2])
        assert len(results) == 4  # 2 runs x 2 conversations

    @patch("src.evaluator.evaluate_checkpoint")
    @patch("src.evaluator.list_checkpoints", return_value=[6, 12])
    @patch("src.evaluator.list_conversations", return_value=["conv001"])
    def test_parallel_evaluation(self, mock_convs, mock_cps, mock_eval_cp):
        mock_eval_cp.return_value = {"turn": 6, "observer_mean": 20.0}
        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        results = evaluate_model(client, cfg, runs=[1], parallel=4)
        assert len(results) == 1
        assert 6 in results[0]["checkpoints"]
        assert 12 in results[0]["checkpoints"]
        assert mock_eval_cp.call_count == 2

    @patch("src.evaluator.evaluate_checkpoint")
    @patch("src.evaluator.list_checkpoints", return_value=[6, 12])
    @patch("src.evaluator.list_conversations", return_value=["conv1", "conv2"])
    def test_parallel_multiple_convs(self, mock_convs, mock_cps, mock_eval_cp):
        mock_eval_cp.return_value = {"turn": 6, "observer_mean": 18.0}
        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        results = evaluate_model(client, cfg, runs=[1, 2], parallel=8)
        assert len(results) == 4  # 2 runs x 2 conversations
        assert mock_eval_cp.call_count == 8  # 4 convs x 2 checkpoints


class TestEvaluateCheckpointCachedObserver:
    @patch("src.evaluator.load_conversation")
    @patch("src.evaluator.load_checkpoint")
    def test_uses_cached_observer(self, mock_load_cp, mock_load_conv):
        mock_load_cp.return_value = {
            "self_report": {
                "raw_response": _make_all_scores(2),
            },
            "observer_ratings": [
                {
                    "items": {
                        "IN-1": 2,
                        "IN-2": 2,
                        "IN-3": 1,
                        "IN-4": 1,
                        "HY-1": 2,
                        "HY-2": 1,
                        "HY-3": 1,
                        "HY-4": 1,
                        "IM-1": 1,
                        "IM-2": 1,
                        "IM-3": 2,
                        "IM-4": 1,
                    },
                    "total_score": 16,
                },
                {
                    "items": {
                        "IN-1": 2,
                        "IN-2": 2,
                        "IN-3": 1,
                        "IN-4": 1,
                        "HY-1": 2,
                        "HY-2": 1,
                        "HY-3": 1,
                        "HY-4": 1,
                        "IM-1": 1,
                        "IM-2": 1,
                        "IM-3": 2,
                        "IM-4": 1,
                    },
                    "total_score": 16,
                },
            ],
            "observer_mean": 16.0,
            "observer_sd": 0.0,
        }
        mock_load_conv.return_value = [
            {"turn": 1, "role": "participant", "content": "My day..."},
        ]

        from src.evaluator import evaluate_checkpoint

        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = evaluate_checkpoint(client, cfg, 1, "conv001", 6)
        assert result["observer_mean"] == 16.0
        client.chat.assert_not_called()


class TestEvaluateCheckpointNoConversation:
    @patch("src.evaluator.load_conversation", return_value=[])
    @patch("src.evaluator.load_checkpoint")
    def test_no_conversation(self, mock_load_cp, mock_load_conv):
        mock_load_cp.return_value = {
            "self_report": {"raw_response": '{"IN-1": 3}'},
        }

        from src.evaluator import evaluate_checkpoint

        client = MagicMock()
        cfg = ModelConfig(model_id="test/model")
        result = evaluate_checkpoint(client, cfg, 1, "conv001", 6)
        assert "error" in result
