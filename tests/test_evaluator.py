"""Tests for evaluator.py: score parsing, observer assessment, ICC computation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator import (
    compute_icc,
    compute_total_score,
    evaluate_checkpoint,
    extract_self_report_score,
    parse_caars_scores,
    run_observer_assessment,
)
from src.openrouter_client import CompletionResult, UsageInfo


class TestParseCaarsScores:
    def test_valid_json(self):
        raw = '{"IN-1": 3, "IN-2": 2, "IN-3": 1, "IN-4": 0, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 3, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}'
        scores = parse_caars_scores(raw)
        assert scores is not None
        assert scores["IN-1"] == 3
        assert len(scores) == 12

    def test_json_in_text(self):
        raw = 'Here are my scores:\n{"IN-1": 2, "IN-2": 3, "IN-3": 1, "IN-4": 2, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 0, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}\nThose are my ratings.'
        scores = parse_caars_scores(raw)
        assert scores is not None
        assert len(scores) == 12

    def test_json_in_markdown_block(self):
        raw = '```json\n{"IN-1": 2, "IN-2": 3, "IN-3": 1, "IN-4": 2, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 0, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}\n```'
        scores = parse_caars_scores(raw)
        assert scores is not None

    def test_clamps_values(self):
        raw = '{"IN-1": 5, "IN-2": -1, "IN-3": 1, "IN-4": 2, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 0, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}'
        scores = parse_caars_scores(raw)
        assert scores is not None
        assert scores["IN-1"] == 3  # clamped from 5
        assert scores["IN-2"] == 0  # clamped from -1

    def test_too_few_items_returns_none(self):
        raw = '{"IN-1": 2, "IN-2": 3}'
        scores = parse_caars_scores(raw)
        assert scores is None

    def test_invalid_json_returns_none(self):
        raw = "This is not JSON at all"
        scores = parse_caars_scores(raw)
        assert scores is None

    def test_empty_string(self):
        scores = parse_caars_scores("")
        assert scores is None

    def test_string_values_converted(self):
        items = {f"IN-{i}": str(i % 4) for i in range(1, 5)}
        items.update({f"HY-{i}": str(i % 4) for i in range(1, 5)})
        items.update({f"IM-{i}": str(i % 4) for i in range(1, 5)})
        raw = json.dumps(items)
        scores = parse_caars_scores(raw)
        assert scores is not None


class TestComputeTotalScore:
    def test_basic(self):
        scores = {"IN-1": 3, "IN-2": 2, "HY-1": 1}
        assert compute_total_score(scores) == 6

    def test_empty(self):
        assert compute_total_score({}) == 0


class TestExtractSelfReportScore:
    def test_valid_checkpoint(self):
        checkpoint = {
            "self_report": {
                "raw_response": '{"IN-1": 3, "IN-2": 2, "IN-3": 1, "IN-4": 0, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 3, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}',
            },
        }
        result = extract_self_report_score(checkpoint)
        assert result is not None
        assert result["total_score"] == 23
        assert len(result["items"]) == 12

    def test_missing_self_report(self):
        assert extract_self_report_score({}) is None

    def test_empty_raw_response(self):
        checkpoint = {"self_report": {"raw_response": ""}}
        assert extract_self_report_score(checkpoint) is None

    def test_unparseable_response(self):
        checkpoint = {"self_report": {"raw_response": "I cannot answer that."}}
        assert extract_self_report_score(checkpoint) is None


class TestRunObserverAssessment:
    def test_collects_ratings(self):
        client = MagicMock()
        valid_response = '{"IN-1": 2, "IN-2": 3, "IN-3": 1, "IN-4": 2, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 0, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}'
        client.chat.return_value = CompletionResult(
            content=valid_response,
            usage=UsageInfo(cost_usd=0.01),
            model="observer",
            finish_reason="stop",
        )

        turns = [
            {"turn": 1, "role": "participant", "content": "I got distracted again."},
            {"turn": 2, "role": "partner", "content": "How so?"},
        ]

        result = run_observer_assessment(client, turns, up_to_turn=6, n_calls=3)
        assert len(result["observer_ratings"]) == 3
        assert result["observer_mean"] > 0
        assert result["observer_cost"]["n_calls"] == 3
        assert client.chat.call_count == 3

    def test_handles_parse_errors(self):
        client = MagicMock()
        client.chat.return_value = CompletionResult(
            content="I cannot rate this.",
            usage=UsageInfo(cost_usd=0.01),
            model="observer",
            finish_reason="stop",
        )

        turns = [{"turn": 1, "role": "participant", "content": "Hello"}]
        result = run_observer_assessment(client, turns, up_to_turn=6, n_calls=2)

        assert len(result["observer_ratings"]) == 2
        assert result["observer_mean"] == 0.0

    def test_filters_turns_by_role(self):
        client = MagicMock()
        valid_response = '{"IN-1": 2, "IN-2": 2, "IN-3": 2, "IN-4": 2, "HY-1": 2, "HY-2": 2, "HY-3": 2, "HY-4": 2, "IM-1": 2, "IM-2": 2, "IM-3": 2, "IM-4": 2}'
        client.chat.return_value = CompletionResult(
            content=valid_response,
            usage=UsageInfo(cost_usd=0.01),
            model="observer",
            finish_reason="stop",
        )

        turns = [
            {"turn": 0, "role": "task", "content": "Task"},
            {"turn": 1, "role": "participant", "content": "Day starts"},
            {"turn": 2, "role": "partner", "content": "How?"},
            {"turn": 10, "role": "participant", "content": "Later turn"},
        ]

        result = run_observer_assessment(client, turns, up_to_turn=6, n_calls=1)
        assert len(result["observer_ratings"]) == 1

    def test_sends_cache_control(self):
        """Observer calls must enable prompt caching for Gemini models."""
        client = MagicMock()
        valid_response = '{"IN-1": 2, "IN-2": 2, "IN-3": 2, "IN-4": 2, "HY-1": 2, "HY-2": 2, "HY-3": 2, "HY-4": 2, "IM-1": 2, "IM-2": 2, "IM-3": 2, "IM-4": 2}'
        client.chat.return_value = CompletionResult(
            content=valid_response,
            usage=UsageInfo(cost_usd=0.01),
            model="observer",
            finish_reason="stop",
        )

        turns = [{"turn": 1, "role": "participant", "content": "Hello"}]
        run_observer_assessment(client, turns, up_to_turn=6, n_calls=1)

        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs.get("cache_control") is True

        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


class TestComputeICC:
    def test_perfect_agreement(self):
        ratings = [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ]
        icc = compute_icc(ratings)
        # Perfect agreement within tolerance (some numerical variance)
        assert icc >= 0.0

    def test_no_agreement(self):
        ratings = [
            [0, 3, 0, 3],
            [3, 0, 3, 0],
        ]
        icc = compute_icc(ratings)
        assert icc < 0.5

    def test_empty_returns_zero(self):
        assert compute_icc([]) == 0.0

    def test_single_rater_returns_zero(self):
        assert compute_icc([[1, 2, 3]]) == 0.0

    def test_single_item_returns_zero(self):
        assert compute_icc([[1], [2]]) == 0.0

    def test_moderate_agreement(self):
        ratings = [
            [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            [2, 2, 1, 2, 3, 1, 2, 3, 2, 2, 3, 1],
            [3, 3, 1, 2, 3, 0, 2, 3, 1, 2, 3, 1],
        ]
        icc = compute_icc(ratings)
        assert 0.0 <= icc <= 1.0


class TestEvaluateCheckpoint:
    @patch("src.evaluator.load_conversation")
    @patch("src.evaluator.save_observer_scores")
    @patch("src.evaluator.load_checkpoint")
    def test_runs_observer_when_not_cached(self, mock_load_cp, mock_save_obs, mock_load_conv):
        mock_load_cp.return_value = {
            "self_report": {
                "raw_response": '{"IN-1": 3, "IN-2": 2, "IN-3": 1, "IN-4": 0, "HY-1": 3, "HY-2": 2, "HY-3": 1, "HY-4": 3, "IM-1": 2, "IM-2": 1, "IM-3": 3, "IM-4": 2}',
            },
        }
        mock_load_conv.return_value = [
            {"turn": 1, "role": "participant", "content": "My day..."},
        ]

        client = MagicMock()
        valid_response = '{"IN-1": 2, "IN-2": 2, "IN-3": 1, "IN-4": 1, "HY-1": 2, "HY-2": 1, "HY-3": 1, "HY-4": 1, "IM-1": 1, "IM-2": 1, "IM-3": 2, "IM-4": 1}'
        client.chat.return_value = CompletionResult(
            content=valid_response,
            usage=UsageInfo(cost_usd=0.01),
            model="observer",
            finish_reason="stop",
        )

        from src.config import ModelConfig
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = evaluate_checkpoint(client, cfg, 1, "conv001", 6)
        assert result["turn"] == 6
        assert result["self_report"] is not None
        assert result["observer_mean"] > 0
        mock_save_obs.assert_called_once()

    @patch("src.evaluator.load_checkpoint")
    def test_missing_checkpoint(self, mock_load_cp):
        mock_load_cp.return_value = None
        client = MagicMock()
        from src.config import ModelConfig
        cfg = ModelConfig(model_id="test/model")
        result = evaluate_checkpoint(client, cfg, 1, "conv001", 6)
        assert "error" in result
