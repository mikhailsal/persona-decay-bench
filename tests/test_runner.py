"""Tests for runner.py: conversation flow, checkpoint triggering, turn counting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import CHECKPOINT_TURNS, ModelConfig
from src.openrouter_client import CompletionResult, UsageInfo
from src.runner import (
    _build_partner_messages,
    _build_target_messages,
    _collect_self_report,
    _generate_conversation_id,
    run_conversation,
)


def _make_result(content: str = "Response", cost: float = 0.001) -> CompletionResult:
    return CompletionResult(
        content=content,
        usage=UsageInfo(prompt_tokens=10, completion_tokens=20, cost_usd=cost, elapsed_seconds=0.5),
        model="test/model",
        finish_reason="stop",
    )


class TestGenerateConversationId:
    def test_length(self):
        cid = _generate_conversation_id()
        assert len(cid) == 12

    def test_unique(self):
        ids = {_generate_conversation_id() for _ in range(100)}
        assert len(ids) == 100


class TestBuildTargetMessages:
    def test_system_prompt_first(self):
        turns = [{"role": "task", "content": "Describe your workday."}]
        messages = _build_target_messages(turns)
        assert messages[0]["role"] == "system"
        assert "ADHD" in messages[0]["content"]

    def test_task_as_user(self):
        turns = [{"role": "task", "content": "Describe your workday."}]
        messages = _build_target_messages(turns)
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Describe your workday."

    def test_participant_as_assistant(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "My day starts..."},
        ]
        messages = _build_target_messages(turns)
        assert messages[2]["role"] == "assistant"

    def test_partner_as_user(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "My day starts..."},
            {"role": "partner", "content": "How do you handle..."},
        ]
        messages = _build_target_messages(turns)
        assert messages[3]["role"] == "user"


class TestBuildPartnerMessages:
    def test_system_prompt(self):
        turns = [{"role": "task", "content": "Task"}]
        messages = _build_partner_messages(turns)
        assert messages[0]["role"] == "system"
        assert "neutral" in messages[0]["content"].lower()

    def test_participant_as_user(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "My response"},
        ]
        messages = _build_partner_messages(turns)
        assert messages[2]["role"] == "user"

    def test_partner_as_assistant(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "Response"},
            {"role": "partner", "content": "Question"},
        ]
        messages = _build_partner_messages(turns)
        assert messages[3]["role"] == "assistant"


class TestCollectSelfReport:
    def test_calls_api(self):
        client = MagicMock()
        client.chat.return_value = _make_result('{"IN-1": 3, "IN-2": 2}')

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "Response"},
        ]

        result = _collect_self_report(client, cfg, turns, 6)
        assert "raw_response" in result
        assert "cost" in result
        assert result["cost"]["cost_usd"] == 0.001
        client.chat.assert_called_once()


class TestRunConversation:
    @patch("src.runner.append_turn")
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.conversation_exists", return_value=False)
    def test_basic_flow(self, mock_exists, mock_save_cp, mock_append):
        client = MagicMock()
        client.chat.return_value = _make_result("Model response")

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="test-conv",
            max_turns=6,
            checkpoint_turns=[6],
        )

        assert result["conversation_id"] == "test-conv"
        assert result["status"] == "completed"
        assert len(result["turns"]) > 0
        assert mock_append.call_count > 0

    @patch("src.runner.load_conversation")
    @patch("src.runner.conversation_exists", return_value=True)
    def test_skips_cached_conversation(self, mock_exists, mock_load):
        # expected_messages = 2 + 2 * max_turns; default max_turns=36 => 74
        mock_load.return_value = [{"turn": i} for i in range(74)]

        client = MagicMock()
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="cached-conv",
        )

        assert result["status"] == "cached"
        client.chat.assert_not_called()

    @patch("src.runner.append_turn")
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.conversation_exists", return_value=False)
    def test_collects_self_report_at_checkpoints(self, mock_exists, mock_save_cp, mock_append):
        call_count = 0
        def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_result(f"Response {call_count}")

        client = MagicMock()
        client.chat.side_effect = mock_chat

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="cp-conv",
            max_turns=6,
            checkpoint_turns=[6],
        )

        assert mock_save_cp.call_count >= 1
        cp_call = mock_save_cp.call_args
        assert cp_call[0][3] == 6  # exchange round number

    @patch("src.runner.append_turn")
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.conversation_exists", return_value=False)
    def test_turn_counting(self, mock_exists, mock_save_cp, mock_append):
        client = MagicMock()
        client.chat.return_value = _make_result("Response")

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="count-conv",
            max_turns=4,
            checkpoint_turns=[],
        )

        turn_numbers = [t["turn"] for t in result["turns"]]
        assert 0 in turn_numbers  # task
        assert 1 in turn_numbers  # first participant
        # 4 exchanges: task(0) + init(1) + 4*(partner+participant) = 10 messages
        assert len(result["turns"]) == 2 + 2 * 4
        exchanges = [t.get("exchange") for t in result["turns"] if t.get("exchange") is not None]
        assert max(exchanges) == 4
