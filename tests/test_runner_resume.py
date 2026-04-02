"""Tests for runner resume-from-cache functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import ModelConfig
from src.openrouter_client import CompletionResult, UsageInfo
from src.runner import run_conversation


def _make_result(content: str = "Response", cost: float = 0.001) -> CompletionResult:
    return CompletionResult(
        content=content,
        usage=UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=cost,
            cached_tokens=0,
            cache_write_tokens=0,
        ),
    )


class TestResumeExchange:
    def test_empty(self):
        from src.runner import _resume_exchange

        assert _resume_exchange(0) == 1
        assert _resume_exchange(1) == 1

    def test_after_init(self):
        from src.runner import _resume_exchange

        assert _resume_exchange(2) == 1

    def test_mid_conversation(self):
        from src.runner import _resume_exchange

        assert _resume_exchange(10) == 5
        assert _resume_exchange(74) == 37


class TestFindExistingConversation:
    @patch("src.runner.list_conversations", return_value=[])
    def test_no_conversations(self, _mock):
        from src.runner import _find_existing_conversation

        cfg = ModelConfig(model_id="test/m")
        assert _find_existing_conversation("dir", 1, cfg, 36) is None

    @patch("src.runner.load_conversation")
    @patch("src.runner.list_conversations", return_value=["abc123"])
    def test_complete(self, _mock_list, mock_load):
        from src.runner import _find_existing_conversation

        mock_load.return_value = [{"turn": i} for i in range(74)]
        cfg = ModelConfig(model_id="test/m")
        result = _find_existing_conversation("dir", 1, cfg, 36)
        assert result is not None
        assert result["status"] == "cached"
        assert result["conversation_id"] == "abc123"

    @patch("src.runner.load_conversation")
    @patch("src.runner.list_conversations", return_value=["abc123"])
    def test_partial(self, _mock_list, mock_load):
        from src.runner import _find_existing_conversation

        mock_load.return_value = [{"turn": i} for i in range(20)]
        cfg = ModelConfig(model_id="test/m")
        result = _find_existing_conversation("dir", 1, cfg, 36)
        assert result is not None
        assert result["status"] == "partial"

    @patch("src.runner.load_conversation")
    @patch("src.runner.list_conversations", return_value=["abc123"])
    def test_empty_conv(self, _mock_list, mock_load):
        from src.runner import _find_existing_conversation

        mock_load.return_value = []
        cfg = ModelConfig(model_id="test/m")
        assert _find_existing_conversation("dir", 1, cfg, 36) is None


class TestMaybeCollectCheckpoint:
    @patch("src.runner.checkpoint_exists", return_value=True)
    def test_already_cached(self, _mock):
        from src.runner import _maybe_collect_checkpoint

        client = MagicMock()
        cfg = ModelConfig(model_id="test/m")
        cost = _maybe_collect_checkpoint(client, cfg, [], "d", 1, "c", 6)
        assert cost == 0.0
        client.chat.assert_not_called()

    @patch("src.runner.save_checkpoint")
    @patch("src.runner.checkpoint_exists", return_value=False)
    def test_new_checkpoint(self, _mock_exists, _mock_save):
        from src.runner import _maybe_collect_checkpoint

        client = MagicMock()
        client.chat.return_value = _make_result("SR response")
        cfg = ModelConfig(model_id="test/m", temperature=0.7, reasoning_effort="none")
        cost = _maybe_collect_checkpoint(client, cfg, [], "d", 1, "c", 6)
        assert cost > 0
        _mock_save.assert_called_once()


class TestConversationResume:
    @patch("src.runner.append_turn")
    @patch("src.runner.checkpoint_exists", return_value=False)
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.load_conversation")
    @patch("src.runner.conversation_exists", return_value=True)
    def test_resumes_partial(self, _mock_exists, mock_load, _mock_save_cp, _mock_cp_exists, _mock_append):
        mock_load.return_value = [
            {"turn": 0, "role": "task", "content": "task"},
            {"turn": 1, "role": "participant", "content": "init"},
            {"turn": 2, "role": "partner", "content": "p1"},
            {"turn": 3, "role": "participant", "content": "a1"},
        ]
        client = MagicMock()
        client.chat.return_value = _make_result("Resumed")
        cfg = ModelConfig(model_id="test/m", temperature=0.7, reasoning_effort="none")
        result = run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="partial-conv",
            max_turns=3,
            checkpoint_turns=[],
        )
        assert result["status"] == "completed"


class TestRunAllConversationsResume:
    @patch("src.runner._find_existing_conversation")
    @patch("src.runner.run_conversation")
    def test_skips_complete(self, mock_run_conv, mock_find):
        from src.runner import run_all_conversations

        cfg = ModelConfig(model_id="test/m")
        mock_find.return_value = {
            "conversation_id": "done123",
            "model_config": cfg,
            "run": 1,
            "turns": [{"t": i} for i in range(74)],
            "status": "cached",
        }
        client = MagicMock()
        results = run_all_conversations(client=client, model_config=cfg, n_runs=1, max_turns=36)
        assert len(results) == 1
        assert results[0]["status"] == "cached"
        mock_run_conv.assert_not_called()

    @patch("src.runner._find_existing_conversation")
    @patch("src.runner.run_conversation")
    def test_resumes_partial(self, mock_run_conv, mock_find):
        from src.runner import run_all_conversations

        cfg = ModelConfig(model_id="test/m")
        mock_find.return_value = {
            "conversation_id": "part456",
            "model_config": cfg,
            "run": 1,
            "turns": [{"t": i} for i in range(20)],
            "status": "partial",
        }
        mock_run_conv.return_value = {"status": "completed", "conversation_id": "part456"}
        client = MagicMock()
        results = run_all_conversations(client=client, model_config=cfg, n_runs=1, max_turns=36)
        assert len(results) == 1
        mock_run_conv.assert_called_once()
        assert mock_run_conv.call_args.kwargs.get("conversation_id") == "part456"
