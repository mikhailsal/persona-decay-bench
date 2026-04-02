"""Tests for runner.py: conversation flow, checkpoint triggering, turn counting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config import ModelConfig
from src.openrouter_client import CompletionResult, UsageInfo
from src.runner import (
    _generate_conversation_id,
    run_conversation,
)
from src.runner_helpers import (
    build_partner_messages,
    build_target_messages,
    collect_self_report,
    inject_explicit_cache_breakpoint,
    print_turn_with_cache,
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
        messages = build_target_messages(turns)
        assert messages[0]["role"] == "system"
        assert "ADHD" in messages[0]["content"]

    def test_task_as_user(self):
        turns = [{"role": "task", "content": "Describe your workday."}]
        messages = build_target_messages(turns)
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Describe your workday."

    def test_participant_as_assistant(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "My day starts..."},
        ]
        messages = build_target_messages(turns)
        assert messages[2]["role"] == "assistant"

    def test_partner_as_user(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "My day starts..."},
            {"role": "partner", "content": "How do you handle..."},
        ]
        messages = build_target_messages(turns)
        assert messages[3]["role"] == "user"


class TestBuildPartnerMessages:
    def test_system_prompt(self):
        turns = [{"role": "task", "content": "Task"}]
        messages = build_partner_messages(turns)
        assert messages[0]["role"] == "system"
        assert "neutral" in messages[0]["content"].lower()

    def test_participant_as_user(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "My response"},
        ]
        messages = build_partner_messages(turns)
        assert messages[2]["role"] == "user"

    def test_partner_as_assistant(self):
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "Response"},
            {"role": "partner", "content": "Question"},
        ]
        messages = build_partner_messages(turns)
        assert messages[3]["role"] == "assistant"


class TestMessageFormatAndCaching:
    """Verify messages use plain string content (no array conversion) so that
    automatic top-level caching produces stable prefix hashes across turns."""

    def _sample_turns(self):
        return [
            {"role": "task", "content": "Describe your workday."},
            {"role": "participant", "content": "I start by checking emails..."},
            {"role": "partner", "content": "What happens next?"},
            {"role": "participant", "content": "Then I get distracted..."},
        ]

    def test_target_messages_use_plain_strings(self):
        turns = self._sample_turns()
        messages = build_target_messages(turns)
        for msg in messages:
            assert isinstance(
                msg["content"], str
            ), f"Message content should be plain string, got {type(msg['content'])}"

    def test_partner_messages_use_plain_strings(self):
        turns = self._sample_turns()
        messages = build_partner_messages(turns)
        for msg in messages:
            assert isinstance(
                msg["content"], str
            ), f"Message content should be plain string, got {type(msg['content'])}"

    def test_prefix_stays_identical_across_turns(self):
        """Messages built from a prefix of turns must be byte-identical to
        the corresponding prefix in a later request with more turns."""
        turns_short = self._sample_turns()[:3]
        turns_long = self._sample_turns()

        msgs_short = build_target_messages(turns_short)
        msgs_long = build_target_messages(turns_long)

        for i, (a, b) in enumerate(zip(msgs_short, msgs_long, strict=False)):
            assert a == b, f"Message {i} differs between short and long builds"

    def test_reasoning_details_preserved(self):
        turns = [
            {"role": "task", "content": "Task"},
            {
                "role": "participant",
                "content": "Response with reasoning",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "abc123"}],
            },
        ]
        messages = build_target_messages(turns)
        assistant_msg = messages[2]
        assert "reasoning_details" in assistant_msg
        assert assistant_msg["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "abc123"}]
        assert isinstance(assistant_msg["content"], str)

    @patch("src.runner.append_turn")
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.conversation_exists", return_value=False)
    def test_grok41_run_passes_reasoning_and_cache_control(self, mock_exists, mock_save_cp, mock_append):
        client = MagicMock()
        client.chat.return_value = _make_result("Response")

        cfg = ModelConfig(model_id="x-ai/grok-4.1-fast", temperature=0.7, reasoning_effort="low")

        run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="grok-test",
            max_turns=2,
            checkpoint_turns=[],
        )

        target_calls = [call for call in client.chat.call_args_list if call.kwargs.get("model") == "x-ai/grok-4.1-fast"]
        assert len(target_calls) >= 1
        for call in target_calls:
            assert call.kwargs.get("reasoning_effort") == "low"
            assert call.kwargs.get("cache_control") is True

    @patch("src.runner.append_turn")
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.conversation_exists", return_value=False)
    def test_all_api_calls_enable_cache_control(self, mock_exists, mock_save_cp, mock_append):
        """Every API call in run_conversation must pass cache_control=True."""
        client = MagicMock()
        client.chat.return_value = _make_result("Response")

        cfg = ModelConfig(model_id="x-ai/grok-4.1-fast", temperature=0.7, reasoning_effort="low")

        run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="cache-test",
            max_turns=3,
            checkpoint_turns=[],
        )

        for i, call in enumerate(client.chat.call_args_list):
            assert (
                call.kwargs.get("cache_control") is True
            ), f"API call {i} (model={call.kwargs.get('model')}) missing cache_control=True"


class TestInjectExplicitCacheBreakpoint:
    """Verify inject_explicit_cache_breakpoint correctly marks the last message."""

    def test_converts_last_message_to_array_with_cache_control(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        inject_explicit_cache_breakpoint(messages)
        last = messages[-1]
        assert isinstance(last["content"], list)
        assert len(last["content"]) == 1
        assert last["content"][0]["type"] == "text"
        assert last["content"][0]["text"] == "Hi there"
        assert last["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_does_not_modify_earlier_messages(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        inject_explicit_cache_breakpoint(messages)
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)

    def test_handles_empty_list(self):
        messages: list[dict] = []
        inject_explicit_cache_breakpoint(messages)
        assert messages == []

    def test_single_message(self):
        messages = [{"role": "system", "content": "System only"}]
        inject_explicit_cache_breakpoint(messages)
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["text"] == "System only"

    def test_idempotent_on_already_converted(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Already array", "cache_control": {"type": "ephemeral"}}],
            },
        ]
        inject_explicit_cache_breakpoint(messages)
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 1


class TestCollectSelfReport:
    def test_calls_api(self):
        client = MagicMock()
        client.chat.return_value = _make_result('{"IN-1": 3, "IN-2": 2}')

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        turns = [
            {"role": "task", "content": "Task"},
            {"role": "participant", "content": "Response"},
        ]

        result = collect_self_report(client, cfg, turns, 6)
        assert "raw_response" in result
        assert "cost" in result
        assert result["cost"]["cost_usd"] == 0.001
        client.chat.assert_called_once()

    def test_self_report_messages_have_cache_breakpoint_before_questionnaire(self):
        """The self-report request must have an explicit cache breakpoint on the
        last conversation message (right before the questionnaire) so the prefix
        can be served from cache."""
        client = MagicMock()
        client.chat.return_value = _make_result('{"IN-1": 3}')

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        turns = [
            {"role": "task", "content": "Describe your workday."},
            {"role": "participant", "content": "I start by checking emails..."},
            {"role": "partner", "content": "What happens next?"},
            {"role": "participant", "content": "Then I get distracted..."},
        ]

        collect_self_report(client, cfg, turns, 6)

        call_kwargs = client.chat.call_args.kwargs
        messages = call_kwargs["messages"]

        last_conv_msg = messages[-2]
        assert isinstance(
            last_conv_msg["content"], list
        ), "Second-to-last message (last conversation msg) should be array with cache_control"
        assert last_conv_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

        questionnaire_msg = messages[-1]
        assert isinstance(questionnaire_msg["content"], str), "Last message (questionnaire) should be plain string"


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

        run_conversation(
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


class TestPrintTurnVerbose:
    def test_verbose_false_truncates(self, capsys):
        content = "A" * 200
        print_turn_with_cache(
            "Turn  1",
            "participant",
            content,
            100,
            0,
            0,
            verbose=False,
        )
        captured = capsys.readouterr().out
        assert "..." in captured
        assert "A" * 200 not in captured

    def test_verbose_true_shows_full_content(self, capsys):
        content = "A" * 200
        print_turn_with_cache(
            "Turn  1",
            "participant",
            content,
            100,
            0,
            0,
            verbose=True,
        )
        captured = capsys.readouterr().out
        assert "participant" in captured.lower()
        assert "..." not in captured.split("\n")[0]

    def test_verbose_short_content_no_panel(self, capsys):
        content = "Short reply"
        print_turn_with_cache(
            "Turn  1",
            "participant",
            content,
            50,
            0,
            0,
            verbose=True,
        )
        captured = capsys.readouterr().out
        assert "Short reply" in captured

    @patch("src.runner.append_turn")
    @patch("src.runner.save_checkpoint")
    @patch("src.runner.conversation_exists", return_value=False)
    def test_run_conversation_verbose_param(self, mock_exists, mock_save_cp, mock_append):
        client = MagicMock()
        client.chat.return_value = _make_result("Model response")

        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")

        result = run_conversation(
            client=client,
            model_config=cfg,
            run_number=1,
            conversation_id="verb-conv",
            max_turns=2,
            checkpoint_turns=[],
            verbose=True,
        )
        assert result["status"] == "completed"
