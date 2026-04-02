"""Tests for prompts.py: persona prompts, CAARS items, partner prompt."""

from __future__ import annotations

from src.prompts import (
    CAARS_ITEMS,
    HIGH_ADHD_PERSONA,
    LIKERT_SCALE,
    MAX_CAARS_SCORE,
    PARTNER_SYSTEM_PROMPT,
    WORKDAY_TASK,
    build_observer_prompt,
    build_self_report_prompt,
    format_conversation_for_observer,
)


class TestPersonaPrompts:
    def test_high_adhd_persona_not_empty(self):
        assert len(HIGH_ADHD_PERSONA) > 100

    def test_high_adhd_mentions_adhd(self):
        assert "ADHD" in HIGH_ADHD_PERSONA

    def test_partner_prompt_neutral(self):
        assert "neutral" in PARTNER_SYSTEM_PROMPT.lower()

    def test_partner_prompt_no_opinions(self):
        assert "opinions" in PARTNER_SYSTEM_PROMPT.lower()

    def test_workday_task_mentions_workday(self):
        assert "workday" in WORKDAY_TASK.lower()


class TestCaarsItems:
    def test_12_items(self):
        assert len(CAARS_ITEMS) == 12

    def test_all_have_ids(self):
        ids = {item.id for item in CAARS_ITEMS}
        assert len(ids) == 12

    def test_three_dimensions(self):
        dims = {item.dimension for item in CAARS_ITEMS}
        assert dims == {"inattention", "hyperactivity", "impulsivity"}

    def test_four_per_dimension(self):
        for dim in ("inattention", "hyperactivity", "impulsivity"):
            items = [i for i in CAARS_ITEMS if i.dimension == dim]
            assert len(items) == 4, f"Expected 4 items for {dim}, got {len(items)}"

    def test_self_report_text_not_empty(self):
        for item in CAARS_ITEMS:
            assert len(item.self_report_text) > 10

    def test_observer_text_not_empty(self):
        for item in CAARS_ITEMS:
            assert len(item.observer_text) > 10

    def test_observer_uses_third_person(self):
        for item in CAARS_ITEMS:
            assert "This person" in item.observer_text or "this person" in item.observer_text

    def test_self_report_uses_first_person(self):
        for item in CAARS_ITEMS:
            assert "I " in item.self_report_text or "I'" in item.self_report_text

    def test_max_score_is_36(self):
        assert MAX_CAARS_SCORE == 36


class TestLikertScale:
    def test_four_levels(self):
        assert len(LIKERT_SCALE) == 4

    def test_range_0_to_3(self):
        assert set(LIKERT_SCALE.keys()) == {0, 1, 2, 3}


class TestBuildSelfReportPrompt:
    def test_includes_all_item_ids(self):
        prompt = build_self_report_prompt()
        for item in CAARS_ITEMS:
            assert item.id in prompt

    def test_mentions_json(self):
        prompt = build_self_report_prompt()
        assert "JSON" in prompt

    def test_includes_scale(self):
        prompt = build_self_report_prompt()
        assert "0 =" in prompt
        assert "3 =" in prompt

    def test_includes_example(self):
        prompt = build_self_report_prompt()
        assert '"IN-1"' in prompt


class TestBuildObserverPrompt:
    def test_includes_conversation(self):
        prompt = build_observer_prompt("Hello world")
        assert "Hello world" in prompt

    def test_includes_all_observer_items(self):
        prompt = build_observer_prompt("test")
        for item in CAARS_ITEMS:
            assert item.id in prompt

    def test_mentions_observable_behavior(self):
        prompt = build_observer_prompt("test")
        assert "observable" in prompt.lower()

    def test_mentions_json(self):
        prompt = build_observer_prompt("test")
        assert "JSON" in prompt


class TestFormatConversation:
    def test_labels_correctly(self):
        turns = [
            {"role": "participant", "content": "I had a busy day."},
            {"role": "partner", "content": "Tell me more."},
        ]
        result = format_conversation_for_observer(turns)
        assert "[Participant]:" in result
        assert "[Partner]:" in result

    def test_includes_content(self):
        turns = [
            {"role": "participant", "content": "I struggle with focus."},
        ]
        result = format_conversation_for_observer(turns)
        assert "I struggle with focus." in result

    def test_empty_turns(self):
        result = format_conversation_for_observer([])
        assert result == ""
