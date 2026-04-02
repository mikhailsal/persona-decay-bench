"""Tests for cache.py: JSONL conversations, checkpoint files, cache paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cache import (
    append_turn,
    checkpoint_exists,
    clear_all_cache,
    clear_observer_scores,
    conversation_exists,
    list_all_cached_models,
    list_available_runs,
    list_checkpoints,
    list_conversations,
    load_checkpoint,
    load_conversation,
    save_checkpoint,
    save_conversation,
    save_observer_scores,
    sum_run_total_cost_usd,
)


@pytest.fixture
def cache_dir(tmp_path):
    with patch("src.cache.CACHE_DIR", tmp_path):
        yield tmp_path


class TestConversationJSONL:
    def test_append_and_load(self, cache_dir):
        turn1 = {"turn": 1, "role": "participant", "content": "Hello"}
        turn2 = {"turn": 2, "role": "partner", "content": "Hi there"}
        append_turn("model@low-t0.7", 1, "conv001", turn1)
        append_turn("model@low-t0.7", 1, "conv001", turn2)

        turns = load_conversation("model@low-t0.7", 1, "conv001")
        assert len(turns) == 2
        assert turns[0]["content"] == "Hello"
        assert turns[1]["content"] == "Hi there"

    def test_save_conversation_overwrites(self, cache_dir):
        turns = [
            {"turn": 1, "role": "participant", "content": "First"},
            {"turn": 2, "role": "partner", "content": "Second"},
        ]
        save_conversation("model@low-t0.7", 1, "conv002", turns)

        loaded = load_conversation("model@low-t0.7", 1, "conv002")
        assert len(loaded) == 2

        new_turns = [{"turn": 1, "role": "participant", "content": "Replaced"}]
        save_conversation("model@low-t0.7", 1, "conv002", new_turns)

        loaded2 = load_conversation("model@low-t0.7", 1, "conv002")
        assert len(loaded2) == 1
        assert loaded2[0]["content"] == "Replaced"

    def test_load_nonexistent_returns_empty(self, cache_dir):
        turns = load_conversation("model@low-t0.7", 99, "nonexistent")
        assert turns == []

    def test_conversation_exists(self, cache_dir):
        assert not conversation_exists("model@low-t0.7", 1, "conv003")
        append_turn("model@low-t0.7", 1, "conv003", {"turn": 1, "content": "hi"})
        assert conversation_exists("model@low-t0.7", 1, "conv003")


class TestCheckpoints:
    def test_save_and_load(self, cache_dir):
        data = {"self_report": {"total_score": 24}, "observer_mean": 18.0}
        save_checkpoint("model@low-t0.7", 1, "conv001", 6, data)

        loaded = load_checkpoint("model@low-t0.7", 1, "conv001", 6)
        assert loaded is not None
        assert loaded["self_report"]["total_score"] == 24
        assert loaded["metadata"]["turn"] == 6
        assert "timestamp" in loaded["metadata"]

    def test_load_nonexistent_returns_none(self, cache_dir):
        assert load_checkpoint("model@low-t0.7", 1, "conv001", 99) is None

    def test_checkpoint_exists(self, cache_dir):
        assert not checkpoint_exists("model@low-t0.7", 1, "conv001", 6)
        save_checkpoint("model@low-t0.7", 1, "conv001", 6, {"data": True})
        assert checkpoint_exists("model@low-t0.7", 1, "conv001", 6)

    def test_list_checkpoints(self, cache_dir):
        for turn in [6, 12, 18]:
            save_checkpoint("model@low-t0.7", 1, "conv001", turn, {"turn": turn})

        checkpoints = list_checkpoints("model@low-t0.7", 1, "conv001")
        assert checkpoints == [6, 12, 18]

    def test_list_checkpoints_empty(self, cache_dir):
        assert list_checkpoints("model@low-t0.7", 1, "nonexistent") == []


class TestObserverScores:
    def test_save_observer_to_existing_checkpoint(self, cache_dir):
        save_checkpoint("model@low-t0.7", 1, "conv001", 6, {"self_report": {"total_score": 20}})

        observer_data = {
            "observer_ratings": [{"total_score": 18}],
            "observer_mean": 18.0,
            "observer_sd": 0.0,
        }
        save_observer_scores("model@low-t0.7", 1, "conv001", 6, observer_data)

        loaded = load_checkpoint("model@low-t0.7", 1, "conv001", 6)
        assert loaded["observer_mean"] == 18.0
        assert loaded["self_report"]["total_score"] == 20

    def test_save_observer_creates_checkpoint(self, cache_dir):
        observer_data = {
            "observer_ratings": [{"total_score": 15}],
            "observer_mean": 15.0,
            "observer_sd": 1.0,
        }
        save_observer_scores("model@low-t0.7", 1, "conv002", 12, observer_data)

        loaded = load_checkpoint("model@low-t0.7", 1, "conv002", 12)
        assert loaded is not None
        assert loaded["observer_mean"] == 15.0


class TestDiscovery:
    def test_list_available_runs(self, cache_dir):
        (cache_dir / "model@low-t0.7" / "run_1" / "high").mkdir(parents=True)
        (cache_dir / "model@low-t0.7" / "run_2" / "high").mkdir(parents=True)

        runs = list_available_runs("model@low-t0.7")
        assert runs == [1, 2]

    def test_list_available_runs_empty(self, cache_dir):
        assert list_available_runs("nonexistent@low-t0.7") == []

    def test_list_conversations(self, cache_dir):
        append_turn("model@low-t0.7", 1, "conv_a", {"turn": 1})
        append_turn("model@low-t0.7", 1, "conv_b", {"turn": 1})

        convs = list_conversations("model@low-t0.7", 1)
        assert sorted(convs) == ["conv_a", "conv_b"]

    def test_list_conversations_empty(self, cache_dir):
        assert list_conversations("model@low-t0.7", 99) == []

    def test_list_all_cached_models(self, cache_dir):
        (cache_dir / "modelA@low-t0.7" / "run_1").mkdir(parents=True)
        (cache_dir / "modelB@none-t0.7" / "run_1").mkdir(parents=True)

        models = list_all_cached_models()
        assert "modelA@low-t0.7" in models
        assert "modelB@none-t0.7" in models

    def test_list_all_cached_ignores_no_at(self, cache_dir):
        (cache_dir / "invalid_dir" / "run_1").mkdir(parents=True)
        assert list_all_cached_models() == []


class TestCostAggregation:
    def test_sum_run_total_cost(self, cache_dir):
        save_checkpoint("model@low-t0.7", 1, "conv001", 6, {
            "self_report_cost": {"cost_usd": 0.01},
            "observer_cost": {"cost_usd": 0.05},
        })
        save_checkpoint("model@low-t0.7", 1, "conv001", 12, {
            "self_report_cost": {"cost_usd": 0.02},
        })

        total = sum_run_total_cost_usd("model@low-t0.7", 1)
        assert abs(total - 0.08) < 1e-9

    def test_sum_run_empty(self, cache_dir):
        assert sum_run_total_cost_usd("model@low-t0.7", 99) == 0.0


class TestCacheClearing:
    def test_clear_all(self, cache_dir):
        append_turn("model@low-t0.7", 1, "conv001", {"turn": 1})
        save_checkpoint("model@low-t0.7", 1, "conv001", 6, {"data": True})

        count = clear_all_cache()
        assert count >= 2
        assert list_conversations("model@low-t0.7", 1) == []

    def test_clear_all_empty(self, cache_dir):
        assert clear_all_cache() == 0

    def test_clear_observer_scores(self, cache_dir):
        save_checkpoint("model@low-t0.7", 1, "conv001", 6, {
            "self_report": {"total_score": 20},
            "observer_ratings": [{"total_score": 18}],
            "observer_mean": 18.0,
            "observer_sd": 1.0,
        })

        count = clear_observer_scores()
        assert count == 1

        loaded = load_checkpoint("model@low-t0.7", 1, "conv001", 6)
        assert "observer_ratings" not in loaded
        assert "observer_mean" not in loaded
        assert loaded["self_report"]["total_score"] == 20

    def test_clear_observer_scores_empty(self, cache_dir):
        assert clear_observer_scores() == 0
