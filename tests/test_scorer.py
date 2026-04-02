"""Tests for scorer.py: dimension scores, composite index, multi-run statistics."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.scorer import (
    DecayCurve,
    DimensionScores,
    ModelScore,
    MultiRunStats,
    _avg_decay_curves,
    _avg_dimension_scores,
    _bootstrap_ci,
    _compute_multi_run_stats,
    _safe_avg,
    _safe_std,
    compute_dimension_scores,
    compute_stability_index,
    score_model,
)


class TestSafeAvg:
    def test_normal(self):
        assert _safe_avg([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        assert _safe_avg([]) == 0.0

    def test_single(self):
        assert _safe_avg([5.0]) == 5.0


class TestSafeStd:
    def test_normal(self):
        sd = _safe_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert sd > 0

    def test_empty(self):
        assert _safe_std([]) == 0.0

    def test_single(self):
        assert _safe_std([5.0]) == 0.0

    def test_identical(self):
        assert _safe_std([3.0, 3.0, 3.0]) == 0.0


class TestComputeDimensionScores:
    def test_all_zeros(self):
        data = [{"turn": 6, "observer_mean": 0.0, "self_report_total": 0.0}]
        dims = compute_dimension_scores(data)
        assert dims.initial_expression == 0.0

    def test_perfect_scores(self):
        data = [
            {"turn": 6, "observer_mean": 17.5, "self_report_total": 17.5},
            {"turn": 12, "observer_mean": 17.5, "self_report_total": 17.5},
            {"turn": 18, "observer_mean": 17.5, "self_report_total": 17.5},
            {"turn": 24, "observer_mean": 17.5, "self_report_total": 17.5},
            {"turn": 30, "observer_mean": 17.5, "self_report_total": 17.5},
            {"turn": 36, "observer_mean": 17.5, "self_report_total": 17.5},
        ]
        dims = compute_dimension_scores(data)
        assert dims.initial_expression == 10.0
        assert dims.decay_resistance == 10.0
        assert dims.self_report_consistency == 10.0
        assert dims.observer_self_agreement == 10.0

    def test_decay(self):
        data = [
            {"turn": 6, "observer_mean": 18.0, "self_report_total": 25.0},
            {"turn": 12, "observer_mean": 15.0, "self_report_total": 25.0},
            {"turn": 18, "observer_mean": 12.0, "self_report_total": 25.0},
        ]
        dims = compute_dimension_scores(data)
        assert dims.initial_expression > 9.0
        assert dims.decay_resistance < 10.0

    def test_empty_data(self):
        dims = compute_dimension_scores([])
        assert dims.initial_expression == 0.0

    def test_single_checkpoint(self):
        data = [{"turn": 6, "observer_mean": 15.0, "self_report_total": 20.0}]
        dims = compute_dimension_scores(data)
        assert dims.initial_expression > 0
        assert dims.decay_resistance == 5.0  # default for single point
        assert dims.self_report_consistency == 5.0

    def test_extended_stability(self):
        data = [
            {"turn": 6, "observer_mean": 17.0, "self_report_total": 17.0},
            {"turn": 12, "observer_mean": 16.0, "self_report_total": 17.0},
            {"turn": 18, "observer_mean": 15.0, "self_report_total": 17.0},
            {"turn": 24, "observer_mean": 14.5, "self_report_total": 17.0},
            {"turn": 30, "observer_mean": 14.0, "self_report_total": 17.0},
            {"turn": 36, "observer_mean": 13.5, "self_report_total": 17.0},
        ]
        dims = compute_dimension_scores(data)
        assert dims.extended_stability > 0


class TestComputeStabilityIndex:
    def test_max_score(self):
        dims = DimensionScores(
            initial_expression=10.0,
            decay_resistance=10.0,
            self_report_consistency=10.0,
            observer_self_agreement=10.0,
            extended_stability=10.0,
        )
        index = compute_stability_index(dims)
        assert index == 100.0

    def test_zero_score(self):
        dims = DimensionScores()
        index = compute_stability_index(dims)
        assert index == 0.0

    def test_partial_score(self):
        dims = DimensionScores(
            initial_expression=5.0,
            decay_resistance=5.0,
            self_report_consistency=5.0,
            observer_self_agreement=5.0,
            extended_stability=5.0,
        )
        index = compute_stability_index(dims)
        assert index == 50.0


class TestBootstrapCI:
    def test_single_value(self):
        lo, hi = _bootstrap_ci([50.0])
        assert lo == 50.0
        assert hi == 50.0

    def test_empty_values(self):
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_multiple_values(self):
        values = [80.0, 85.0, 82.0, 78.0, 81.0]
        lo, hi = _bootstrap_ci(values)
        assert lo < hi
        assert lo >= 0.0
        assert hi <= 100.0

    def test_deterministic(self):
        values = [80.0, 85.0, 82.0]
        lo1, hi1 = _bootstrap_ci(values, seed=42)
        lo2, hi2 = _bootstrap_ci(values, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2


class TestComputeMultiRunStats:
    def test_single_run(self):
        stats = _compute_multi_run_stats([85.0])
        assert stats.n_runs == 1
        assert stats.mean_index == 85.0
        assert stats.ci_low == 85.0

    def test_multiple_runs(self):
        stats = _compute_multi_run_stats([80.0, 85.0, 82.0, 78.0, 81.0])
        assert stats.n_runs == 5
        assert 78.0 <= stats.mean_index <= 85.0
        assert stats.std_dev > 0
        assert stats.ci_low < stats.ci_high

    def test_empty(self):
        stats = _compute_multi_run_stats([])
        assert stats.n_runs == 0
        assert stats.mean_index == 0.0


class TestMultiRunStats:
    def test_to_dict_single_run(self):
        stats = MultiRunStats(n_runs=1, per_run_indices=[85.0])
        d = stats.to_dict()
        assert d["n_runs"] == 1
        assert "mean_index" not in d

    def test_to_dict_multiple_runs(self):
        stats = MultiRunStats(
            n_runs=3,
            per_run_indices=[80.0, 85.0, 82.0],
            mean_index=82.3,
            std_dev=2.5,
            ci_low=79.0,
            ci_high=85.0,
        )
        d = stats.to_dict()
        assert d["mean_index"] == 82.3
        assert d["ci_method"] == "bootstrap"


class TestDecayCurve:
    def test_to_dict(self):
        curve = DecayCurve(
            turns=[6, 12, 18],
            observer_means=[17.5, 15.0, 12.5],
            self_report_scores=[25.0, 24.0, 23.0],
        )
        d = curve.to_dict()
        assert d["turns"] == [6, 12, 18]
        assert len(d["observer_means"]) == 3

    def test_empty(self):
        curve = DecayCurve()
        d = curve.to_dict()
        assert d["turns"] == []


class TestDimensionScores:
    def test_to_dict(self):
        dims = DimensionScores(initial_expression=8.5, decay_resistance=7.3)
        d = dims.to_dict()
        assert d["initial_expression"] == 8.5
        assert d["decay_resistance"] == 7.3


class TestAvgDimensionScores:
    def test_averages(self):
        dims1 = DimensionScores(initial_expression=8.0, decay_resistance=6.0)
        dims2 = DimensionScores(initial_expression=10.0, decay_resistance=8.0)
        avg = _avg_dimension_scores([dims1, dims2])
        assert avg.initial_expression == 9.0
        assert avg.decay_resistance == 7.0

    def test_single(self):
        dims = DimensionScores(initial_expression=5.0)
        avg = _avg_dimension_scores([dims])
        assert avg is dims

    def test_empty(self):
        avg = _avg_dimension_scores([])
        assert avg.initial_expression == 0.0


class TestAvgDecayCurves:
    def test_same_turns(self):
        c1 = DecayCurve(turns=[6, 12], observer_means=[18.0, 16.0], self_report_scores=[25.0, 24.0])
        c2 = DecayCurve(turns=[6, 12], observer_means=[16.0, 14.0], self_report_scores=[23.0, 22.0])
        avg = _avg_decay_curves([c1, c2])
        assert avg.turns == [6, 12]
        assert avg.observer_means[0] == 17.0  # (18+16)/2

    def test_empty(self):
        avg = _avg_decay_curves([])
        assert avg.turns == []


class TestModelScore:
    def test_to_dict(self):
        ms = ModelScore(
            model_id="test/model",
            stability_index=85.0,
            n_conversations=5,
        )
        d = ms.to_dict()
        assert d["model_id"] == "test/model"
        assert d["stability_index"] == 85.0

    def test_to_dict_with_multi_run(self):
        ms = ModelScore(
            model_id="test/model",
            stability_index=85.0,
            multi_run=MultiRunStats(n_runs=3, per_run_indices=[80, 85, 90], mean_index=85.0, std_dev=5.0, ci_low=78.0, ci_high=92.0),
        )
        d = ms.to_dict()
        assert "multi_run" in d


class TestScoreModel:
    @patch("src.scorer.list_available_runs", return_value=[])
    def test_no_runs_returns_empty(self, mock_runs):
        from src.config import ModelConfig
        cfg = ModelConfig(model_id="test/model")
        ms = score_model("test/model", config=cfg)
        assert ms.stability_index == 0.0
        assert ms.n_conversations == 0

    @patch("src.scorer.list_conversations", return_value=["conv001"])
    @patch("src.scorer.list_available_runs", return_value=[1])
    @patch("src.scorer.list_checkpoints", return_value=[6, 12])
    @patch("src.scorer.load_checkpoint")
    def test_with_data(self, mock_load_cp, mock_checkpoints, mock_runs, mock_convs):
        mock_load_cp.side_effect = [
            {
                "self_report": {"raw_response": '{"IN-1": 3, "IN-2": 2, "IN-3": 3, "IN-4": 2, "HY-1": 3, "HY-2": 2, "HY-3": 3, "HY-4": 2, "IM-1": 3, "IM-2": 2, "IM-3": 3, "IM-4": 2}'},
                "observer_mean": 17.0,
                "metadata": {"turn": 6},
            },
            {
                "self_report": {"raw_response": '{"IN-1": 3, "IN-2": 2, "IN-3": 3, "IN-4": 2, "HY-1": 3, "HY-2": 2, "HY-3": 3, "HY-4": 2, "IM-1": 3, "IM-2": 2, "IM-3": 3, "IM-4": 2}'},
                "observer_mean": 15.0,
                "metadata": {"turn": 12},
            },
        ]

        from src.config import ModelConfig
        cfg = ModelConfig(model_id="test/model", temperature=0.7, reasoning_effort="none")
        ms = score_model("test/model", config=cfg)
        assert ms.stability_index > 0
        assert ms.n_conversations == 1
        assert len(ms.decay_curve.turns) == 2
