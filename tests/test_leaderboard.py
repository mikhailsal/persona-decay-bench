"""Tests for leaderboard.py: table display, Markdown report, JSON export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.leaderboard import (
    _fmt_score,
    _score_color,
    display_leaderboard,
    export_json,
    generate_markdown_report,
)
from src.scorer import DecayCurve, DimensionScores, ModelScore, MultiRunStats


def _make_model_score(
    model_id: str = "test/model",
    stability_index: float = 85.0,
    n_conversations: int = 5,
    **dim_kwargs,
) -> ModelScore:
    dims = DimensionScores(
        initial_expression=dim_kwargs.get("init", 9.0),
        decay_resistance=dim_kwargs.get("decay", 8.0),
        self_report_consistency=dim_kwargs.get("sr_con", 7.0),
        observer_self_agreement=dim_kwargs.get("gap", 6.0),
        extended_stability=dim_kwargs.get("ext", 7.5),
    )
    curve = DecayCurve(
        turns=[6, 12, 18],
        observer_means=[17.5, 15.0, 12.5],
        self_report_scores=[25.0, 24.0, 23.0],
    )
    return ModelScore(
        model_id=model_id,
        stability_index=stability_index,
        dimensions=dims,
        decay_curve=curve,
        n_conversations=n_conversations,
    )


class TestScoreColor:
    def test_high_score(self):
        assert "green" in _score_color(90.0)

    def test_medium_score(self):
        color = _score_color(50.0)
        assert "yellow" in color or "green" in color

    def test_low_score(self):
        assert "red" in _score_color(10.0)

    def test_zero(self):
        assert "red" in _score_color(0.0)


class TestFmtScore:
    def test_formats_value(self):
        text = _fmt_score(8.5)
        assert "8.5" in text.plain

    def test_none_shows_dash(self):
        text = _fmt_score(None)
        assert "—" in text.plain


class TestDisplayLeaderboard:
    def test_displays_without_error(self, capsys):
        scores = [
            _make_model_score("model-a", 92.0),
            _make_model_score("model-b", 85.0),
        ]
        display_leaderboard(scores)

    def test_empty_scores(self, capsys):
        display_leaderboard([])

    def test_detailed_mode(self, capsys):
        scores = [_make_model_score("model-a", 90.0)]
        display_leaderboard(scores, detailed=True)

    def test_with_multi_run(self, capsys):
        ms = _make_model_score("model-a", 85.0)
        ms.multi_run = MultiRunStats(
            n_runs=3,
            per_run_indices=[80, 85, 90],
            mean_index=85.0,
            ci_low=79.0,
            ci_high=91.0,
        )
        display_leaderboard([ms])

    def test_sorts_by_index(self, capsys):
        scores = [
            _make_model_score("low-model", 60.0),
            _make_model_score("high-model", 95.0),
        ]
        display_leaderboard(scores)


class TestGenerateMarkdownReport:
    def test_generates_file(self, tmp_path):
        scores = [
            _make_model_score("model-a", 92.0),
            _make_model_score("model-b", 85.0),
        ]
        output = tmp_path / "LEADERBOARD.md"
        md = generate_markdown_report(scores, output_path=output)
        assert output.exists()
        assert "Persona Decay Benchmark" in md
        assert "model-a" in md
        assert "model-b" in md

    def test_includes_rankings_table(self, tmp_path):
        scores = [_make_model_score("model-a", 90.0)]
        md = generate_markdown_report(scores, output_path=tmp_path / "lb.md")
        assert "| # |" in md
        assert "| 1 |" in md

    def test_includes_decay_curves(self, tmp_path):
        scores = [_make_model_score("model-a", 90.0)]
        md = generate_markdown_report(scores, output_path=tmp_path / "lb.md")
        assert "Decay Curves" in md
        assert "T6=" in md

    def test_includes_methodology(self, tmp_path):
        scores = [_make_model_score("model-a", 90.0)]
        md = generate_markdown_report(scores, output_path=tmp_path / "lb.md")
        assert "Methodology" in md
        assert "ADHD" in md

    def test_with_ci(self, tmp_path):
        ms = _make_model_score("model-a", 85.0)
        ms.multi_run = MultiRunStats(
            n_runs=3,
            per_run_indices=[80, 85, 90],
            mean_index=85.0,
            ci_low=79.0,
            ci_high=91.0,
        )
        md = generate_markdown_report([ms], output_path=tmp_path / "lb.md")
        assert "95% CI" in md

    def test_empty_scores(self, tmp_path):
        md = generate_markdown_report([], output_path=tmp_path / "lb.md")
        assert "Persona Decay Benchmark" in md


class TestExportJson:
    def test_creates_file(self, tmp_path):
        scores = [_make_model_score("model-a", 92.0)]
        output = tmp_path / "leaderboard.json"
        data = export_json(scores, output_path=output)
        assert output.exists()

        loaded = json.loads(output.read_text())
        assert loaded["benchmark"] == "persona-decay-bench"
        assert len(loaded["models"]) == 1

    def test_includes_decay_curves(self, tmp_path):
        scores = [_make_model_score("model-a", 92.0)]
        data = export_json(scores, output_path=tmp_path / "lb.json")
        assert "model-a" in data["decay_curves"]
        assert data["decay_curves"]["model-a"]["turns"] == [6, 12, 18]

    def test_includes_scores(self, tmp_path):
        scores = [_make_model_score("model-a", 92.0)]
        data = export_json(scores, output_path=tmp_path / "lb.json")
        assert data["scores"][0]["stability_index"] == 92.0

    def test_empty_scores(self, tmp_path):
        data = export_json([], output_path=tmp_path / "lb.json")
        assert data["models"] == []

    def test_sorted_by_index(self, tmp_path):
        scores = [
            _make_model_score("low-model", 60.0),
            _make_model_score("high-model", 95.0),
        ]
        data = export_json(scores, output_path=tmp_path / "lb.json")
        assert data["models"][0] == "high-model"
