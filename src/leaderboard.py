"""Leaderboard: Rich terminal table, Markdown report generation, JSON export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import RESULTS_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from src.scorer import ModelScore

console = Console()


def _score_color(value: float, max_val: float = 100.0) -> str:
    """Return a rich color based on score percentage."""
    pct = value / max_val if max_val > 0 else 0
    if pct >= 0.8:
        return "bold green"
    if pct >= 0.6:
        return "green"
    if pct >= 0.4:
        return "yellow"
    if pct >= 0.2:
        return "red"
    return "bold red"


def _fmt_score(value: float | None, max_val: float = 10.0) -> Text:
    """Format a score with color."""
    if value is None:
        return Text("—", style="dim")
    color = _score_color(value, max_val)
    return Text(f"{value:.1f}", style=color)


def _build_table_row(
    ms: ModelScore,
    rank: int,
    has_multi_run: bool,
    detailed: bool,
) -> list[Any]:
    """Build a single row for the leaderboard table."""
    name = ms.model_id.rsplit("@", 1)[0] if "@" in ms.model_id else ms.model_id
    index_text = Text(f"{ms.stability_index:.1f}", style=_score_color(ms.stability_index))
    row: list[Any] = [str(rank), name, index_text]

    if has_multi_run:
        if ms.multi_run.n_runs >= 2:
            row.append(Text(f"{ms.multi_run.ci_low:.1f}-{ms.multi_run.ci_high:.1f}", style="dim"))
            row.append(str(ms.multi_run.n_runs))
        else:
            row.append(Text("--", style="dim"))
            row.append("1")

    dims = ms.dimensions
    if detailed:
        for val in [
            dims.initial_expression,
            dims.decay_resistance,
            dims.self_report_consistency,
            dims.observer_self_agreement,
            dims.extended_stability,
        ]:
            row.append(_fmt_score(val))
    else:
        row.append(_fmt_score(dims.decay_resistance))
        row.append(_fmt_score(dims.initial_expression))
    return row


def display_leaderboard(model_scores: list[ModelScore], *, detailed: bool = False) -> None:
    """Display the leaderboard table in the terminal."""
    if not model_scores:
        console.print("[dim]No scores to display.[/dim]")
        return

    sorted_scores = sorted(model_scores, key=lambda s: s.stability_index, reverse=True)
    has_multi_run = any(ms.multi_run.n_runs >= 2 for ms in sorted_scores)

    table = Table(
        title="Persona Decay Benchmark",
        title_style="bold",
        show_lines=False,
        box=None,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", width=2)
    table.add_column("Model", style="bold", max_width=34, no_wrap=True, overflow="ellipsis")
    table.add_column("PSI", justify="right", width=5)
    if has_multi_run:
        table.add_column("95% CI", justify="center", width=11)
        table.add_column("N", justify="right", width=2)
    if detailed:
        for col, w in [("Init", 4), ("Decay", 5), ("SRCon", 5), ("Gap", 4), ("Ext", 4)]:
            table.add_column(col, justify="right", width=w)
    else:
        table.add_column("Decay", justify="right", width=5)
        table.add_column("Init", justify="right", width=4)

    for rank, ms in enumerate(sorted_scores, 1):
        table.add_row(*_build_table_row(ms, rank, has_multi_run, detailed))

    console.print()
    console.print(table)
    console.print()


def _md_rankings_table(sorted_scores: list[ModelScore]) -> list[str]:
    """Generate the markdown rankings table section."""
    lines: list[str] = ["## Rankings", ""]
    has_ci = any(ms.multi_run.n_runs >= 2 for ms in sorted_scores)

    header = "| # | Model | PSI |"
    separator = "|---|-------|-----|"
    if has_ci:
        header += " 95% CI | N |"
        separator += "--------|---|"
    header += " Init | Decay | SR Con | Gap | Ext |"
    separator += "------|-------|--------|-----|-----|"
    lines.extend([header, separator])

    for rank, ms in enumerate(sorted_scores, 1):
        dims = ms.dimensions
        row = f"| {rank} | {ms.model_id} | {ms.stability_index:.1f} |"
        if has_ci:
            if ms.multi_run.n_runs >= 2:
                row += f" {ms.multi_run.ci_low:.1f}-{ms.multi_run.ci_high:.1f} | {ms.multi_run.n_runs} |"
            else:
                row += " -- | 1 |"
        row += (
            f" {dims.initial_expression:.1f} | {dims.decay_resistance:.1f} |"
            f" {dims.self_report_consistency:.1f} | {dims.observer_self_agreement:.1f} |"
            f" {dims.extended_stability:.1f} |"
        )
        lines.append(row)
    lines.append("")
    return lines


def _md_decay_and_divergence(sorted_scores: list[ModelScore]) -> list[str]:
    """Generate decay curves and divergence sections."""
    lines: list[str] = [
        "## Decay Curves",
        "",
        "Observer-rated ADHD expression at each checkpoint turn (0-36 scale):",
        "",
    ]
    for ms in sorted_scores:
        if ms.decay_curve.turns:
            curve_str = ", ".join(
                f"T{t}={v:.1f}" for t, v in zip(ms.decay_curve.turns, ms.decay_curve.observer_means, strict=True)
            )
            lines.append(f"- **{ms.model_id}**: {curve_str}")
    lines.extend(["", "## Self-Report vs Observer Divergence", ""])
    for ms in sorted_scores:
        if ms.decay_curve.turns:
            sr_vals = ms.decay_curve.self_report_scores
            obs_vals = ms.decay_curve.observer_means
            sr_mean = sum(sr_vals) / len(sr_vals) if sr_vals else 0
            obs_mean = sum(obs_vals) / len(obs_vals) if obs_vals else 0
            gap = abs(sr_mean - obs_mean)
            lines.append(f"- **{ms.model_id}**: SR mean={sr_mean:.1f}, Observer mean={obs_mean:.1f}, Gap={gap:.1f}")
    lines.append("")
    return lines


def generate_markdown_report(model_scores: list[ModelScore], output_path: Path | None = None) -> str:
    """Generate a Markdown leaderboard report."""
    path = output_path or (RESULTS_DIR / "LEADERBOARD.md")
    sorted_scores = sorted(model_scores, key=lambda s: s.stability_index, reverse=True)

    lines: list[str] = [
        "# Persona Decay Benchmark -- Leaderboard",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "Measures how well LLMs sustain high-intensity ADHD persona expression over ",
        "36-turn conversations. Higher Persona Stability Index (PSI) = better persona maintenance.",
        "",
    ]
    lines.extend(_md_rankings_table(sorted_scores))
    lines.extend(_md_decay_and_divergence(sorted_scores))
    lines.extend(
        [
            "## Methodology",
            "",
            "- **Persona**: High-intensity ADHD (from Stable Personas paper, arXiv:2601.22812v1)",
            "- **Conversation**: 36 turns with neutral partner (gemini-3.1-flash-lite)",
            "- **Assessment**: 12-item ADHD questionnaire at turns 6, 12, 18, 24, 30, 36",
            "- **Observer**: 3 independent gemini-3-flash-preview evaluations per checkpoint",
            "- **PSI formula**: Init(20%) + Decay(40%) + SR-Consistency(15%) + Gap(10%) + Extended(15%)",
            "",
        ]
    )

    md_text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_text, encoding="utf-8")
    return md_text


def export_json(
    model_scores: list[ModelScore],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Export leaderboard data as JSON.

    Args:
        model_scores: List of scored models.
        output_path: Where to write the file. Defaults to results/leaderboard.json.

    Returns:
        The exported data dict.
    """
    path = output_path or (RESULTS_DIR / "leaderboard.json")
    sorted_scores = sorted(model_scores, key=lambda s: s.stability_index, reverse=True)

    data: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "persona-decay-bench",
        "models": [ms.model_id for ms in sorted_scores],
        "scores": [ms.to_dict() for ms in sorted_scores],
        "decay_curves": {ms.model_id: ms.decay_curve.to_dict() for ms in sorted_scores if ms.decay_curve.turns},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return data
