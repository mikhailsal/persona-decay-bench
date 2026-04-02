"""Leaderboard: Rich terminal table, Markdown report generation, JSON export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import RESULTS_DIR
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


def display_leaderboard(
    model_scores: list[ModelScore],
    *,
    detailed: bool = False,
) -> None:
    """Display the leaderboard table in the terminal.

    Args:
        model_scores: List of scored models.
        detailed: If True, show all dimension columns.
    """
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
        table.add_column("Init", justify="right", width=4)
        table.add_column("Decay", justify="right", width=5)
        table.add_column("SRCon", justify="right", width=5)
        table.add_column("Gap", justify="right", width=4)
        table.add_column("Ext", justify="right", width=4)
    else:
        table.add_column("Decay", justify="right", width=5)
        table.add_column("Init", justify="right", width=4)

    for rank, ms in enumerate(sorted_scores, 1):
        dims = ms.dimensions
        index_text = Text(
            f"{ms.stability_index:.1f}",
            style=_score_color(ms.stability_index),
        )

        # Shorten model name
        name = ms.model_id
        if "@" in name:
            base = name.rsplit("@", 1)[0]
            name = base

        row: list[Any] = [str(rank), name, index_text]

        if has_multi_run:
            if ms.multi_run.n_runs >= 2:
                ci_text = f"{ms.multi_run.ci_low:.1f}-{ms.multi_run.ci_high:.1f}"
                row.append(Text(ci_text, style="dim"))
                row.append(str(ms.multi_run.n_runs))
            else:
                row.append(Text("—", style="dim"))
                row.append("1")

        if detailed:
            row.append(_fmt_score(dims.initial_expression))
            row.append(_fmt_score(dims.decay_resistance))
            row.append(_fmt_score(dims.self_report_consistency))
            row.append(_fmt_score(dims.observer_self_agreement))
            row.append(_fmt_score(dims.extended_stability))
        else:
            row.append(_fmt_score(dims.decay_resistance))
            row.append(_fmt_score(dims.initial_expression))

        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def generate_markdown_report(
    model_scores: list[ModelScore],
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown leaderboard report.

    Args:
        model_scores: List of scored models.
        output_path: Where to write the file. Defaults to results/LEADERBOARD.md.

    Returns:
        The generated Markdown string.
    """
    path = output_path or (RESULTS_DIR / "LEADERBOARD.md")
    sorted_scores = sorted(model_scores, key=lambda s: s.stability_index, reverse=True)

    lines: list[str] = []
    lines.append("# Persona Decay Benchmark — Leaderboard")
    lines.append("")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")
    lines.append("Measures how well LLMs sustain high-intensity ADHD persona expression over ")
    lines.append("36-turn conversations. Higher Persona Stability Index (PSI) = better persona maintenance.")
    lines.append("")

    # Summary table
    lines.append("## Rankings")
    lines.append("")
    has_ci = any(ms.multi_run.n_runs >= 2 for ms in sorted_scores)

    header = "| # | Model | PSI |"
    separator = "|---|-------|-----|"
    if has_ci:
        header += " 95% CI | N |"
        separator += "--------|---|"
    header += " Init | Decay | SR Con | Gap | Ext |"
    separator += "------|-------|--------|-----|-----|"

    lines.append(header)
    lines.append(separator)

    for rank, ms in enumerate(sorted_scores, 1):
        dims = ms.dimensions
        row = f"| {rank} | {ms.model_id} | {ms.stability_index:.1f} |"
        if has_ci:
            if ms.multi_run.n_runs >= 2:
                row += f" {ms.multi_run.ci_low:.1f}-{ms.multi_run.ci_high:.1f} | {ms.multi_run.n_runs} |"
            else:
                row += " — | 1 |"
        row += (
            f" {dims.initial_expression:.1f} |"
            f" {dims.decay_resistance:.1f} |"
            f" {dims.self_report_consistency:.1f} |"
            f" {dims.observer_self_agreement:.1f} |"
            f" {dims.extended_stability:.1f} |"
        )
        lines.append(row)

    lines.append("")

    # Decay curves section
    lines.append("## Decay Curves")
    lines.append("")
    lines.append("Observer-rated ADHD expression at each checkpoint turn (0-36 scale):")
    lines.append("")

    for ms in sorted_scores:
        if ms.decay_curve.turns:
            curve_str = ", ".join(
                f"T{t}={v:.1f}"
                for t, v in zip(ms.decay_curve.turns, ms.decay_curve.observer_means)
            )
            lines.append(f"- **{ms.model_id}**: {curve_str}")

    lines.append("")

    # Self-report vs observer
    lines.append("## Self-Report vs Observer Divergence")
    lines.append("")
    for ms in sorted_scores:
        if ms.decay_curve.turns:
            sr_mean = sum(ms.decay_curve.self_report_scores) / len(ms.decay_curve.self_report_scores) if ms.decay_curve.self_report_scores else 0
            obs_mean = sum(ms.decay_curve.observer_means) / len(ms.decay_curve.observer_means) if ms.decay_curve.observer_means else 0
            gap = abs(sr_mean - obs_mean)
            lines.append(
                f"- **{ms.model_id}**: SR mean={sr_mean:.1f}, Observer mean={obs_mean:.1f}, Gap={gap:.1f}"
            )

    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Persona**: High-intensity ADHD (from Stable Personas paper, arXiv:2601.22812v1)")
    lines.append("- **Conversation**: 36 turns with neutral partner (gemini-3.1-flash-lite)")
    lines.append("- **Assessment**: 12-item ADHD questionnaire at turns 6, 12, 18, 24, 30, 36")
    lines.append("- **Observer**: 3 independent gemini-3-flash-preview evaluations per checkpoint")
    lines.append("- **PSI formula**: Init(20%) + Decay(40%) + SR-Consistency(15%) + Gap(10%) + Extended(15%)")
    lines.append("")

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
        "decay_curves": {
            ms.model_id: ms.decay_curve.to_dict()
            for ms in sorted_scores
            if ms.decay_curve.turns
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return data
