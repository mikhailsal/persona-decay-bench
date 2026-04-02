"""CLI: Click commands for run, evaluate, leaderboard, generate-report, estimate-cost, clear-cache."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from src.config import (
    CHECKPOINT_TURNS,
    MAX_TURNS,
    MODEL_CONFIGS,
    OBSERVER_CALLS,
    OBSERVER_MODEL,
    PARTNER_MODEL,
    RUNS_PER_MODEL,
    ModelConfig,
    ensure_dirs,
    get_model_config,
    load_api_key,
    load_model_configs,
)

console = Console()


def _resolve_models(models_str: str | None) -> list[ModelConfig]:
    """Resolve a comma-separated model list into ModelConfig objects.

    If models_str is None, returns all active models from YAML config.
    """
    if models_str:
        model_ids = [m.strip() for m in models_str.split(",") if m.strip()]
        return [get_model_config(mid) for mid in model_ids]

    active = [cfg for cfg in MODEL_CONFIGS.values() if cfg.active]
    if not active:
        console.print("[red]No active models found. Add models to configs/models.yaml or use --models.[/red]")
        sys.exit(1)
    return active


@click.group()
def cli() -> None:
    """Persona Decay Benchmark — measure how well LLMs sustain persona expression."""
    ensure_dirs()


@cli.command()
@click.option("--models", type=str, default=None, help="Comma-separated model IDs to run.")
@click.option("--parallel", type=int, default=1, help="Number of parallel workers.")
@click.option("--runs", type=int, default=RUNS_PER_MODEL, help="Number of conversations per model.")
@click.option("--max-turns", type=int, default=MAX_TURNS, help="Max conversation turns.")
def run(models: str | None, parallel: int, runs: int, max_turns: int) -> None:
    """Run persona conversations for specified models."""
    api_key = load_api_key()
    from src.openrouter_client import OpenRouterClient
    from src.runner import run_all_conversations

    client = OpenRouterClient(api_key)
    model_configs = _resolve_models(models)

    console.print(f"\n[bold]Running {len(model_configs)} model(s), {runs} conversations each, {max_turns} turns[/bold]\n")

    for cfg in model_configs:
        console.print(f"[bold cyan]Model: {cfg.label}[/bold cyan]")
        try:
            results = run_all_conversations(
                client=client,
                model_config=cfg,
                n_runs=runs,
                max_turns=max_turns,
            )
            completed = sum(1 for r in results if r.get("status") in ("completed", "cached"))
            console.print(f"  [green]Completed {completed}/{runs} conversations[/green]\n")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]\n")


@cli.command()
@click.option("--models", type=str, default=None, help="Comma-separated model IDs to evaluate.")
def evaluate(models: str | None) -> None:
    """Run observer assessments on completed conversations."""
    api_key = load_api_key()
    from src.evaluator import evaluate_model
    from src.openrouter_client import OpenRouterClient

    client = OpenRouterClient(api_key)
    model_configs = _resolve_models(models)

    console.print(f"\n[bold]Evaluating {len(model_configs)} model(s) with {OBSERVER_CALLS}x {OBSERVER_MODEL}[/bold]\n")

    for cfg in model_configs:
        console.print(f"[bold cyan]Model: {cfg.label}[/bold cyan]")
        try:
            results = evaluate_model(client, cfg)
            n_checkpoints = sum(len(r.get("checkpoints", {})) for r in results)
            console.print(f"  [green]Evaluated {n_checkpoints} checkpoints across {len(results)} conversations[/green]\n")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]\n")


@cli.command()
@click.option("--detailed", is_flag=True, help="Show all dimension columns.")
def leaderboard(detailed: bool) -> None:
    """Display the leaderboard from cached results."""
    from src.cache import list_all_cached_models
    from src.config import get_config_by_dir_name
    from src.scorer import score_model

    cached_models = list_all_cached_models()
    if not cached_models:
        console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
        return

    scores = []
    for config_dir in cached_models:
        cfg = get_config_by_dir_name(config_dir)
        if cfg is None:
            cfg = ModelConfig(model_id=config_dir.split("@")[0].replace("--", "/", 1))
        ms = score_model(cfg.label, config=cfg)
        if ms.stability_index > 0 or ms.n_conversations > 0:
            scores.append(ms)

    from src.leaderboard import display_leaderboard
    display_leaderboard(scores, detailed=detailed)


@cli.command("generate-report")
def generate_report() -> None:
    """Generate Markdown leaderboard report and JSON export."""
    from src.cache import list_all_cached_models
    from src.config import get_config_by_dir_name
    from src.leaderboard import export_json, generate_markdown_report
    from src.scorer import score_model

    cached_models = list_all_cached_models()
    if not cached_models:
        console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
        return

    scores = []
    for config_dir in cached_models:
        cfg = get_config_by_dir_name(config_dir)
        if cfg is None:
            cfg = ModelConfig(model_id=config_dir.split("@")[0].replace("--", "/", 1))
        ms = score_model(cfg.label, config=cfg)
        if ms.stability_index > 0 or ms.n_conversations > 0:
            scores.append(ms)

    md = generate_markdown_report(scores)
    data = export_json(scores)

    console.print(f"[green]Generated results/LEADERBOARD.md ({len(scores)} models)[/green]")
    console.print(f"[green]Generated results/leaderboard.json[/green]")


@cli.command("estimate-cost")
@click.option("--models", type=str, default=None, help="Comma-separated model IDs.")
@click.option("--runs", type=int, default=RUNS_PER_MODEL, help="Number of conversations per model.")
def estimate_cost(models: str | None, runs: int) -> None:
    """Estimate benchmark cost without running."""
    api_key = load_api_key(required=False)

    model_configs = _resolve_models(models)

    # Rough cost estimates based on typical token usage:
    # - 36 turns × ~500 tokens/response = ~18K completion tokens per conversation
    # - Context grows: ~200K prompt tokens per conversation (cumulative)
    # - Self-report: 6 checkpoints × ~2K prompt + ~200 completion = ~13K tokens
    # - Observer: 6 checkpoints × 3 calls × ~5K prompt + ~200 completion = ~93K tokens
    avg_prompt_per_conv = 200_000
    avg_completion_per_conv = 18_000
    avg_sr_prompt = 13_000
    avg_sr_completion = 1_200
    avg_obs_prompt = 90_000
    avg_obs_completion = 3_600

    console.print("\n[bold]Cost Estimate[/bold]")
    console.print(f"  Conversations per model: {runs}")
    console.print(f"  Turns per conversation: {MAX_TURNS}")
    console.print(f"  Checkpoints: {len(CHECKPOINT_TURNS)}")
    console.print(f"  Observer calls per checkpoint: {OBSERVER_CALLS}")
    console.print()

    if api_key:
        from src.openrouter_client import OpenRouterClient
        client = OpenRouterClient(api_key)
        try:
            client.fetch_pricing()
            has_pricing = True
        except Exception:
            has_pricing = False
    else:
        has_pricing = False

    total_estimated = 0.0
    for cfg in model_configs:
        if has_pricing:
            pricing = client.get_model_pricing(cfg.model_id)
            partner_pricing = client.get_model_pricing(PARTNER_MODEL)
            observer_pricing = client.get_model_pricing(OBSERVER_MODEL)

            conv_cost = (
                avg_prompt_per_conv * pricing.prompt_price
                + avg_completion_per_conv * pricing.completion_price
            )
            sr_cost = (
                avg_sr_prompt * pricing.prompt_price
                + avg_sr_completion * pricing.completion_price
            )
            partner_cost = (
                avg_prompt_per_conv * partner_pricing.prompt_price
                + avg_completion_per_conv * partner_pricing.completion_price
            )
            obs_cost = (
                avg_obs_prompt * observer_pricing.prompt_price
                + avg_obs_completion * observer_pricing.completion_price
            )

            per_conv = conv_cost + sr_cost + partner_cost + obs_cost
            model_total = per_conv * runs
            total_estimated += model_total

            console.print(f"  {cfg.label}: ~${model_total:.2f} ({runs} convs × ${per_conv:.4f}/conv)")
        else:
            console.print(f"  {cfg.label}: (set API key for pricing estimate)")

    if has_pricing:
        console.print(f"\n  [bold]Total estimate: ~${total_estimated:.2f}[/bold]")
    console.print()


@cli.command("clear-cache")
@click.option("--scores-only", is_flag=True, help="Clear only observer scores, keeping conversations.")
@click.confirmation_option(prompt="Are you sure you want to clear cached data?")
def clear_cache(scores_only: bool) -> None:
    """Clear cached conversations and/or scores."""
    from src.cache import clear_all_cache, clear_observer_scores

    if scores_only:
        count = clear_observer_scores()
        console.print(f"[green]Cleared observer scores from {count} checkpoint(s).[/green]")
    else:
        count = clear_all_cache()
        console.print(f"[green]Cleared {count} file(s) from cache.[/green]")


if __name__ == "__main__":
    cli()
