"""CLI: Click commands for run, evaluate, leaderboard, generate-report, estimate-cost, clear-cache.

Supports parallel execution: ``--parallel N`` runs N models concurrently
via a ``ThreadPoolExecutor``, inspired by the AI Independence Bench
parallel runner.  Pass ``--verbose`` / ``-v`` to stream full model
responses to the terminal in real time (sequential mode only; verbose
is automatically disabled when running in parallel).
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import click
from rich.console import Console

from src.config import (
    API_CALL_TIMEOUT,
    CHECKPOINT_TURNS,
    MAX_TURNS,
    MODEL_CONFIGS,
    OBSERVER_CALLS,
    OBSERVER_MODEL,
    PARTNER_MODEL,
    RUNS_PER_MODEL,
    ModelConfig,
    ModelPricing,
    ensure_dirs,
    get_model_config,
    load_api_key,
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


# ---------------------------------------------------------------------------
# Single-model pipeline (designed to be called from a thread pool)
# ---------------------------------------------------------------------------


@dataclass
class _ModelRunResult:
    """Result of running conversations for a single model."""

    label: str
    completed: int = 0
    total: int = 0
    error: str | None = None


def _run_single_model(
    api_key: str,
    cfg: ModelConfig,
    runs: int,
    max_turns: int,
    timeout: float,
    verbose: bool,
) -> _ModelRunResult:
    """Run all conversations for one model.  Thread-safe: creates its own client."""
    from src.openrouter_client import OpenRouterClient
    from src.runner import run_all_conversations

    result = _ModelRunResult(label=cfg.label, total=runs)
    try:
        client = OpenRouterClient(api_key, timeout=timeout)
        conv_results = run_all_conversations(
            client=client,
            model_config=cfg,
            n_runs=runs,
            max_turns=max_turns,
            verbose=verbose,
        )
        result.completed = sum(1 for r in conv_results if r.get("status") in ("completed", "cached"))
        console.print(f"  [green]{cfg.label}: completed {result.completed}/{runs} conversations[/green]")
    except Exception as e:
        result.error = str(e)
        console.print(f"  [red]{cfg.label}: ERROR — {e}[/red]")
    return result


@dataclass
class _ModelEvalResult:
    """Result of evaluating a single model."""

    label: str
    n_checkpoints: int = 0
    n_conversations: int = 0
    error: str | None = None


def _eval_single_model(
    api_key: str,
    cfg: ModelConfig,
    timeout: float,
    verbose: bool,
) -> _ModelEvalResult:
    """Run observer assessments for one model.  Thread-safe: creates its own client."""
    from src.evaluator import evaluate_model
    from src.openrouter_client import OpenRouterClient

    result = _ModelEvalResult(label=cfg.label)
    try:
        client = OpenRouterClient(api_key, timeout=timeout)
        eval_results = evaluate_model(client, cfg, verbose=verbose)
        result.n_conversations = len(eval_results)
        result.n_checkpoints = sum(len(r.get("checkpoints", {})) for r in eval_results)
        console.print(
            f"  [green]{cfg.label}: evaluated {result.n_checkpoints} checkpoints "
            f"across {result.n_conversations} conversations[/green]"
        )
    except Exception as e:
        result.error = str(e)
        console.print(f"  [red]{cfg.label}: ERROR — {e}[/red]")
    return result


@click.group()
def cli() -> None:
    """Persona Decay Benchmark — measure how well LLMs sustain persona expression."""
    ensure_dirs()


@cli.command()
@click.option("--models", type=str, default=None, help="Comma-separated model IDs to run.")
@click.option("--parallel", "-p", type=int, default=1, help="Number of models to run in parallel.")
@click.option("--runs", type=int, default=RUNS_PER_MODEL, help="Number of conversations per model.")
@click.option("--max-turns", type=int, default=MAX_TURNS, help="Max conversation turns.")
@click.option("--timeout", type=float, default=API_CALL_TIMEOUT, help="Per-API-call timeout in seconds.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show full model responses in real time.")
def run(models: str | None, parallel: int, runs: int, max_turns: int, timeout: float, verbose: bool) -> None:
    """Run persona conversations for specified models."""
    api_key = load_api_key()
    model_configs = _resolve_models(models)
    n_workers = max(1, min(parallel, len(model_configs)))

    if verbose and n_workers > 1:
        console.print("[yellow]Verbose mode disabled — not supported with parallel execution.[/yellow]")
        verbose = False

    console.print(f"\n[bold]Running {len(model_configs)} model(s), {runs} conversations each, {max_turns} turns[/bold]")
    if n_workers > 1:
        console.print(f"  [yellow]Parallel workers: {n_workers}[/yellow]")
    if verbose:
        console.print("  [yellow]Verbose mode: ON[/yellow]")
    console.print()

    if n_workers == 1:
        for cfg in model_configs:
            console.print(f"[bold cyan]Model: {cfg.label}[/bold cyan]")
            result = _run_single_model(api_key, cfg, runs, max_turns, timeout, verbose)
            if result.error:
                console.print(f"  [red]Error: {result.error}[/red]\n")
            else:
                console.print()
    else:
        console.print(f"[bold blue]Launching {n_workers} models in parallel...[/bold blue]\n")
        results: list[_ModelRunResult] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_single_model, api_key, cfg, runs, max_turns, timeout, verbose): cfg.label
                for cfg in model_configs
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    mr = future.result()
                except Exception as e:
                    mr = _ModelRunResult(label=label, error=str(e))
                    console.print(f"  [red]{label}: FATAL — {e}[/red]")
                results.append(mr)

        failed = [r for r in results if r.error]
        succeeded = [r for r in results if not r.error]
        console.print(f"\n[bold]Summary: {len(succeeded)} succeeded, {len(failed)} failed[/bold]")
        if failed:
            for r in failed:
                console.print(f"  [red]{r.label}: {r.error}[/red]")


@cli.command()
@click.option("--models", type=str, default=None, help="Comma-separated model IDs to evaluate.")
@click.option("--parallel", "-p", type=int, default=1, help="Number of models to evaluate in parallel.")
@click.option("--timeout", type=float, default=API_CALL_TIMEOUT, help="Per-API-call timeout in seconds.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show full observer responses in real time.")
def evaluate(models: str | None, parallel: int, timeout: float, verbose: bool) -> None:
    """Run observer assessments on completed conversations."""
    api_key = load_api_key()
    model_configs = _resolve_models(models)
    n_workers = max(1, min(parallel, len(model_configs)))

    if verbose and n_workers > 1:
        console.print("[yellow]Verbose mode disabled — not supported with parallel execution.[/yellow]")
        verbose = False

    console.print(f"\n[bold]Evaluating {len(model_configs)} model(s) with {OBSERVER_CALLS}x {OBSERVER_MODEL}[/bold]")
    if n_workers > 1:
        console.print(f"  [yellow]Parallel workers: {n_workers}[/yellow]")
    if verbose:
        console.print("  [yellow]Verbose mode: ON[/yellow]")
    console.print()

    if n_workers == 1:
        for cfg in model_configs:
            console.print(f"[bold cyan]Model: {cfg.label}[/bold cyan]")
            result = _eval_single_model(api_key, cfg, timeout, verbose)
            if result.error:
                console.print(f"  [red]Error: {result.error}[/red]\n")
            else:
                console.print()
    else:
        console.print(f"[bold blue]Launching {n_workers} models in parallel...[/bold blue]\n")
        results: list[_ModelEvalResult] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_eval_single_model, api_key, cfg, timeout, verbose): cfg.label for cfg in model_configs
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    er = future.result()
                except Exception as e:
                    er = _ModelEvalResult(label=label, error=str(e))
                    console.print(f"  [red]{label}: FATAL — {e}[/red]")
                results.append(er)

        failed = [r for r in results if r.error]
        succeeded = [r for r in results if not r.error]
        total_cp = sum(r.n_checkpoints for r in succeeded)
        console.print(
            f"\n[bold]Summary: {len(succeeded)} succeeded ({total_cp} checkpoints), " f"{len(failed)} failed[/bold]"
        )
        if failed:
            for r in failed:
                console.print(f"  [red]{r.label}: {r.error}[/red]")


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

    generate_markdown_report(scores)
    export_json(scores)

    console.print(f"[green]Generated results/LEADERBOARD.md ({len(scores)} models)[/green]")
    console.print("[green]Generated results/leaderboard.json[/green]")


@cli.command("estimate-cost")
@click.option("--models", type=str, default=None, help="Comma-separated model IDs.")
@click.option("--runs", type=int, default=RUNS_PER_MODEL, help="Number of conversations per model.")
def estimate_cost(models: str | None, runs: int) -> None:
    """Estimate benchmark cost without running."""
    load_api_key(required=False)

    model_configs = _resolve_models(models)

    # Rough cost estimates based on typical token usage:
    # - 36 turns x ~500 tokens/response = ~18K completion tokens per conversation
    # - Context grows: ~200K prompt tokens per conversation (cumulative)
    # - Self-report: 6 checkpoints x ~2K prompt + ~200 completion = ~13K tokens
    # - Observer: 6 checkpoints x 3 calls x ~5K prompt + ~200 completion = ~93K tokens
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

    from src.openrouter_client import OpenRouterClient

    pricing_map: dict[str, ModelPricing] = {}
    try:
        console.print("  [dim]Fetching pricing from OpenRouter...[/dim]")
        pricing_map = OpenRouterClient.fetch_public_pricing()
        has_pricing = bool(pricing_map)
    except Exception as exc:
        console.print(f"  [dim]Could not fetch pricing: {exc}[/dim]")
        has_pricing = False

    total_estimated = 0.0
    for cfg in model_configs:
        if has_pricing:
            pricing = pricing_map.get(cfg.model_id, ModelPricing())
            partner_pricing = pricing_map.get(PARTNER_MODEL, ModelPricing())
            observer_pricing = pricing_map.get(OBSERVER_MODEL, ModelPricing())

            conv_cost = avg_prompt_per_conv * pricing.prompt_price + avg_completion_per_conv * pricing.completion_price
            sr_cost = avg_sr_prompt * pricing.prompt_price + avg_sr_completion * pricing.completion_price
            partner_cost = (
                avg_prompt_per_conv * partner_pricing.prompt_price
                + avg_completion_per_conv * partner_pricing.completion_price
            )
            obs_cost = (
                avg_obs_prompt * observer_pricing.prompt_price + avg_obs_completion * observer_pricing.completion_price
            )

            per_conv = conv_cost + sr_cost + partner_cost + obs_cost
            model_total = per_conv * runs
            total_estimated += model_total

            console.print(f"  {cfg.label}: ~${model_total:.2f} ({runs} convs x ${per_conv:.4f}/conv)")
        else:
            console.print(f"  {cfg.label}: (could not fetch pricing)")

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
