"""Runner: orchestrates multi-turn persona conversations with checkpoint collection."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.text import Text

from src.cache import (
    append_turn,
    checkpoint_exists,
    conversation_exists,
    list_conversations,
    load_conversation,
    save_checkpoint,
)
from src.config import (
    CHECKPOINT_TURNS,
    MAX_TURNS,
    PARTNER_MAX_TOKENS,
    PARTNER_MODEL,
    PARTNER_TEMPERATURE,
    RESPONSE_MAX_TOKENS,
    ModelConfig,
)
from src.prompts import WORKDAY_TASK
from src.runner_helpers import (
    build_participant_turn,
    build_partner_messages,
    build_partner_turn,
    build_target_messages,
    collect_self_report,
    print_turn_with_cache,
)

if TYPE_CHECKING:
    from src.openrouter_client import OpenRouterClient

log = logging.getLogger(__name__)
console = Console()


def _generate_conversation_id() -> str:
    """Generate a short unique conversation ID."""
    return uuid.uuid4().hex[:12]


def _check_cached_conversation(
    config_dir: str,
    run_number: int,
    conv_id: str,
    model_config: ModelConfig,
    max_turns: int,
) -> dict[str, Any] | None:
    """Return a cached result dict if the conversation is already complete."""
    expected_messages = 2 + 2 * max_turns
    if not conversation_exists(config_dir, run_number, conv_id):
        return None
    existing_turns = load_conversation(config_dir, run_number, conv_id)
    if not existing_turns or len(existing_turns) < expected_messages:
        return None
    log.info("Conversation %s already complete (%d messages)", conv_id, len(existing_turns))
    return {
        "conversation_id": conv_id,
        "model_config": model_config,
        "run": run_number,
        "turns": existing_turns,
        "status": "cached",
    }


def _find_existing_conversation(
    config_dir: str,
    run_number: int,
    model_config: ModelConfig,
    max_turns: int,
) -> dict[str, Any] | None:
    """Find an existing conversation for a run — complete or partial."""
    expected_messages = 2 + 2 * max_turns
    conv_ids = list_conversations(config_dir, run_number)
    best_partial: dict[str, Any] | None = None
    for cid in conv_ids:
        turns = load_conversation(config_dir, run_number, cid)
        if not turns:
            continue
        if len(turns) >= expected_messages:
            log.info("Conversation %s already complete (%d msgs)", cid, len(turns))
            return {
                "conversation_id": cid,
                "model_config": model_config,
                "run": run_number,
                "turns": turns,
                "status": "cached",
            }
        if best_partial is None or len(turns) > len(best_partial["turns"]):
            best_partial = {
                "conversation_id": cid,
                "model_config": model_config,
                "run": run_number,
                "turns": turns,
                "status": "partial",
            }
    return best_partial


def _maybe_collect_checkpoint(
    client: OpenRouterClient,
    model_config: ModelConfig,
    turns: list[dict[str, Any]],
    config_dir: str,
    run_number: int,
    conv_id: str,
    exchange_round: int,
) -> float:
    """Collect a self-report checkpoint if not already cached. Returns cost."""
    if checkpoint_exists(config_dir, run_number, conv_id, exchange_round):
        console.print(f"    ── Checkpoint at exchange {exchange_round} (already cached) ──")
        return 0.0
    console.print(f"    ── Checkpoint at exchange {exchange_round} ──")
    sr_data = collect_self_report(client, model_config, turns, exchange_round)
    cost: float = sr_data["cost"]["cost_usd"]
    save_checkpoint(
        config_dir,
        run_number,
        conv_id,
        exchange_round,
        {
            "turn": exchange_round,
            "self_report": sr_data,
            "conversation_snapshot_turns": len(turns),
        },
    )
    console.print(f"    ── Self-report collected (${cost:.4f}) ──")
    return cost


def _run_exchange_loop(
    client: OpenRouterClient,
    model_config: ModelConfig,
    turns: list[dict[str, Any]],
    config_dir: str,
    run_number: int,
    conv_id: str,
    max_turns: int,
    checkpoints: list[int],
    *,
    start_exchange: int = 1,
    verbose: bool = False,
) -> float:
    """Run the partner/participant exchange loop. Returns total cost accumulated.

    When *start_exchange* > 1 the loop resumes from the given exchange,
    allowing a conversation to be continued from a partial cache.
    """
    total_cost = 0.0
    msg_number = len(turns) + 1
    for exchange_round in range(start_exchange, max_turns + 1):
        partner_result = client.chat(
            model=PARTNER_MODEL,
            messages=build_partner_messages(turns),
            max_tokens=PARTNER_MAX_TOKENS,
            temperature=PARTNER_TEMPERATURE,
            cache_control=True,
        )
        total_cost += partner_result.usage.cost_usd
        partner_turn = build_partner_turn(partner_result, msg_number, exchange_round)
        turns.append(partner_turn)
        append_turn(config_dir, run_number, conv_id, partner_turn)
        print_turn_with_cache(
            f"Turn {exchange_round:2d}",
            "partner",
            partner_result.content,
            partner_result.usage.prompt_tokens,
            partner_result.usage.cached_tokens,
            partner_result.usage.cache_write_tokens,
            verbose=verbose,
        )
        msg_number += 1

        target_result = client.chat(
            model=model_config.model_id,
            messages=build_target_messages(turns),
            max_tokens=RESPONSE_MAX_TOKENS,
            temperature=model_config.effective_temperature,
            reasoning_effort=model_config.effective_reasoning,
            provider=model_config.provider,
            cache_control=True,
        )
        total_cost += target_result.usage.cost_usd
        pturn = build_participant_turn(target_result, msg_number, exchange_round)
        turns.append(pturn)
        append_turn(config_dir, run_number, conv_id, pturn)
        print_turn_with_cache(
            f"Turn {exchange_round:2d}",
            "participant",
            target_result.content,
            target_result.usage.prompt_tokens,
            target_result.usage.cached_tokens,
            target_result.usage.cache_write_tokens,
            verbose=verbose,
        )
        msg_number += 1

        if exchange_round in checkpoints:
            total_cost += _maybe_collect_checkpoint(
                client,
                model_config,
                turns,
                config_dir,
                run_number,
                conv_id,
                exchange_round,
            )

    return total_cost


def _build_completed_result(
    model_config: ModelConfig,
    conv_id: str,
    run_number: int,
    turns: list[dict[str, Any]],
    total_cost_usd: float,
) -> dict[str, Any]:
    """Build the result dict for a completed conversation and log cost breakdown."""
    p_cost = sum(t.get("cost_usd", 0) for t in turns if t.get("role") == "participant")
    pr_cost = sum(t.get("cost_usd", 0) for t in turns if t.get("role") == "partner")
    console.print(
        Text(
            f"  [{model_config.label}] Done {conv_id}. "
            f"${total_cost_usd:.4f} (participant ${p_cost:.4f}, partner ${pr_cost:.4f})",
            style="green",
        )
    )
    return {
        "conversation_id": conv_id,
        "model_config": model_config,
        "run": run_number,
        "turns": turns,
        "total_cost_usd": total_cost_usd,
        "participant_cost_usd": p_cost,
        "partner_cost_usd": pr_cost,
        "status": "completed",
    }


def _make_task_turn() -> dict[str, Any]:
    """Build the initial task turn that kicks off the conversation."""
    return {
        "turn": 0,
        "role": "task",
        "content": WORKDAY_TASK,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _resume_exchange(n_messages: int) -> int:
    """Derive which exchange round to resume from given *n_messages* loaded.

    Layout: task(1) + init(1) + pairs of (partner, participant) per exchange.
    """
    if n_messages < 2:
        return 1
    return (n_messages - 2) // 2 + 1


def _start_fresh_conversation(
    client: OpenRouterClient,
    model_config: ModelConfig,
    config_dir: str,
    run_number: int,
    conv_id: str,
    max_turns: int,
    checkpoints: list[int],
    *,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Start a brand-new conversation from scratch. Returns (turns, cost)."""
    turns: list[dict[str, Any]] = []
    console.print(
        Text(f"\n  [{model_config.label}] Starting conversation {conv_id} (run {run_number})", style="bold cyan")
    )
    task_turn = _make_task_turn()
    turns.append(task_turn)
    append_turn(config_dir, run_number, conv_id, task_turn)

    result = client.chat(
        model=model_config.model_id,
        messages=build_target_messages(turns),
        max_tokens=RESPONSE_MAX_TOKENS,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.effective_reasoning,
        provider=model_config.provider,
        cache_control=True,
    )
    initial_turn = build_participant_turn(result, 1, 0)
    turns.append(initial_turn)
    append_turn(config_dir, run_number, conv_id, initial_turn)
    print_turn_with_cache(
        "Init  ",
        "participant",
        result.content,
        result.usage.prompt_tokens,
        result.usage.cached_tokens,
        result.usage.cache_write_tokens,
        verbose=verbose,
    )
    loop_cost = _run_exchange_loop(
        client,
        model_config,
        turns,
        config_dir,
        run_number,
        conv_id,
        max_turns,
        checkpoints,
        verbose=verbose,
    )
    return turns, result.usage.cost_usd + loop_cost


def run_conversation(
    client: OpenRouterClient,
    model_config: ModelConfig,
    run_number: int,
    conversation_id: str | None = None,
    *,
    max_turns: int = MAX_TURNS,
    checkpoint_turns: list[int] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single multi-turn persona conversation.

    Supports resuming: if *conversation_id* points to a partial cache the
    conversation continues from where it left off instead of starting over.
    """
    config_dir = model_config.config_dir_name
    conv_id = conversation_id or _generate_conversation_id()
    checkpoints = checkpoint_turns or CHECKPOINT_TURNS

    cached = _check_cached_conversation(config_dir, run_number, conv_id, model_config, max_turns)
    if cached is not None:
        return cached

    existing = (
        load_conversation(config_dir, run_number, conv_id)
        if conversation_exists(config_dir, run_number, conv_id)
        else []
    )
    if len(existing) >= 2:
        turns = existing
        start_ex = _resume_exchange(len(turns))
        console.print(
            Text(
                f"\n  [{model_config.label}] Resuming conversation {conv_id} "
                f"(run {run_number}) from exchange {start_ex} ({len(turns)} msgs cached)",
                style="bold yellow",
            )
        )
        loop_cost = _run_exchange_loop(
            client,
            model_config,
            turns,
            config_dir,
            run_number,
            conv_id,
            max_turns,
            checkpoints,
            start_exchange=start_ex,
            verbose=verbose,
        )
        total_cost_usd = loop_cost
    else:
        turns, total_cost_usd = _start_fresh_conversation(
            client,
            model_config,
            config_dir,
            run_number,
            conv_id,
            max_turns,
            checkpoints,
            verbose=verbose,
        )

    return _build_completed_result(model_config, conv_id, run_number, turns, total_cost_usd)


def _run_single_run(
    client: OpenRouterClient,
    model_config: ModelConfig,
    run_num: int,
    max_turns: int,
    checkpoint_turns: list[int] | None,
    verbose: bool,
) -> dict[str, Any]:
    """Execute a single run, handling cache detection. Used by both sequential and parallel paths."""
    config_dir = model_config.config_dir_name
    existing = _find_existing_conversation(config_dir, run_num, model_config, max_turns)
    if existing and existing["status"] == "cached":
        console.print(
            Text(
                f"\n  [{model_config.label}] Run {run_num} already complete "
                f"({existing['conversation_id']}), skipping.",
                style="dim",
            )
        )
        return existing
    conv_id = existing["conversation_id"] if existing else _generate_conversation_id()
    return run_conversation(
        client=client,
        model_config=model_config,
        run_number=run_num,
        conversation_id=conv_id,
        max_turns=max_turns,
        checkpoint_turns=checkpoint_turns,
        verbose=verbose,
    )


def run_all_conversations(
    client: OpenRouterClient,
    model_config: ModelConfig,
    *,
    n_runs: int = 5,
    max_turns: int = MAX_TURNS,
    checkpoint_turns: list[int] | None = None,
    verbose: bool = False,
    parallel_runs: int = 1,
) -> list[dict[str, Any]]:
    """Run all conversations for a model, with optional parallel execution.

    When *parallel_runs* > 1, runs execute concurrently via a thread pool.
    """
    n_workers = max(1, min(parallel_runs, n_runs))

    if n_workers <= 1:
        results: list[dict[str, Any]] = []
        for run_num in range(1, n_runs + 1):
            result = _run_single_run(
                client,
                model_config,
                run_num,
                max_turns,
                checkpoint_turns,
                verbose,
            )
            results.append(result)
        return results

    if verbose:
        console.print(
            f"  [yellow]Verbose mode disabled for {model_config.label} — "
            f"not supported with {n_workers} parallel runs.[/yellow]"
        )
        verbose = False

    console.print(f"  [bold blue][{model_config.label}] Launching {n_workers} runs in parallel...[/bold blue]")

    results_by_run: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_single_run,
                client,
                model_config,
                run_num,
                max_turns,
                checkpoint_turns,
                verbose,
            ): run_num
            for run_num in range(1, n_runs + 1)
        }
        for future in as_completed(futures):
            run_num = futures[future]
            try:
                results_by_run[run_num] = future.result()
            except Exception as exc:
                console.print(f"  [red][{model_config.label}] Run {run_num} FAILED — {exc}[/red]")
                results_by_run[run_num] = {
                    "run": run_num,
                    "model_config": model_config,
                    "status": "error",
                    "error": str(exc),
                }

    return [results_by_run[r] for r in sorted(results_by_run)]
