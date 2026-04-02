"""Runner: orchestrates multi-turn conversations with neutral partner and checkpoint collection.

Flow for one conversation:
  1. Target model gets ADHD persona system prompt + workday task as first user message
  2. Neutral partner (gemini-3.1-flash-lite) generates follow-up questions
  3. At checkpoint turns (6, 12, 18, 24, 30, 36), collect CAARS self-report
  4. Conversation turns are saved incrementally to JSONL
"""

from __future__ import annotations

import logging
import uuid
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
    """Find an existing conversation for a run — complete or partial.

    Scans all conversation IDs under the run directory.  If a complete one
    is found it is returned with status ``"cached"``.  If a partial one is
    found it is returned with status ``"partial"`` so the caller can resume.
    """
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

    console.print(
        Text(
            f"  [{model_config.label}] Conversation {conv_id} complete. Total cost: ${total_cost_usd:.4f}",
            style="green",
        )
    )
    return {
        "conversation_id": conv_id,
        "model_config": model_config,
        "run": run_number,
        "turns": turns,
        "total_cost_usd": total_cost_usd,
        "status": "completed",
    }


def run_all_conversations(
    client: OpenRouterClient,
    model_config: ModelConfig,
    *,
    n_runs: int = 5,
    max_turns: int = MAX_TURNS,
    checkpoint_turns: list[int] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run all conversations for a model (n_runs conversations).

    Automatically detects and resumes partial conversations from cache,
    and skips fully completed ones.
    """
    config_dir = model_config.config_dir_name
    results = []
    for run_num in range(1, n_runs + 1):
        existing = _find_existing_conversation(config_dir, run_num, model_config, max_turns)
        if existing and existing["status"] == "cached":
            console.print(
                Text(
                    f"\n  [{model_config.label}] Run {run_num} already complete "
                    f"({existing['conversation_id']}), skipping.",
                    style="dim",
                )
            )
            results.append(existing)
            continue
        conv_id = existing["conversation_id"] if existing else _generate_conversation_id()
        result = run_conversation(
            client=client,
            model_config=model_config,
            run_number=run_num,
            conversation_id=conv_id,
            max_turns=max_turns,
            checkpoint_turns=checkpoint_turns,
            verbose=verbose,
        )
        results.append(result)
    return results
