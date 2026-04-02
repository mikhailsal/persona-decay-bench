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
    conversation_exists,
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
from src.prompts import (
    HIGH_ADHD_PERSONA,
    PARTNER_SYSTEM_PROMPT,
    WORKDAY_TASK,
    build_self_report_prompt,
)

if TYPE_CHECKING:
    from src.openrouter_client import CompletionResult, OpenRouterClient

log = logging.getLogger(__name__)
console = Console()


def _print_turn_with_cache(
    label: str,
    role: str,
    content: str,
    prompt_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
) -> None:
    """Print a turn line with cache status."""
    cache_info = ""
    if cached_tokens > 0:
        pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
        cache_info = f" [green]CACHE READ {cached_tokens}/{prompt_tokens} ({pct:.0f}%)[/green]"
    elif cache_write_tokens > 0:
        cache_info = f" [yellow]CACHE WRITE {cache_write_tokens} tokens[/yellow]"
    elif prompt_tokens > 0:
        cache_info = f" [dim]no cache ({prompt_tokens} prompt tokens)[/dim]"
    console.print(f"    {label} [{role:11s}]: {content[:70]}...{cache_info}")


def _generate_conversation_id() -> str:
    """Generate a short unique conversation ID."""
    return uuid.uuid4().hex[:12]


def _build_target_messages(
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the message list for the target model from conversation turns.

    The target model sees: system prompt (persona) + all turns as user/assistant.
    Reasoning details are preserved on assistant messages to maintain reasoning
    continuity.

    All messages use plain string content so that the prefix stays byte-identical
    across requests.  Caching is driven by the top-level ``cache_control`` field
    in the request body (automatic caching), which lets the provider place and
    advance the cache breakpoint without format mismatches between turns.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": HIGH_ADHD_PERSONA},
    ]
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "participant":
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if turn.get("reasoning_details"):
                msg["reasoning_details"] = turn["reasoning_details"]
            elif turn.get("reasoning_content"):
                msg["reasoning"] = turn["reasoning_content"]
            messages.append(msg)
        elif role == "partner" or role == "task":
            messages.append({"role": "user", "content": content})

    return messages


def _build_partner_messages(
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the message list for the neutral partner.

    The partner sees: system prompt + all turns, with participant as user
    and partner as assistant.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PARTNER_SYSTEM_PROMPT},
    ]
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "participant":
            messages.append({"role": "user", "content": content})
        elif role == "partner":
            messages.append({"role": "assistant", "content": content})
        elif role == "task":
            messages.append({"role": "user", "content": content})

    return messages


def _inject_explicit_cache_breakpoint(messages: list[dict[str, Any]]) -> None:
    """Add an explicit ``cache_control`` breakpoint on the last message.

    Automatic caching always places its breakpoint on the very last block of a
    request.  When the self-report questionnaire is appended *after* the
    conversation history, the automatic breakpoint lands on the questionnaire
    (new content every time) rather than on the conversation prefix that the
    prior turn already wrote to cache.  As a result the lookback never finds the
    prior entry and the entire prefix is re-written.

    By placing an explicit breakpoint on the last conversation message (the one
    right before the questionnaire), we ensure the cache system checks that
    exact position first.  Since the prior conversation turn wrote its automatic
    cache entry at the same position, the prefix hash matches and we get a
    cache read instead of a full write.

    The content is converted to ``[{"type": "text", ...}]`` array form only for
    this single message to attach the ``cache_control`` marker; the rest of the
    messages remain plain strings so that prefix identity is preserved.
    """
    if not messages:
        return
    last = messages[-1]
    content = last.get("content", "")
    if isinstance(content, str):
        last["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]


def _collect_self_report(
    client: OpenRouterClient,
    model_config: ModelConfig,
    turns: list[dict[str, Any]],
    turn_number: int,
) -> dict[str, Any]:
    """Collect CAARS self-report from the target model at a checkpoint.

    Sends a separate API call with the persona prompt + conversation history +
    self-report questionnaire.  An explicit cache breakpoint is placed on the
    last conversation message so the lookback finds the entry written by the
    preceding turn, avoiding a full cache re-write of the entire prefix.
    """
    target_messages = _build_target_messages(turns)
    _inject_explicit_cache_breakpoint(target_messages)
    target_messages.append(
        {
            "role": "user",
            "content": build_self_report_prompt(),
        }
    )

    # Use the same temperature and max_tokens as conversation turns so the
    # prompt cache entry written by the preceding turn can be reused.
    # OpenRouter/Anthropic includes these parameters in the cache key despite
    # Anthropic's docs not listing them as cache invalidators.
    result = client.chat(
        model=model_config.model_id,
        messages=target_messages,
        max_tokens=RESPONSE_MAX_TOKENS,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.effective_reasoning,
        provider=model_config.provider,
        cache_control=True,
    )

    return {
        "raw_response": result.content,
        "cost": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "cost_usd": result.usage.cost_usd,
            "elapsed_seconds": result.usage.elapsed_seconds,
        },
    }


def _build_participant_turn(
    result: CompletionResult,
    msg_number: int,
    exchange: int,
) -> dict[str, Any]:
    """Build a participant turn dict from a completion result."""
    turn_data: dict[str, Any] = {
        "turn": msg_number,
        "exchange": exchange,
        "role": "participant",
        "content": result.content,
        "cost_usd": result.usage.cost_usd,
        "tokens": result.usage.prompt_tokens + result.usage.completion_tokens,
        "prompt_tokens": result.usage.prompt_tokens,
        "cached_tokens": result.usage.cached_tokens,
        "cache_write_tokens": result.usage.cache_write_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if result.reasoning_details:
        turn_data["reasoning_details"] = result.reasoning_details
    elif result.reasoning_content:
        turn_data["reasoning_content"] = result.reasoning_content
    return turn_data


def _build_partner_turn(
    result: CompletionResult,
    msg_number: int,
    exchange: int,
) -> dict[str, Any]:
    """Build a partner turn dict from a completion result."""
    return {
        "turn": msg_number,
        "exchange": exchange,
        "role": "partner",
        "content": result.content,
        "cost_usd": result.usage.cost_usd,
        "prompt_tokens": result.usage.prompt_tokens,
        "cached_tokens": result.usage.cached_tokens,
        "cache_write_tokens": result.usage.cache_write_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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


def _run_exchange_loop(
    client: OpenRouterClient,
    model_config: ModelConfig,
    turns: list[dict[str, Any]],
    config_dir: str,
    run_number: int,
    conv_id: str,
    max_turns: int,
    checkpoints: list[int],
) -> float:
    """Run the partner/participant exchange loop. Returns total cost accumulated."""
    total_cost = 0.0
    msg_number = 2
    for exchange_round in range(1, max_turns + 1):
        partner_result = client.chat(
            model=PARTNER_MODEL,
            messages=_build_partner_messages(turns),
            max_tokens=PARTNER_MAX_TOKENS,
            temperature=PARTNER_TEMPERATURE,
            cache_control=True,
        )
        total_cost += partner_result.usage.cost_usd
        partner_turn = _build_partner_turn(partner_result, msg_number, exchange_round)
        turns.append(partner_turn)
        append_turn(config_dir, run_number, conv_id, partner_turn)
        _print_turn_with_cache(
            f"Turn {exchange_round:2d}",
            "partner",
            partner_result.content,
            partner_result.usage.prompt_tokens,
            partner_result.usage.cached_tokens,
            partner_result.usage.cache_write_tokens,
        )
        msg_number += 1

        target_result = client.chat(
            model=model_config.model_id,
            messages=_build_target_messages(turns),
            max_tokens=RESPONSE_MAX_TOKENS,
            temperature=model_config.effective_temperature,
            reasoning_effort=model_config.effective_reasoning,
            provider=model_config.provider,
            cache_control=True,
        )
        total_cost += target_result.usage.cost_usd
        pturn = _build_participant_turn(target_result, msg_number, exchange_round)
        turns.append(pturn)
        append_turn(config_dir, run_number, conv_id, pturn)
        _print_turn_with_cache(
            f"Turn {exchange_round:2d}",
            "participant",
            target_result.content,
            target_result.usage.prompt_tokens,
            target_result.usage.cached_tokens,
            target_result.usage.cache_write_tokens,
        )
        msg_number += 1

        if exchange_round in checkpoints:
            console.print(f"    ── Checkpoint at exchange {exchange_round} ──")
            sr_data = _collect_self_report(client, model_config, turns, exchange_round)
            total_cost += sr_data["cost"]["cost_usd"]
            checkpoint_data = {
                "turn": exchange_round,
                "self_report": sr_data,
                "conversation_snapshot_turns": len(turns),
            }
            save_checkpoint(config_dir, run_number, conv_id, exchange_round, checkpoint_data)
            console.print(f"    ── Self-report collected (${sr_data['cost']['cost_usd']:.4f}) ──")

    return total_cost


def run_conversation(
    client: OpenRouterClient,
    model_config: ModelConfig,
    run_number: int,
    conversation_id: str | None = None,
    *,
    max_turns: int = MAX_TURNS,
    checkpoint_turns: list[int] | None = None,
) -> dict[str, Any]:
    """Run a single multi-turn persona conversation."""
    config_dir = model_config.config_dir_name
    conv_id = conversation_id or _generate_conversation_id()
    checkpoints = checkpoint_turns or CHECKPOINT_TURNS

    cached = _check_cached_conversation(config_dir, run_number, conv_id, model_config, max_turns)
    if cached is not None:
        return cached

    turns: list[dict[str, Any]] = []
    console.print(
        Text(f"\n  [{model_config.label}] Starting conversation {conv_id} (run {run_number})", style="bold cyan")
    )

    task_turn: dict[str, Any] = {
        "turn": 0,
        "role": "task",
        "content": WORKDAY_TASK,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    turns.append(task_turn)
    append_turn(config_dir, run_number, conv_id, task_turn)

    result = client.chat(
        model=model_config.model_id,
        messages=_build_target_messages(turns),
        max_tokens=RESPONSE_MAX_TOKENS,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.effective_reasoning,
        provider=model_config.provider,
        cache_control=True,
    )
    initial_turn = _build_participant_turn(result, 1, 0)
    turns.append(initial_turn)
    append_turn(config_dir, run_number, conv_id, initial_turn)
    _print_turn_with_cache(
        "Init  ",
        "participant",
        result.content,
        result.usage.prompt_tokens,
        result.usage.cached_tokens,
        result.usage.cache_write_tokens,
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
    )
    total_cost_usd = result.usage.cost_usd + loop_cost

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
) -> list[dict[str, Any]]:
    """Run all conversations for a model (n_runs conversations).

    Returns list of conversation result dicts.
    """
    results = []
    for run_num in range(1, n_runs + 1):
        conv_id = _generate_conversation_id()
        result = run_conversation(
            client=client,
            model_config=model_config,
            run_number=run_num,
            conversation_id=conv_id,
            max_turns=max_turns,
            checkpoint_turns=checkpoint_turns,
        )
        results.append(result)
    return results
