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
from typing import Any

from rich.console import Console
from rich.text import Text

from src.cache import (
    append_turn,
    conversation_exists,
    load_conversation,
    save_checkpoint,
    save_conversation,
)
from src.config import (
    CHECKPOINT_TURNS,
    MAX_TURNS,
    PARTNER_MAX_TOKENS,
    PARTNER_MODEL,
    PARTNER_TEMPERATURE,
    RESPONSE_MAX_TOKENS,
    SELF_REPORT_MAX_TOKENS,
    ModelConfig,
)
from src.openrouter_client import CompletionResult, OpenRouterClient
from src.prompts import (
    HIGH_ADHD_PERSONA,
    PARTNER_SYSTEM_PROMPT,
    WORKDAY_TASK,
    build_self_report_prompt,
)

log = logging.getLogger(__name__)
console = Console()


def _generate_conversation_id() -> str:
    """Generate a short unique conversation ID."""
    return uuid.uuid4().hex[:12]


def _build_target_messages(
    turns: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the message list for the target model from conversation turns.

    The target model sees: system prompt (persona) + all turns as user/assistant.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": HIGH_ADHD_PERSONA},
    ]
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "participant":
            messages.append({"role": "assistant", "content": content})
        elif role == "partner":
            messages.append({"role": "user", "content": content})
        elif role == "task":
            messages.append({"role": "user", "content": content})
    return messages


def _build_partner_messages(
    turns: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the message list for the neutral partner.

    The partner sees: system prompt + all turns, with participant as user
    and partner as assistant.
    """
    messages: list[dict[str, str]] = [
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


def _collect_self_report(
    client: OpenRouterClient,
    model_config: ModelConfig,
    turns: list[dict[str, Any]],
    turn_number: int,
) -> dict[str, Any]:
    """Collect CAARS self-report from the target model at a checkpoint.

    Sends a separate API call with the persona prompt + conversation history +
    self-report questionnaire.
    """
    target_messages = _build_target_messages(turns)
    target_messages.append({
        "role": "user",
        "content": build_self_report_prompt(),
    })

    result = client.chat(
        model=model_config.model_id,
        messages=target_messages,
        max_tokens=SELF_REPORT_MAX_TOKENS,
        temperature=0.3,
        reasoning_effort=model_config.effective_reasoning,
        provider=model_config.provider,
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


def run_conversation(
    client: OpenRouterClient,
    model_config: ModelConfig,
    run_number: int,
    conversation_id: str | None = None,
    *,
    max_turns: int = MAX_TURNS,
    checkpoint_turns: list[int] | None = None,
) -> dict[str, Any]:
    """Run a single multi-turn persona conversation.

    Args:
        client: OpenRouter API client.
        model_config: Target model configuration.
        run_number: Run number for cache path.
        conversation_id: Optional conversation ID (auto-generated if None).
        max_turns: Maximum number of conversation turns.
        checkpoint_turns: Turn numbers at which to collect self-reports.

    Returns:
        Dict with conversation metadata, turns, and checkpoint data.
    """
    config_dir = model_config.config_dir_name
    conv_id = conversation_id or _generate_conversation_id()
    checkpoints = checkpoint_turns or CHECKPOINT_TURNS

    if conversation_exists(config_dir, run_number, conv_id):
        existing_turns = load_conversation(config_dir, run_number, conv_id)
        if existing_turns and len(existing_turns) >= max_turns + 1:
            log.info("Conversation %s already complete (%d turns)", conv_id, len(existing_turns))
            return {
                "conversation_id": conv_id,
                "model_config": model_config,
                "run": run_number,
                "turns": existing_turns,
                "status": "cached",
            }

    turns: list[dict[str, Any]] = []
    total_cost_usd = 0.0

    console.print(
        Text(f"\n  [{model_config.label}] Starting conversation {conv_id} (run {run_number})", style="bold cyan")
    )

    # Turn 0: inject workday task
    task_turn = {
        "turn": 0,
        "role": "task",
        "content": WORKDAY_TASK,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    turns.append(task_turn)
    append_turn(config_dir, run_number, conv_id, task_turn)

    # Turn 1: target model responds to workday task
    target_messages = _build_target_messages(turns)
    result = client.chat(
        model=model_config.model_id,
        messages=target_messages,
        max_tokens=RESPONSE_MAX_TOKENS,
        temperature=model_config.effective_temperature,
        reasoning_effort=model_config.effective_reasoning,
        provider=model_config.provider,
    )
    total_cost_usd += result.usage.cost_usd

    participant_turn = {
        "turn": 1,
        "role": "participant",
        "content": result.content,
        "cost_usd": result.usage.cost_usd,
        "tokens": result.usage.prompt_tokens + result.usage.completion_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    turns.append(participant_turn)
    append_turn(config_dir, run_number, conv_id, participant_turn)
    console.print(f"    Turn  1 [participant]: {result.content[:80]}...")

    turn_number = 2
    while turn_number <= max_turns:
        # Partner generates a follow-up question
        partner_messages = _build_partner_messages(turns)
        partner_result = client.chat(
            model=PARTNER_MODEL,
            messages=partner_messages,
            max_tokens=PARTNER_MAX_TOKENS,
            temperature=PARTNER_TEMPERATURE,
        )
        total_cost_usd += partner_result.usage.cost_usd

        partner_turn_data = {
            "turn": turn_number,
            "role": "partner",
            "content": partner_result.content,
            "cost_usd": partner_result.usage.cost_usd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        turns.append(partner_turn_data)
        append_turn(config_dir, run_number, conv_id, partner_turn_data)
        console.print(f"    Turn {turn_number:2d} [partner]:     {partner_result.content[:80]}...")

        turn_number += 1
        if turn_number > max_turns:
            break

        # Target model responds
        target_messages = _build_target_messages(turns)
        target_result = client.chat(
            model=model_config.model_id,
            messages=target_messages,
            max_tokens=RESPONSE_MAX_TOKENS,
            temperature=model_config.effective_temperature,
            reasoning_effort=model_config.effective_reasoning,
            provider=model_config.provider,
        )
        total_cost_usd += target_result.usage.cost_usd

        participant_turn_data = {
            "turn": turn_number,
            "role": "participant",
            "content": target_result.content,
            "cost_usd": target_result.usage.cost_usd,
            "tokens": target_result.usage.prompt_tokens + target_result.usage.completion_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        turns.append(participant_turn_data)
        append_turn(config_dir, run_number, conv_id, participant_turn_data)
        console.print(f"    Turn {turn_number:2d} [participant]: {target_result.content[:80]}...")

        # Checkpoint: collect self-report at designated turns
        if turn_number in checkpoints:
            console.print(f"    ── Checkpoint at turn {turn_number} ──")
            sr_data = _collect_self_report(client, model_config, turns, turn_number)
            total_cost_usd += sr_data["cost"]["cost_usd"]

            checkpoint_data = {
                "turn": turn_number,
                "self_report": sr_data,
                "conversation_snapshot_turns": len(turns),
            }
            save_checkpoint(config_dir, run_number, conv_id, turn_number, checkpoint_data)
            console.print(f"    ── Self-report collected (${sr_data['cost']['cost_usd']:.4f}) ──")

        turn_number += 1

    console.print(
        Text(f"  [{model_config.label}] Conversation {conv_id} complete. "
             f"Total cost: ${total_cost_usd:.4f}", style="green")
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
