"""Runner helpers: message builders, turn constructors, self-report collection.

Extracted from ``src.runner`` to keep the main orchestration module within
the project's 500-line file-size limit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rich.console import Console

from src.config import (
    SELF_REPORT_MAX_TOKENS,
    ModelConfig,
)
from src.prompts import (
    HIGH_ADHD_PERSONA,
    PARTNER_SYSTEM_PROMPT,
    build_self_report_prompt,
)

if TYPE_CHECKING:
    from src.openrouter_client import CompletionResult, OpenRouterClient

console = Console()


def print_turn_with_cache(
    label: str,
    role: str,
    content: str,
    prompt_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
    *,
    verbose: bool = False,
) -> None:
    """Print a turn line with cache status.

    When *verbose* is True the full response text is displayed in a
    Rich panel below the summary line so the user can read model output
    in real time.
    """
    cache_info = ""
    if cached_tokens > 0:
        pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
        cache_info = f" [green]CACHE READ {cached_tokens}/{prompt_tokens} ({pct:.0f}%)[/green]"
    elif cache_write_tokens > 0:
        cache_info = f" [yellow]CACHE WRITE {cache_write_tokens} tokens[/yellow]"
    elif prompt_tokens > 0:
        cache_info = f" [dim]no cache ({prompt_tokens} prompt tokens)[/dim]"

    preview = content if verbose else f"{content[:70]}..."
    console.print(f"    {label} [{role:11s}]: {preview}{cache_info}")

    if verbose and len(content) > 70:
        from rich.panel import Panel

        console.print(Panel(content, title=f"{role}", border_style="dim", expand=False, width=100))


def build_target_messages(
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


def build_partner_messages(
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


def inject_explicit_cache_breakpoint(messages: list[dict[str, Any]]) -> None:
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


def collect_self_report(
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
    target_messages = build_target_messages(turns)
    inject_explicit_cache_breakpoint(target_messages)
    target_messages.append({"role": "user", "content": build_self_report_prompt()})

    result = client.chat(
        model=model_config.model_id,
        messages=target_messages,
        max_tokens=SELF_REPORT_MAX_TOKENS,
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


def build_participant_turn(
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


def build_partner_turn(
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
