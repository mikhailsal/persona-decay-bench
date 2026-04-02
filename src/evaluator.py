"""Evaluator: observer assessment + self-report score extraction.

Observer assessment: 3 independent calls to gemini-3-flash-preview.
Self-report extraction: parses JSON responses from target models.
ICC computation: two-way random, single measures, absolute agreement.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from src.cache import (
    list_checkpoints,
    list_conversations,
    load_checkpoint,
    load_conversation,
    save_observer_scores,
)
from src.config import (
    OBSERVER_CALLS,
    OBSERVER_MAX_TOKENS,
    OBSERVER_MODEL,
    OBSERVER_TEMPERATURE,
    ModelConfig,
)
from src.prompts import CAARS_ITEMS, build_observer_prompt, format_conversation_for_observer

if TYPE_CHECKING:
    from src.openrouter_client import OpenRouterClient

log = logging.getLogger(__name__)


def parse_caars_scores(raw_response: str) -> dict[str, int] | None:
    """Parse a CAARS JSON response into item scores.

    Handles responses that may contain extra text around the JSON object.
    Returns None if parsing fails.
    """
    text = raw_response.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _validate_scores(data)
    except json.JSONDecodeError:
        pass

    # Extract JSON object from surrounding text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return _validate_scores(data)
        except json.JSONDecodeError:
            pass

    # Try finding JSON in markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1))
            if isinstance(data, dict):
                return _validate_scores(data)
        except json.JSONDecodeError:
            pass

    return None


def _validate_scores(data: dict[str, Any]) -> dict[str, int] | None:
    """Validate parsed scores: all expected item IDs present, values 0-3."""
    expected_ids = {item.id for item in CAARS_ITEMS}
    scores: dict[str, int] = {}

    for item_id in expected_ids:
        val = data.get(item_id)
        if val is None:
            continue
        try:
            int_val = int(val)
            scores[item_id] = max(0, min(3, int_val))
        except (ValueError, TypeError):
            continue

    if len(scores) < len(expected_ids) * 0.5:
        return None

    return scores


def compute_total_score(scores: dict[str, int]) -> int:
    """Sum all item scores for a total CAARS score."""
    return sum(scores.values())


def extract_self_report_score(checkpoint_data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and parse self-report scores from a checkpoint.

    Returns dict with 'items' (score dict), 'total_score', and 'raw_response'.
    """
    sr = checkpoint_data.get("self_report", {})
    raw = sr.get("raw_response", "")
    if not raw:
        return None

    scores = parse_caars_scores(raw)
    if scores is None:
        return None

    return {
        "items": scores,
        "total_score": compute_total_score(scores),
        "raw_response": raw,
    }


def _filter_turns_for_observer(
    conversation_turns: list[dict[str, Any]],
    up_to_turn: int,
) -> str:
    """Filter and format conversation turns for observer evaluation."""
    relevant = [
        t
        for t in conversation_turns
        if t.get("role") in ("participant", "partner") and (t.get("exchange") is None or t["exchange"] <= up_to_turn)
    ]
    formatted = [{"role": t["role"], "content": t["content"]} for t in relevant]
    return build_observer_prompt(format_conversation_for_observer(formatted))


def _run_single_observer_call(
    client: OpenRouterClient,
    observer_prompt: str,
    call_idx: int,
) -> tuple[dict[str, Any], float]:
    """Execute one observer call and return (rating_entry, cost)."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are an expert behavioral psychologist trained in ADHD assessment."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": observer_prompt, "cache_control": {"type": "ephemeral"}},
            ],
        },
    ]
    result = client.chat(
        model=OBSERVER_MODEL,
        messages=messages,
        max_tokens=OBSERVER_MAX_TOKENS,
        temperature=OBSERVER_TEMPERATURE,
        cache_control=True,
    )
    scores = parse_caars_scores(result.content)
    entry: dict[str, Any] = {
        "evaluator": f"{OBSERVER_MODEL}/run-{call_idx + 1}",
        "raw_response": result.content,
    }
    if scores is not None:
        entry["items"] = scores
        entry["total_score"] = compute_total_score(scores)
    else:
        log.warning("Observer call %d failed to parse scores", call_idx + 1)
        entry["items"] = {}
        entry["total_score"] = 0
        entry["parse_error"] = True
    return entry, result.usage.cost_usd


def run_observer_assessment(
    client: OpenRouterClient,
    conversation_turns: list[dict[str, Any]],
    up_to_turn: int,
    *,
    n_calls: int = OBSERVER_CALLS,
) -> dict[str, Any]:
    """Run observer assessment using multiple independent evaluator calls."""
    observer_prompt = _filter_turns_for_observer(conversation_turns, up_to_turn)

    ratings: list[dict[str, Any]] = []
    total_cost = 0.0
    for call_idx in range(n_calls):
        entry, cost = _run_single_observer_call(client, observer_prompt, call_idx)
        ratings.append(entry)
        total_cost += cost

    valid_totals = [r["total_score"] for r in ratings if not r.get("parse_error") and r["total_score"] > 0]
    observer_mean = float(np.mean(valid_totals)) if valid_totals else 0.0
    observer_sd = float(np.std(valid_totals, ddof=1)) if len(valid_totals) >= 2 else 0.0

    return {
        "observer_ratings": ratings,
        "observer_mean": round(observer_mean, 2),
        "observer_sd": round(observer_sd, 2),
        "observer_cost": {"cost_usd": total_cost, "n_calls": n_calls},
    }


def compute_icc(ratings_matrix: list[list[int]]) -> float:
    """Compute ICC(2,1) — two-way random, single measures, absolute agreement.

    Args:
        ratings_matrix: List of rater scores, each inner list is one rater's
            scores across all items. Shape: [n_raters x n_items].

    Returns:
        ICC value (float). Returns 0.0 if computation fails.
    """
    if not ratings_matrix or len(ratings_matrix) < 2:
        return 0.0

    try:
        data = np.array(ratings_matrix, dtype=float)
        n_raters, n_items = data.shape
        if n_items < 2:
            return 0.0

        grand_mean = np.mean(data)
        item_means = np.mean(data, axis=0)
        rater_means = np.mean(data, axis=1)

        # Sum of squares
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_rows = n_raters * np.sum((item_means - grand_mean) ** 2)  # between items
        ss_cols = n_items * np.sum((rater_means - grand_mean) ** 2)  # between raters
        ss_error = ss_total - ss_rows - ss_cols

        # Mean squares
        ms_rows = ss_rows / (n_items - 1)
        ms_cols = ss_cols / (n_raters - 1) if n_raters > 1 else 0.0
        ms_error = ss_error / ((n_items - 1) * (n_raters - 1)) if (n_items - 1) * (n_raters - 1) > 0 else 0.0

        # ICC(2,1)
        numerator = ms_rows - ms_error
        denominator = ms_rows + (n_raters - 1) * ms_error + n_raters * (ms_cols - ms_error) / n_items

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)
    except Exception:
        return 0.0


def evaluate_checkpoint(
    client: OpenRouterClient,
    model_config: ModelConfig,
    run_number: int,
    conversation_id: str,
    turn: int,
) -> dict[str, Any]:
    """Run full evaluation (self-report extraction + observer assessment) for a checkpoint.

    Returns dict with self_report, observer data, and ICC.
    """
    config_dir = model_config.config_dir_name
    checkpoint = load_checkpoint(config_dir, run_number, conversation_id, turn)
    if checkpoint is None:
        return {"error": f"No checkpoint found for turn {turn}"}

    # Extract self-report
    sr_parsed = extract_self_report_score(checkpoint)

    # Load conversation for observer assessment
    conversation_turns = load_conversation(config_dir, run_number, conversation_id)
    if not conversation_turns:
        return {"error": "No conversation data found"}

    # Run observer assessment if not already done
    if "observer_ratings" not in checkpoint or not checkpoint["observer_ratings"]:
        observer_data = run_observer_assessment(client, conversation_turns, turn)
        save_observer_scores(config_dir, run_number, conversation_id, turn, observer_data)
    else:
        observer_data = {
            "observer_ratings": checkpoint["observer_ratings"],
            "observer_mean": checkpoint.get("observer_mean", 0.0),
            "observer_sd": checkpoint.get("observer_sd", 0.0),
        }

    # Compute ICC from observer ratings
    valid_ratings = [r for r in observer_data["observer_ratings"] if not r.get("parse_error") and r.get("items")]

    icc = 0.0
    if len(valid_ratings) >= 2:
        item_ids = sorted(CAARS_ITEMS[0].id for _ in range(1))
        item_ids = sorted({item.id for item in CAARS_ITEMS})
        ratings_matrix = []
        for r in valid_ratings:
            rater_scores = [r["items"].get(iid, 0) for iid in item_ids]
            ratings_matrix.append(rater_scores)
        icc = compute_icc(ratings_matrix)

    return {
        "turn": turn,
        "self_report": sr_parsed,
        "observer_mean": observer_data.get("observer_mean", 0.0),
        "observer_sd": observer_data.get("observer_sd", 0.0),
        "observer_ratings": observer_data.get("observer_ratings", []),
        "icc": round(icc, 3),
    }


def evaluate_model(
    client: OpenRouterClient,
    model_config: ModelConfig,
    *,
    runs: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run observer evaluation for all conversations and checkpoints of a model.

    Returns list of evaluation results per conversation.
    """
    from src.cache import list_available_runs

    config_dir = model_config.config_dir_name
    available_runs = runs or list_available_runs(config_dir)
    if not available_runs:
        return []

    all_results: list[dict[str, Any]] = []

    for run_num in available_runs:
        conversations = list_conversations(config_dir, run_num)
        for conv_id in conversations:
            checkpoints = list_checkpoints(config_dir, run_num, conv_id)
            conv_results: dict[str, Any] = {
                "conversation_id": conv_id,
                "run": run_num,
                "checkpoints": {},
            }

            for turn in checkpoints:
                result = evaluate_checkpoint(
                    client,
                    model_config,
                    run_num,
                    conv_id,
                    turn,
                )
                conv_results["checkpoints"][turn] = result

            all_results.append(conv_results)

    return all_results
