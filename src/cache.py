"""Cache: save/load conversations as JSONL, checkpoint JSON files, path construction.

Cache structure:
  cache/{config_dir_name}/run_{N}/high/{conversation_id}/
    conversation.jsonl          — one JSON object per line (turn-by-turn)
    checkpoint_turn_06.json     — self-report + observer assessments at turn 6
    checkpoint_turn_12.json     — ... at turn 12
    ...
    checkpoint_turn_36.json     — ... at turn 36
"""

from __future__ import annotations

import contextlib
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.config import CACHE_DIR

if TYPE_CHECKING:
    from pathlib import Path


def _conversation_dir(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> Path:
    """Build path to a conversation directory."""
    return CACHE_DIR / config_dir_name / f"run_{run}" / "high" / conversation_id


def _conversation_path(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> Path:
    """Build path to the conversation JSONL file."""
    return _conversation_dir(config_dir_name, run, conversation_id) / "conversation.jsonl"


def _checkpoint_path(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turn: int,
) -> Path:
    """Build path to a checkpoint JSON file."""
    return _conversation_dir(config_dir_name, run, conversation_id) / f"checkpoint_turn_{turn:02d}.json"


# ---------------------------------------------------------------------------
# Conversation JSONL
# ---------------------------------------------------------------------------


def append_turn(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turn_data: dict[str, Any],
) -> Path:
    """Append a single turn to the conversation JSONL file."""
    path = _conversation_path(config_dir_name, run, conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(turn_data, ensure_ascii=False) + "\n")
    return path


def load_conversation(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Load all turns from a conversation JSONL file."""
    path = _conversation_path(config_dir_name, run, conversation_id)
    if not path.exists():
        return []
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return turns


def save_conversation(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turns: list[dict[str, Any]],
) -> Path:
    """Save a full conversation (overwriting any existing JSONL)."""
    path = _conversation_path(config_dir_name, run, conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    return path


def conversation_exists(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> bool:
    """Check whether a conversation JSONL exists and has content."""
    path = _conversation_path(config_dir_name, run, conversation_id)
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Checkpoint JSON
# ---------------------------------------------------------------------------


def save_checkpoint(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turn: int,
    data: dict[str, Any],
) -> Path:
    """Save a checkpoint (self-report + observer assessments)."""
    path = _checkpoint_path(config_dir_name, run, conversation_id, turn)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.setdefault("metadata", {})
    data["metadata"]["turn"] = turn
    data["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_checkpoint(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turn: int,
) -> dict[str, Any] | None:
    """Load a checkpoint, or None if it doesn't exist."""
    path = _checkpoint_path(config_dir_name, run, conversation_id, turn)
    if not path.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return result


def checkpoint_exists(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turn: int,
) -> bool:
    """Check whether a checkpoint file exists."""
    return _checkpoint_path(config_dir_name, run, conversation_id, turn).exists()


def list_checkpoints(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> list[int]:
    """List all checkpoint turn numbers for a conversation."""
    conv_dir = _conversation_dir(config_dir_name, run, conversation_id)
    if not conv_dir.exists():
        return []
    turns = []
    for path in sorted(conv_dir.glob("checkpoint_turn_*.json")):
        m = re.match(r"checkpoint_turn_(\d+)\.json", path.name)
        if m:
            turns.append(int(m.group(1)))
    return sorted(turns)


def save_observer_scores(
    config_dir_name: str,
    run: int,
    conversation_id: str,
    turn: int,
    observer_data: dict[str, Any],
) -> None:
    """Add observer scores to an existing checkpoint."""
    path = _checkpoint_path(config_dir_name, run, conversation_id, turn)
    if not path.exists():
        save_checkpoint(config_dir_name, run, conversation_id, turn, observer_data)
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}

    data["observer_ratings"] = observer_data.get("observer_ratings", [])
    data["observer_mean"] = observer_data.get("observer_mean")
    data["observer_sd"] = observer_data.get("observer_sd")
    if "observer_cost" in observer_data:
        data["observer_cost"] = observer_data["observer_cost"]

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_available_runs(config_dir_name: str) -> list[int]:
    """List all run numbers for a config."""
    model_dir = CACHE_DIR / config_dir_name
    if not model_dir.exists():
        return []
    runs: list[int] = []
    for d in sorted(model_dir.iterdir()):
        if d.is_dir():
            m = re.match(r"^run_(\d+)$", d.name)
            if m:
                runs.append(int(m.group(1)))
    return sorted(runs)


def list_conversations(
    config_dir_name: str,
    run: int,
) -> list[str]:
    """List all conversation IDs for a given config + run."""
    high_dir = CACHE_DIR / config_dir_name / f"run_{run}" / "high"
    if not high_dir.exists():
        return []
    return sorted(d.name for d in high_dir.iterdir() if d.is_dir() and (d / "conversation.jsonl").exists())


def list_all_cached_models() -> list[str]:
    """List all config dir names that have cached data."""
    if not CACHE_DIR.exists():
        return []
    return sorted(d.name for d in CACHE_DIR.iterdir() if d.is_dir() and "@" in d.name)


# ---------------------------------------------------------------------------
# Cost aggregation
# ---------------------------------------------------------------------------


def sum_run_total_cost_usd(config_dir_name: str, run: int = 1) -> float:
    """Sum all costs in checkpoint files for a run."""
    root = CACHE_DIR / config_dir_name / f"run_{run}"
    if not root.is_dir():
        return 0.0
    total = 0.0
    for path in root.rglob("checkpoint_turn_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for cost_key in ("self_report_cost", "observer_cost"):
            block = data.get(cost_key)
            if isinstance(block, dict) and block.get("cost_usd") is not None:
                try:
                    total += float(block["cost_usd"])
                except (TypeError, ValueError):
                    continue
    return total


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------


def clear_all_cache() -> int:
    """Clear all cached data. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for path in CACHE_DIR.rglob("*"):
        if path.is_file():
            path.unlink()
            count += 1
    for d in sorted(CACHE_DIR.rglob("*"), reverse=True):
        if d.is_dir():
            with contextlib.suppress(OSError):
                d.rmdir()
    return count


def clear_observer_scores() -> int:
    """Clear only observer scores from checkpoints, keeping self-reports."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for path in CACHE_DIR.rglob("checkpoint_turn_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            changed = False
            for key in ("observer_ratings", "observer_mean", "observer_sd", "observer_cost"):
                if key in data:
                    del data[key]
                    changed = True
            if changed:
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                count += 1
        except (json.JSONDecodeError, OSError):
            continue
    return count
