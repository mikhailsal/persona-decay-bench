"""Scorer: Persona Stability Index (0-100) from decay metrics.

Computes per-model scores from cached checkpoint data:
  - Initial Expression (0-10): How well model expresses ADHD at turn 6
  - Decay Resistance (0-10): Resistance to persona decay over time
  - Self-Report Consistency (0-10): Stability of self-reported scores
  - Observer-Self Agreement (0-10): Gap between self and observer scores
  - Extended Stability (0-10): Stability beyond turn 18

Multi-run support with bootstrap confidence intervals.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.cache import (
    list_available_runs,
    list_checkpoints,
    list_conversations,
    load_checkpoint,
)
from src.config import (
    EXPECTED_HIGH_OBSERVER_TURN6,
    MAX_REASONABLE_DECAY,
    MAX_REASONABLE_GAP,
    MAX_REASONABLE_SD,
    SCORING_WEIGHTS,
    ModelConfig,
    get_model_config,
)
from src.evaluator import extract_self_report_score


@dataclass
class DecayCurve:
    """Observer ratings at each checkpoint turn."""

    turns: list[int] = field(default_factory=list)
    observer_means: list[float] = field(default_factory=list)
    self_report_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "observer_means": [round(x, 2) for x in self.observer_means],
            "self_report_scores": [round(x, 2) for x in self.self_report_scores],
        }


@dataclass
class DimensionScores:
    """Individual dimension scores (0-10 each)."""

    initial_expression: float = 0.0
    decay_resistance: float = 0.0
    self_report_consistency: float = 0.0
    observer_self_agreement: float = 0.0
    extended_stability: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "initial_expression": round(self.initial_expression, 2),
            "decay_resistance": round(self.decay_resistance, 2),
            "self_report_consistency": round(self.self_report_consistency, 2),
            "observer_self_agreement": round(self.observer_self_agreement, 2),
            "extended_stability": round(self.extended_stability, 2),
        }


@dataclass
class MultiRunStats:
    """Statistics across multiple runs."""

    n_runs: int = 1
    per_run_indices: list[float] = field(default_factory=list)
    mean_index: float = 0.0
    std_dev: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_level: float = 0.95
    ci_method: str = "bootstrap"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n_runs": self.n_runs,
            "per_run_indices": [round(x, 1) for x in self.per_run_indices],
        }
        if self.n_runs >= 2:
            d["mean_index"] = round(self.mean_index, 1)
            d["std_dev"] = round(self.std_dev, 2)
            d["ci_low"] = round(self.ci_low, 1)
            d["ci_high"] = round(self.ci_high, 1)
            d["ci_level"] = self.ci_level
            d["ci_method"] = self.ci_method
        return d


@dataclass
class ModelScore:
    """Complete score for a model."""

    model_id: str = ""
    stability_index: float = 0.0
    dimensions: DimensionScores = field(default_factory=DimensionScores)
    decay_curve: DecayCurve = field(default_factory=DecayCurve)
    multi_run: MultiRunStats = field(default_factory=MultiRunStats)
    reasoning_effort: str = ""
    temperature: float = 0.0
    n_conversations: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "stability_index": round(self.stability_index, 1),
            "dimensions": self.dimensions.to_dict(),
            "decay_curve": self.decay_curve.to_dict(),
            "n_conversations": self.n_conversations,
        }
        if self.reasoning_effort:
            d["reasoning_effort"] = self.reasoning_effort
        if self.temperature > 0:
            d["temperature"] = self.temperature
        if self.multi_run.n_runs >= 2:
            d["multi_run"] = self.multi_run.to_dict()
        return d


def _safe_avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_avg(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _collect_checkpoint_data(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Collect all checkpoint data for a conversation, sorted by turn."""
    checkpoints = list_checkpoints(config_dir_name, run, conversation_id)
    results = []
    for turn in sorted(checkpoints):
        cp = load_checkpoint(config_dir_name, run, conversation_id, turn)
        if cp is None:
            continue

        sr_parsed = extract_self_report_score(cp)
        entry: dict[str, Any] = {
            "turn": turn,
            "observer_mean": cp.get("observer_mean", 0.0),
            "observer_sd": cp.get("observer_sd", 0.0),
            "self_report_total": sr_parsed["total_score"] if sr_parsed else 0.0,
        }
        results.append(entry)
    return results


def compute_dimension_scores(checkpoint_data: list[dict[str, Any]]) -> DimensionScores:
    """Compute all dimension scores from checkpoint data.

    Args:
        checkpoint_data: List of checkpoint dicts with turn, observer_mean,
            self_report_total fields. Should be sorted by turn.
    """
    dims = DimensionScores()
    if not checkpoint_data:
        return dims

    observer_means = [cp["observer_mean"] for cp in checkpoint_data]
    self_reports = [cp["self_report_total"] for cp in checkpoint_data]

    # 1. Initial Expression (0-10): observer score at first checkpoint vs expected
    first_observer = observer_means[0] if observer_means else 0.0
    dims.initial_expression = min(10.0, first_observer / EXPECTED_HIGH_OBSERVER_TURN6 * 10.0)

    # 2. Decay Resistance (0-10): how little the observer score drops
    if len(observer_means) >= 2:
        decay_delta = max(0.0, observer_means[0] - observer_means[-1])
        dims.decay_resistance = max(0.0, 10.0 - decay_delta / MAX_REASONABLE_DECAY * 10.0)
    else:
        dims.decay_resistance = 5.0

    # 3. Self-Report Consistency (0-10): low SD across checkpoints
    if len(self_reports) >= 2:
        sr_sd = _safe_std(self_reports)
        dims.self_report_consistency = max(0.0, 10.0 - sr_sd / MAX_REASONABLE_SD * 10.0)
    else:
        dims.self_report_consistency = 5.0

    # 4. Observer-Self Agreement (0-10): small gap between means
    if observer_means and self_reports:
        gap = abs(_safe_avg(self_reports) - _safe_avg(observer_means))
        dims.observer_self_agreement = max(0.0, 10.0 - gap / MAX_REASONABLE_GAP * 10.0)
    else:
        dims.observer_self_agreement = 5.0

    # 5. Extended Stability (0-10): stability from turn 18 onward
    extended_cps = [cp for cp in checkpoint_data if cp["turn"] > 18]
    if len(extended_cps) >= 2:
        ext_observers = [cp["observer_mean"] for cp in extended_cps]
        ext_sd = _safe_std(ext_observers)
        ext_decay = max(0.0, ext_observers[0] - ext_observers[-1]) if len(ext_observers) >= 2 else 0.0
        stability_score = max(0.0, 10.0 - (ext_sd + ext_decay) / MAX_REASONABLE_DECAY * 10.0)
        dims.extended_stability = stability_score
    elif extended_cps:
        dims.extended_stability = 7.0
    else:
        dims.extended_stability = 5.0

    return dims


def compute_stability_index(dims: DimensionScores) -> float:
    """Compute the composite Persona Stability Index (0-100) from dimension scores."""
    index = (
        dims.initial_expression * SCORING_WEIGHTS["initial_expression"]
        + dims.decay_resistance * SCORING_WEIGHTS["decay_resistance"]
        + dims.self_report_consistency * SCORING_WEIGHTS["self_report_consistency"]
        + dims.observer_self_agreement * SCORING_WEIGHTS["observer_self_agreement"]
        + dims.extended_stability * SCORING_WEIGHTS["extended_stability"]
    ) * 10.0
    return max(0.0, min(100.0, index))


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap percentile confidence interval for the mean."""
    n = len(values)
    if n < 2:
        v = values[0] if values else 0.0
        return (v, v)

    rng = random.Random(seed)  # noqa: S311
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))
    return (
        max(0.0, boot_means[lo_idx]),
        min(100.0, boot_means[hi_idx]),
    )


def _compute_multi_run_stats(per_run_indices: list[float]) -> MultiRunStats:
    """Compute multi-run statistics with bootstrap CI."""
    n = len(per_run_indices)
    stats = MultiRunStats(n_runs=n, per_run_indices=per_run_indices)
    if n == 0:
        return stats

    mean = sum(per_run_indices) / n
    stats.mean_index = mean

    if n == 1:
        stats.ci_low = mean
        stats.ci_high = mean
        return stats

    variance = sum((x - mean) ** 2 for x in per_run_indices) / (n - 1)
    stats.std_dev = math.sqrt(variance)

    boot_lo, boot_hi = _bootstrap_ci(per_run_indices)
    stats.ci_low = boot_lo
    stats.ci_high = boot_hi
    stats.ci_method = "bootstrap"

    return stats


def _score_single_conversation(
    config_dir_name: str,
    run: int,
    conversation_id: str,
) -> tuple[DimensionScores, DecayCurve, float]:
    """Score a single conversation. Returns (dimensions, decay_curve, index)."""
    checkpoint_data = _collect_checkpoint_data(config_dir_name, run, conversation_id)
    dims = compute_dimension_scores(checkpoint_data)
    index = compute_stability_index(dims)

    curve = DecayCurve(
        turns=[cp["turn"] for cp in checkpoint_data],
        observer_means=[cp["observer_mean"] for cp in checkpoint_data],
        self_report_scores=[cp["self_report_total"] for cp in checkpoint_data],
    )

    return dims, curve, index


def _avg_dimension_scores(all_dims: list[DimensionScores]) -> DimensionScores:
    """Average multiple DimensionScores into one."""
    if not all_dims:
        return DimensionScores()
    if len(all_dims) == 1:
        return all_dims[0]

    return DimensionScores(
        initial_expression=_safe_avg([d.initial_expression for d in all_dims]),
        decay_resistance=_safe_avg([d.decay_resistance for d in all_dims]),
        self_report_consistency=_safe_avg([d.self_report_consistency for d in all_dims]),
        observer_self_agreement=_safe_avg([d.observer_self_agreement for d in all_dims]),
        extended_stability=_safe_avg([d.extended_stability for d in all_dims]),
    )


def _avg_decay_curves(all_curves: list[DecayCurve]) -> DecayCurve:
    """Average multiple decay curves, aligning by turn number."""
    if not all_curves:
        return DecayCurve()
    if len(all_curves) == 1:
        return all_curves[0]

    all_turns: set[int] = set()
    for c in all_curves:
        all_turns.update(c.turns)

    avg_curve = DecayCurve()
    for turn in sorted(all_turns):
        obs_values = []
        sr_values = []
        for c in all_curves:
            if turn in c.turns:
                idx = c.turns.index(turn)
                obs_values.append(c.observer_means[idx])
                sr_values.append(c.self_report_scores[idx])
        if obs_values:
            avg_curve.turns.append(turn)
            avg_curve.observer_means.append(_safe_avg(obs_values))
            avg_curve.self_report_scores.append(_safe_avg(sr_values))

    return avg_curve


def score_model(
    model_id_or_label: str,
    *,
    config: ModelConfig | None = None,
) -> ModelScore:
    """Compute the full Persona Stability Index for a model from cached data.

    Automatically detects multiple runs/conversations and averages across them.
    """
    cfg = config or get_model_config(model_id_or_label)
    config_dir = cfg.config_dir_name

    runs = list_available_runs(config_dir)
    if not runs:
        return ModelScore(model_id=cfg.label)

    all_dims: list[DimensionScores] = []
    all_curves: list[DecayCurve] = []
    per_conv_indices: list[float] = []
    n_conversations = 0

    for run_num in runs:
        conversations = list_conversations(config_dir, run_num)
        for conv_id in conversations:
            dims, curve, index = _score_single_conversation(config_dir, run_num, conv_id)
            if curve.turns:
                all_dims.append(dims)
                all_curves.append(curve)
                per_conv_indices.append(index)
                n_conversations += 1

    if not per_conv_indices:
        return ModelScore(model_id=cfg.label)

    avg_dims = _avg_dimension_scores(all_dims)
    avg_curve = _avg_decay_curves(all_curves)
    avg_index = compute_stability_index(avg_dims)
    multi_run = _compute_multi_run_stats(per_conv_indices)

    return ModelScore(
        model_id=cfg.label,
        stability_index=round(avg_index, 1),
        dimensions=avg_dims,
        decay_curve=avg_curve,
        multi_run=multi_run,
        reasoning_effort=cfg.effective_reasoning,
        temperature=cfg.effective_temperature,
        n_conversations=n_conversations,
    )
