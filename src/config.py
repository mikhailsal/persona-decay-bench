"""Configuration: paths, constants, model registry, YAML loader, API key loading."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
ENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "configs" / "models.yaml"

# ---------------------------------------------------------------------------
# Conversation parameters
# ---------------------------------------------------------------------------
MAX_TURNS = 36
CHECKPOINT_TURNS = [6, 12, 18, 24, 30, 36]
RUNS_PER_MODEL = 5

# ---------------------------------------------------------------------------
# Token / generation limits
# ---------------------------------------------------------------------------
RESPONSE_MAX_TOKENS = 2048
RESPONSE_TEMPERATURE = 1.0
PARTNER_MAX_TOKENS = 512
PARTNER_TEMPERATURE = 1.0
SELF_REPORT_MAX_TOKENS = 1024
OBSERVER_MAX_TOKENS = 1024
OBSERVER_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
PARTNER_MODEL = "google/gemini-3.1-flash-lite-preview"
OBSERVER_MODEL = "google/gemini-3-flash-preview"
OBSERVER_CALLS = 3

# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_APP_NAME = "persona-decay-bench"
OPENROUTER_APP_URL = "https://github.com/mikhailsal/persona-decay-bench"
API_CALL_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Reasoning model support
# ---------------------------------------------------------------------------
REASONING_EFFORT_DEFAULT = "low"

REASONING_EFFORT_BY_PREFIX: dict[str, str] = {
    "google/gemini-3-pro": "low",
    "google/gemini-3.1-pro": "low",
    "google/": "none",
    "qwen/": "none",
    "openai/": "low",
    "anthropic/": "none",
    "x-ai/": "low",
    "moonshotai/": "none",
}


def get_reasoning_effort(model_id: str) -> str:
    """Return the default reasoning effort for a model based on its ID prefix."""
    best_match = ""
    best_effort = REASONING_EFFORT_DEFAULT
    for prefix, effort in REASONING_EFFORT_BY_PREFIX.items():
        if model_id.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_effort = effort
    return best_effort


# ---------------------------------------------------------------------------
# Scoring weights for Persona Stability Index
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "initial_expression": 0.20,
    "decay_resistance": 0.40,
    "self_report_consistency": 0.15,
    "observer_self_agreement": 0.10,
    "extended_stability": 0.15,
}

# Paper reference values for calibration
EXPECTED_HIGH_OBSERVER_TURN6 = 17.5
MAX_REASONABLE_DECAY = 10.0
MAX_REASONABLE_SD = 5.0
MAX_REASONABLE_GAP = 15.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def model_id_to_slug(model_id: str) -> str:
    """Convert 'openai/gpt-5-nano' -> 'openai--gpt-5-nano'."""
    return model_id.replace("/", "--")


def slug_to_model_id(slug: str) -> str:
    """Convert 'openai--gpt-5-nano' -> 'openai/gpt-5-nano'."""
    return slug.replace("--", "/", 1)


def ensure_dirs() -> None:
    """Create cache and results directories if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env file once (idempotent thanks to dotenv internals)."""
    load_dotenv(ENV_PATH)


def get_openrouter_base_url() -> str:
    """Return the OpenRouter base URL from environment or the default."""
    _load_env()
    url = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    return url if url else _DEFAULT_OPENROUTER_BASE_URL


def load_api_key(*, required: bool = True) -> str:
    """Load the OpenRouter API key from environment or .env file.

    Checks OPENROUTER_KEY first, then OPENROUTER_API_KEY for compatibility.
    """
    _load_env()
    key = os.environ.get("OPENROUTER_KEY", "").strip()
    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    placeholder = key in ("", "sk-or-your-key-here")
    if placeholder and required:
        _log.error(
            "OpenRouter API key is not set.\n"
            "  Create a .env file at %s with:\n"
            "  OPENROUTER_KEY=your-key-here\n"
            "  Or export OPENROUTER_KEY / OPENROUTER_API_KEY as an environment variable.",
            ENV_PATH,
        )
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Model pricing
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """Per-token pricing for a model (in USD)."""

    prompt_price: float = 0.0
    completion_price: float = 0.0


# ---------------------------------------------------------------------------
# Per-model configuration registry
# ---------------------------------------------------------------------------


def generate_display_label(model_id: str, reasoning: str, temperature: float) -> str:
    """Auto-generate a display label: ``{name}@{reasoning}-t{temp}``."""
    name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    return f"{name}@{reasoning}-t{temperature}"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific benchmark entry.

    Each ModelConfig represents one row in the leaderboard.

    Attributes:
        model_id: The API model identifier.
        display_label: Human-readable label shown in the leaderboard.
        temperature: Response temperature override.
        reasoning_effort: Reasoning effort override.
        active: Whether this config should be included in default runs.
        provider: OpenRouter provider slug to pin requests to.
    """

    model_id: str
    display_label: str = ""
    temperature: float | None = None
    reasoning_effort: str | None = None
    active: bool = True
    provider: str | None = None

    @property
    def label(self) -> str:
        return self.display_label or self.model_id

    @property
    def effective_temperature(self) -> float:
        return self.temperature if self.temperature is not None else RESPONSE_TEMPERATURE

    @property
    def effective_reasoning(self) -> str:
        return self.reasoning_effort if self.reasoning_effort is not None else get_reasoning_effort(self.model_id)

    @property
    def config_dir_name(self) -> str:
        """Cache directory name: ``{slug}@{reasoning}-t{temp}``."""
        slug = model_id_to_slug(self.model_id)
        if self.provider:
            provider_tag = self.provider.replace("/", "-")
            return f"{slug}+{provider_tag}@{self.effective_reasoning}-t{self.effective_temperature}"
        return f"{slug}@{self.effective_reasoning}-t{self.effective_temperature}"


MODEL_CONFIGS: dict[str, ModelConfig] = {}


def register_config(cfg: ModelConfig) -> None:
    """Register a model configuration. Raises ValueError on duplicate labels."""
    label = cfg.label
    if label in MODEL_CONFIGS:
        raise ValueError(f"Duplicate model config label: {label!r}")
    MODEL_CONFIGS[label] = cfg


def get_model_config(label_or_model_id: str) -> ModelConfig:
    """Resolve a display label or raw model_id to a ModelConfig."""
    if label_or_model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[label_or_model_id]

    matches = [c for c in MODEL_CONFIGS.values() if c.model_id == label_or_model_id]
    if len(matches) == 1:
        return matches[0]

    return ModelConfig(model_id=label_or_model_id)


def get_config_by_dir_name(dir_name: str) -> ModelConfig | None:
    """Look up a ModelConfig by its ``config_dir_name``."""
    for cfg in MODEL_CONFIGS.values():
        if cfg.config_dir_name == dir_name:
            return cfg
    return None


# ---------------------------------------------------------------------------
# YAML configuration loader
# ---------------------------------------------------------------------------


def load_model_configs(path: Path | None = None) -> list[ModelConfig]:
    """Load model configurations from a YAML file and register them."""
    import yaml

    config_path = path or CONFIGS_PATH
    if not config_path.exists():
        return []

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not data or "models" not in data:
        return []

    configs: list[ModelConfig] = []
    for entry in data["models"]:
        model_id = entry["model_id"]
        temperature = float(entry["temperature"])
        reasoning = entry["reasoning_effort"]
        active = entry.get("active", True)
        provider = entry.get("provider") or None

        label = entry.get("display_label") or generate_display_label(
            model_id,
            reasoning,
            temperature,
        )

        cfg = ModelConfig(
            model_id=model_id,
            display_label=label,
            temperature=temperature,
            reasoning_effort=reasoning,
            active=active,
            provider=provider,
        )
        if cfg.label not in MODEL_CONFIGS:
            register_config(cfg)
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Auto-load configs from YAML on import
# ---------------------------------------------------------------------------

_yaml_configs_loaded = False


def _auto_load_configs() -> None:
    global _yaml_configs_loaded
    if _yaml_configs_loaded:
        return
    _yaml_configs_loaded = True
    if CONFIGS_PATH.exists():
        load_model_configs(CONFIGS_PATH)


_auto_load_configs()
