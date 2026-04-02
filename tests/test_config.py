"""Tests for config.py: paths, constants, model registry, YAML loader, API key loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config import (
    CACHE_DIR,
    CHECKPOINT_TURNS,
    CONFIGS_PATH,
    MAX_TURNS,
    MODEL_CONFIGS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SCORING_WEIGHTS,
    ModelConfig,
    ModelPricing,
    ensure_dirs,
    generate_display_label,
    get_config_by_dir_name,
    get_model_config,
    get_reasoning_effort,
    load_api_key,
    load_model_configs,
    model_id_to_slug,
    register_config,
    slug_to_model_id,
)


class TestPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_cache_dir_under_project(self):
        assert CACHE_DIR.parent == PROJECT_ROOT

    def test_results_dir_under_project(self):
        assert RESULTS_DIR.parent == PROJECT_ROOT

    def test_configs_path(self):
        assert CONFIGS_PATH.parent.name == "configs"


class TestConstants:
    def test_max_turns(self):
        assert MAX_TURNS == 36

    def test_checkpoint_turns(self):
        assert CHECKPOINT_TURNS == [6, 12, 18, 24, 30, 36]

    def test_scoring_weights_sum_to_one(self):
        assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 1e-9


class TestSlugConversions:
    def test_model_id_to_slug(self):
        assert model_id_to_slug("openai/gpt-5-nano") == "openai--gpt-5-nano"

    def test_slug_to_model_id(self):
        assert slug_to_model_id("openai--gpt-5-nano") == "openai/gpt-5-nano"

    def test_roundtrip(self):
        mid = "google/gemini-3-flash-preview"
        assert slug_to_model_id(model_id_to_slug(mid)) == mid


class TestReasoningEffort:
    def test_google_gemini_default(self):
        assert get_reasoning_effort("google/gemini-3-flash-preview") == "none"

    def test_google_pro_requires_reasoning(self):
        assert get_reasoning_effort("google/gemini-3-pro-preview") == "low"

    def test_google_31_pro(self):
        assert get_reasoning_effort("google/gemini-3.1-pro-preview") == "low"

    def test_anthropic(self):
        assert get_reasoning_effort("anthropic/claude-sonnet-4.6") == "none"

    def test_xai(self):
        assert get_reasoning_effort("x-ai/grok-4.20-beta") == "low"

    def test_unknown_prefix(self):
        assert get_reasoning_effort("unknown/model") == "low"


class TestModelConfig:
    def test_label_uses_display_label(self):
        cfg = ModelConfig(model_id="openai/gpt-5", display_label="gpt-5-custom")
        assert cfg.label == "gpt-5-custom"

    def test_label_fallback_to_model_id(self):
        cfg = ModelConfig(model_id="openai/gpt-5")
        assert cfg.label == "openai/gpt-5"

    def test_effective_temperature_override(self):
        cfg = ModelConfig(model_id="openai/gpt-5", temperature=1.0)
        assert cfg.effective_temperature == 1.0

    def test_effective_temperature_default(self):
        cfg = ModelConfig(model_id="openai/gpt-5")
        assert cfg.effective_temperature == 1.0

    def test_effective_reasoning_override(self):
        cfg = ModelConfig(model_id="openai/gpt-5", reasoning_effort="high")
        assert cfg.effective_reasoning == "high"

    def test_config_dir_name(self):
        cfg = ModelConfig(model_id="openai/gpt-5", temperature=0.7, reasoning_effort="low")
        assert cfg.config_dir_name == "openai--gpt-5@low-t0.7"

    def test_config_dir_name_with_provider(self):
        cfg = ModelConfig(
            model_id="moonshotai/kimi-k2.5", temperature=0.7, reasoning_effort="none", provider="moonshotai/int4"
        )
        assert cfg.config_dir_name == "moonshotai--kimi-k2.5+moonshotai-int4@none-t0.7"


class TestModelRegistry:
    def setup_method(self):
        self._orig = dict(MODEL_CONFIGS)

    def teardown_method(self):
        MODEL_CONFIGS.clear()
        MODEL_CONFIGS.update(self._orig)

    def test_register_and_get(self):
        cfg = ModelConfig(model_id="test/model-a", display_label="test-a")
        register_config(cfg)
        assert get_model_config("test-a") is cfg

    def test_duplicate_label_raises(self):
        cfg1 = ModelConfig(model_id="test/model-b", display_label="test-b")
        register_config(cfg1)
        cfg2 = ModelConfig(model_id="test/model-c", display_label="test-b")
        with pytest.raises(ValueError, match="Duplicate"):
            register_config(cfg2)

    def test_get_by_model_id(self):
        cfg = ModelConfig(model_id="test/model-d", display_label="test-d-label")
        register_config(cfg)
        assert get_model_config("test/model-d") is cfg

    def test_get_creates_default(self):
        result = get_model_config("nonexistent/model")
        assert result.model_id == "nonexistent/model"
        assert result.display_label == ""

    def test_get_config_by_dir_name(self):
        cfg = ModelConfig(model_id="test/model-e", display_label="test-e", temperature=0.7, reasoning_effort="low")
        register_config(cfg)
        found = get_config_by_dir_name(cfg.config_dir_name)
        assert found is cfg

    def test_get_config_by_dir_name_not_found(self):
        assert get_config_by_dir_name("nonexistent@low-t0.7") is None


class TestGenerateDisplayLabel:
    def test_basic(self):
        label = generate_display_label("openai/gpt-5-nano", "low", 0.7)
        assert label == "gpt-5-nano@low-t0.7"

    def test_no_provider_prefix(self):
        label = generate_display_label("gpt-5", "none", 1.0)
        assert label == "gpt-5@none-t1.0"


class TestEnsureDirs:
    def test_creates_dirs(self, tmp_path):
        with patch("src.config.CACHE_DIR", tmp_path / "cache"), patch("src.config.RESULTS_DIR", tmp_path / "results"):
            ensure_dirs()
            assert (tmp_path / "cache").exists()
            assert (tmp_path / "results").exists()


class TestLoadApiKey:
    def test_loads_from_env_openrouter_key(self):
        with patch.dict(os.environ, {"OPENROUTER_KEY": "sk-test-key", "OPENROUTER_API_KEY": ""}, clear=False):
            key = load_api_key()
            assert key == "sk-test-key"

    def test_loads_from_env_openrouter_api_key(self):
        with patch.dict(os.environ, {"OPENROUTER_KEY": "", "OPENROUTER_API_KEY": "sk-fallback"}, clear=False):
            key = load_api_key()
            assert key == "sk-fallback"

    def test_openrouter_key_takes_priority(self):
        with patch.dict(os.environ, {"OPENROUTER_KEY": "primary", "OPENROUTER_API_KEY": "fallback"}, clear=False):
            key = load_api_key()
            assert key == "primary"

    def test_exits_on_missing_key(self):
        with (
            patch.dict(os.environ, {"OPENROUTER_KEY": "", "OPENROUTER_API_KEY": ""}, clear=False),
            patch("src.config.ENV_PATH", Path("/nonexistent/.env")),
            pytest.raises(SystemExit),
        ):
            load_api_key()

    def test_not_required(self):
        with (
            patch.dict(os.environ, {"OPENROUTER_KEY": "", "OPENROUTER_API_KEY": ""}, clear=False),
            patch("src.config.ENV_PATH", Path("/nonexistent/.env")),
        ):
            key = load_api_key(required=False)
            assert key == ""


class TestModelPricing:
    def test_defaults(self):
        p = ModelPricing()
        assert p.prompt_price == 0.0
        assert p.completion_price == 0.0


class TestGrok41Config:
    """Verify Grok 4.1 configuration for cross-model cache compatibility."""

    def test_grok41_config_dir_name(self):
        cfg = ModelConfig(model_id="x-ai/grok-4.1-fast", temperature=0.7, reasoning_effort="low")
        assert cfg.config_dir_name == "x-ai--grok-4.1-fast@low-t0.7"

    def test_grok41_reasoning_effort(self):
        assert get_reasoning_effort("x-ai/grok-4.1-fast") == "low"

    def test_grok41_effective_settings(self):
        cfg = ModelConfig(model_id="x-ai/grok-4.1-fast", temperature=0.7, reasoning_effort="low")
        assert cfg.effective_temperature == 0.7
        assert cfg.effective_reasoning == "low"

    def test_grok41_slug_roundtrip(self):
        mid = "x-ai/grok-4.1-fast"
        assert slug_to_model_id(model_id_to_slug(mid)) == mid

    def test_grok41_loaded_from_yaml(self):
        matches = [c for c in MODEL_CONFIGS.values() if c.model_id == "x-ai/grok-4.1-fast"]
        assert len(matches) == 1
        cfg = matches[0]
        assert cfg.effective_reasoning == "low"
        assert cfg.effective_temperature == 1.0
        assert cfg.config_dir_name == "x-ai--grok-4.1-fast@low-t1.0"


class TestYamlLoader:
    def test_load_from_yaml(self, tmp_path):
        yaml_data = {
            "models": [
                {
                    "model_id": "test/yaml-model",
                    "temperature": 0.5,
                    "reasoning_effort": "high",
                    "active": True,
                },
            ],
        }
        yaml_path = tmp_path / "test_models.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        orig = dict(MODEL_CONFIGS)
        try:
            configs = load_model_configs(yaml_path)
            assert len(configs) == 1
            assert configs[0].model_id == "test/yaml-model"
            assert configs[0].temperature == 0.5
        finally:
            MODEL_CONFIGS.clear()
            MODEL_CONFIGS.update(orig)

    def test_missing_file_returns_empty(self, tmp_path):
        configs = load_model_configs(tmp_path / "nonexistent.yaml")
        assert configs == []

    def test_empty_yaml_returns_empty(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        configs = load_model_configs(yaml_path)
        assert configs == []

    def test_yaml_with_provider(self, tmp_path):
        yaml_data = {
            "models": [
                {
                    "model_id": "test/pinned-model",
                    "temperature": 0.7,
                    "reasoning_effort": "none",
                    "provider": "test/provider",
                },
            ],
        }
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        orig = dict(MODEL_CONFIGS)
        try:
            configs = load_model_configs(yaml_path)
            assert configs[0].provider == "test/provider"
        finally:
            MODEL_CONFIGS.clear()
            MODEL_CONFIGS.update(orig)
