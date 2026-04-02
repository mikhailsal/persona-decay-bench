.PHONY: help install install-dev test test-cov run evaluate leaderboard generate-report estimate-cost clear-cache clear-scores

# Model(s) to benchmark — override with: make run MODELS="openai/gpt-5-nano"
MODELS ?=

# Number of parallel workers
PARALLEL ?= 1

# ── helpers ──────────────────────────────────────────────────────────────────

_run_flags :=
ifneq ($(MODELS),)
  _run_flags += --models "$(MODELS)"
endif
ifneq ($(PARALLEL),1)
  _run_flags += --parallel $(PARALLEL)
endif

# ── targets ──────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (production dependencies only)
	pip install -e .

install-dev:  ## Install the package with test/dev dependencies
	pip install -e ".[test]"

test:  ## Run the test suite
	python -m pytest tests/

test-cov:  ## Run the test suite with coverage report
	python -m pytest tests/ --cov=src --cov-report=term-missing

run:  ## Run conversations for default or specified models
	python -m src.cli run $(_run_flags)

evaluate:  ## Run observer assessments on completed conversations
	python -m src.cli evaluate $(_run_flags)

leaderboard:  ## Display the leaderboard from cached results
	python -m src.cli leaderboard

generate-report:  ## Generate Markdown leaderboard report
	python -m src.cli generate-report

estimate-cost:  ## Estimate benchmark cost without running
	python -m src.cli estimate-cost $(if $(MODELS),--models "$(MODELS)")

clear-cache:  ## Clear all cached conversations and scores
	python -m src.cli clear-cache

clear-scores:  ## Clear only observer scores, keeping conversations
	python -m src.cli clear-cache --scores-only
