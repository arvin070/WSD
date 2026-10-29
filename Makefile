.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create the virtualenv
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip

setup: $(VENV) ## Install the package and dev dependencies
	$(PIP) install --quiet -e ".[dev]"
	@echo "ready — run 'make train' then 'make serve'"

train: ## Train the model and write models/
	$(PY) -m scamshield.ml.train

serve: ## Run the web app
	$(PY) -m scamshield

test: ## Run the test suite
	$(VENV)/bin/pytest

lint: ## Lint and check formatting
	$(VENV)/bin/ruff check src tests

fmt: ## Auto-fix lint issues
	$(VENV)/bin/ruff check --fix src tests

docker-build: ## Build the container image
	docker build -t scamshield:latest .

docker-run: ## Run the container on port 8000
	docker run --rm -p 8000:8000 scamshield:latest

webhook-dev: ## Run with the WhatsApp webhook mounted (needs .env)
	SCAMSHIELD_WA_ENABLED=true $(PY) -m scamshield

clean: ## Remove caches and generated data
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/corpus.csv
	find . -name '__pycache__' -not -path './.venv/*' -type d -exec rm -rf {} +

.PHONY: help setup train serve test lint fmt webhook-dev docker-build docker-run clean
