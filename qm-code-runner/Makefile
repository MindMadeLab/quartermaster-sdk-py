.PHONY: help install dev lint format typecheck test test-unit test-integration build-runtimes build-runtime build-service run clean

RUNTIME_DIR := src/qm_code_runner/runtime
IMAGE_PREFIX := code-runner-
LANGUAGES := python node go rust deno bun

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Development
# =============================================================================

install: ## Install package in development mode
	pip install -e ".[dev]"

dev: install ## Install and run development server
	uvicorn qm_code_runner.app:app --reload --host 0.0.0.0 --port 8000

lint: ## Run linter
	ruff check src/ tests/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checker
	mypy src/qm_code_runner --strict

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires Docker)
	pytest tests/integration/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=qm_code_runner --cov-report=term-missing --cov-report=html

# =============================================================================
# Docker: Runtime Images
# =============================================================================

build-runtimes: ## Build all runtime Docker images
	@for lang in $(LANGUAGES); do \
		echo "Building $(IMAGE_PREFIX)$$lang..."; \
		docker build -t $(IMAGE_PREFIX)$$lang $(RUNTIME_DIR)/$$lang/; \
	done
	@echo "All runtime images built successfully."

build-runtime: ## Build a specific runtime image (usage: make build-runtime lang=python)
	@if [ -z "$(lang)" ]; then \
		echo "Usage: make build-runtime lang=<language>"; \
		echo "Available: $(LANGUAGES)"; \
		exit 1; \
	fi
	docker build -t $(IMAGE_PREFIX)$(lang) $(RUNTIME_DIR)/$(lang)/

list-runtimes: ## List built runtime images
	@docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | \
		grep "$(IMAGE_PREFIX)" || echo "No runtime images found. Run 'make build-runtimes'"

# =============================================================================
# Docker: Service
# =============================================================================

build-service: ## Build the code-runner service Docker image
	docker build -t qm-code-runner:latest .

run: ## Run the service with Docker Compose
	docker compose up -d

stop: ## Stop the service
	docker compose down

logs: ## Show service logs
	docker compose logs -f code-runner

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-images: ## Remove all code-runner Docker images
	@for lang in $(LANGUAGES); do \
		docker rmi $(IMAGE_PREFIX)$$lang 2>/dev/null || true; \
	done
	docker rmi qm-code-runner:latest 2>/dev/null || true
	@echo "All code-runner images removed."
