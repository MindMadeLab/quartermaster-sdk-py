.PHONY: test-all lint-all typecheck-all

PACKAGES = qm-mcp-client qm-code-runner qm-providers qm-nodes qm-tools qm-graph qm-engine

test-all:
	@for pkg in $(PACKAGES); do \
		echo "Testing $$pkg..."; \
		cd $$pkg && pip install -e ".[dev]" && pytest && cd ..; \
	done

lint-all:
	@for pkg in $(PACKAGES); do \
		echo "Linting $$pkg..."; \
		cd $$pkg && ruff check src/ tests/ && ruff format --check src/ tests/ && cd ..; \
	done

typecheck-all:
	@for pkg in $(PACKAGES); do \
		echo "Typechecking $$pkg..."; \
		cd $$pkg && mypy src/ && cd ..; \
	done
