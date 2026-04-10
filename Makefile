.PHONY: test-all lint-all typecheck-all

PACKAGES = quartermaster-mcp-client quartermaster-code-runner quartermaster-providers quartermaster-nodes quartermaster-tools quartermaster-graph quartermaster-engine

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
