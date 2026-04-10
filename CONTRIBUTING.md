# Contributing to Quartermaster

Thank you for considering a contribution to Quartermaster! This document explains how to set up your development environment, run tests, and submit changes.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) or pip
- Docker (required only for `qm-code-runner`)
- Git

### Clone the repository

```bash
git clone https://github.com/quartermaster-ai/quartermaster.git
cd quartermaster-opensource
```

### Install a single package in development mode

Each package is self-contained. To work on a specific package:

```bash
cd qm-providers
pip install -e ".[dev]"
```

The `[dev]` extra installs test and development dependencies (pytest, ruff, mypy, etc.).

### Install all packages

To install every package in development mode:

```bash
for pkg in qm-mcp-client qm-code-runner qm-providers qm-tools qm-nodes qm-graph qm-engine; do
    cd "$pkg" && pip install -e ".[dev]" && cd ..
done
```

## Running Tests

### Per-package

```bash
cd qm-providers
pytest
```

### All packages via Makefile

From the repository root:

```bash
make test-all       # Run tests for all packages
make lint-all       # Run ruff linting for all packages
make typecheck-all  # Run mypy type checking for all packages
```

## Code Style

### Formatting and linting

We use [ruff](https://docs.astral.sh/ruff/) for both formatting and linting:

```bash
# Check for lint issues
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

### Type checking

We use [mypy](https://mypy-lang.org/) for static type analysis:

```bash
mypy src/
```

### General conventions

- Use type hints on all public function signatures.
- Prefer `dataclass` or Pydantic models over plain dicts for structured data.
- Write docstrings for all public classes and functions.
- Keep imports sorted (ruff handles this automatically).

## Package Structure

The repository is a monorepo with 7 independent packages:

```
quartermaster-opensource/
    qm-mcp-client/       # MCP protocol client (standalone)
    qm-code-runner/       # Docker code execution (standalone)
    qm-providers/         # LLM provider abstraction
    qm-tools/             # Tool definition and registry
    qm-nodes/             # Node type implementations
    qm-graph/             # Graph schema and builder
    qm-engine/            # Flow execution engine
    examples/             # Runnable example scripts
    Makefile              # Cross-package commands
```

Each package follows the same internal layout:

```
qm-<name>/
    src/qm_<name>/        # Source code (or qm_<name>/ directly)
        __init__.py
        ...
    tests/
        test_*.py
    pyproject.toml         # Package metadata and dependencies
    README.md              # Package-specific documentation
```

### Dependency graph

Packages have a layered dependency structure:

- **Standalone**: `qm-mcp-client`, `qm-code-runner` (no internal dependencies)
- **Foundation**: `qm-providers` (used by nodes for LLM calls)
- **Middle layer**: `qm-tools`, `qm-nodes`, `qm-graph`
- **Top layer**: `qm-engine` (orchestrates everything)

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`.
2. **Make your changes** in the relevant package(s).
3. **Add or update tests** for any new or changed functionality.
4. **Run checks locally** before pushing:
   ```bash
   cd <package>
   ruff check src/ tests/
   ruff format --check src/ tests/
   mypy src/
   pytest
   ```
5. **Open a pull request** against `main` with:
   - A clear title summarizing the change.
   - A description explaining what and why.
   - Reference any related issues.
6. **Address review feedback** promptly.

### What makes a good PR

- Focused on a single concern (one feature, one bug fix, one refactor).
- Includes tests that exercise the new or changed behavior.
- Passes all CI checks (lint, type check, tests).
- Does not introduce dependencies on Django, Celery, or other proprietary code.

## License

By contributing to Quartermaster, you agree that your contributions will be licensed under the [Apache License 2.0](./LICENSE).
