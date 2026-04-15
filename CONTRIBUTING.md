# Contributing to Quartermaster

Thank you for considering a contribution to Quartermaster! This document explains how to set up your development environment, run tests, and submit changes.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) or pip
- Docker (required only for `quartermaster-code-runner`)
- Git

### Clone the repository

```bash
git clone https://github.com/MindMadeLab/quartermaster-sdk-py.git
cd quartermaster-sdk-py
```

### Install with uv (recommended)

```bash
# Install all packages in development mode
uv pip install -e quartermaster-providers -e quartermaster-graph \
  -e quartermaster-tools -e quartermaster-nodes -e quartermaster-engine \
  -e quartermaster-mcp-client -e quartermaster-code-runner \
  -e quartermaster-sdk

# Install dev dependencies for a specific package
uv pip install -e "quartermaster-graph[dev]"
```

### Install with pip

```bash
# Single package
cd quartermaster-providers
pip install -e ".[dev]"
```

The `[dev]` extra installs test and development dependencies (pytest, ruff, mypy, etc.).

### Install all packages with pip

```bash
for pkg in quartermaster-providers quartermaster-graph quartermaster-tools \
  quartermaster-nodes quartermaster-engine quartermaster-mcp-client \
  quartermaster-code-runner quartermaster-sdk; do
    pip install -e "$pkg[dev]"
done
```

## Running Tests

### Per-package

```bash
cd quartermaster-providers
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
quartermaster/
    quartermaster-sdk/               # Meta-package (installs everything)
    quartermaster-providers/         # LLM provider abstraction
    quartermaster-graph/             # Graph schema and builder
    quartermaster-tools/             # Tool definition and registry
    quartermaster-nodes/             # Node type implementations
    quartermaster-engine/            # Flow execution engine
    quartermaster-mcp-client/        # MCP protocol client (standalone)
    quartermaster-code-runner/       # Docker code execution (standalone)
    examples/                        # Runnable example scripts
    docs/                            # Documentation
```

Each package follows the same internal layout:

```
quartermaster-<name>/
    src/quartermaster_<name>/        # Source code (or quartermaster_<name>/ directly)
        __init__.py
        ...
    tests/
        test_*.py
    pyproject.toml         # Package metadata and dependencies
    README.md              # Package-specific documentation
```

### Dependency graph

Packages have a layered dependency structure:

- **Standalone**: `quartermaster-mcp-client`, `quartermaster-code-runner` (no internal dependencies)
- **Foundation**: `quartermaster-providers` (used by nodes for LLM calls)
- **Middle layer**: `quartermaster-tools`, `quartermaster-nodes`, `quartermaster-graph`
- **Top layer**: `quartermaster-engine` (orchestrates everything)

## Pull Request Process

1. **Fork the repository** and create a feature branch from `master`.
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
5. **Open a pull request** against `master` with:
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
