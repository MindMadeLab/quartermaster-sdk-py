# Contributing to qm-mcp-client

We welcome contributions! This document describes how to set up the development environment and submit changes.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/quartermaster-ai/quartermaster.git
cd packages/qm-mcp-client

# Create virtual environment and install dependencies
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=qm_mcp_client --cov-report=term-missing

# Run specific test file
pytest tests/test_client.py -v

# Run specific test class
pytest tests/test_client.py::TestClientInit -v
```

## Code Quality

### Linting

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix
```

### Formatting

```bash
# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

### Type Checking

```bash
# Run mypy in strict mode
mypy src/qm_mcp_client --strict
```

## Code Style

- We follow PEP 8 via ruff
- All public functions and classes must have Google-style docstrings
- Type hints are required everywhere (mypy strict mode)
- Maximum line length is 88 characters (ruff default)
- Use `from __future__ import annotations` for modern type syntax

## Testing Guidelines

- All new features must include tests
- Minimum 85% code coverage (enforced in CI)
- Use `pytest` and `respx` for HTTP mocking
- Async tests should use `@pytest.mark.asyncio`
- Group related tests into classes
- Use fixtures from `conftest.py` for shared test data

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD encouraged)
3. Implement your changes
4. Ensure all checks pass: `pytest`, `ruff check`, `ruff format --check`, `mypy --strict`
5. Submit a PR with a clear description
6. Wait for CI to pass and code review

## Architecture

- `client.py` — Main `McpClient` class with async/sync APIs
- `transports.py` — SSE and Streamable HTTP transport implementations
- `types.py` — Dataclasses for tools, parameters, server info
- `errors.py` — Custom exception hierarchy

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
