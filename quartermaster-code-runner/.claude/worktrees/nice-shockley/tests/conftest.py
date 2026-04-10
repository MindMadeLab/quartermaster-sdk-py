"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


TEST_API_KEYS = "test-key-1,test-key-2"
FIRST_TEST_KEY = TEST_API_KEYS.split(",")[0]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a FastAPI test client with auth configured."""
    monkeypatch.setenv("CODE_RUNNER_API_KEYS", TEST_API_KEYS)
    # Ensure runtime_dir points to the actual runtime directory
    runtime_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "quartermaster_code_runner",
        "runtime",
    )
    monkeypatch.setenv("RUNTIME_DIR", os.path.abspath(runtime_dir))

    from quartermaster_code_runner.app import app

    with TestClient(app) as c:
        c.headers = {"X-API-Key": FIRST_TEST_KEY}  # type: ignore[assignment]
        yield c


@pytest.fixture
def unauthenticated_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client without authentication configured."""
    # Clear any auth settings
    monkeypatch.delenv("CODE_RUNNER_API_KEYS", raising=False)
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    runtime_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "quartermaster_code_runner",
        "runtime",
    )
    monkeypatch.setenv("RUNTIME_DIR", os.path.abspath(runtime_dir))

    from quartermaster_code_runner.app import app

    with TestClient(app) as c:
        yield c
