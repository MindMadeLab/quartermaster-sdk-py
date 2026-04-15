"""Smoke-tests for the examples that don't need an LLM API key.

The previous CI "examples" job was removed because it false-passed on
LLM-using scripts (no API keys → ``run_graph()`` silently degrades).
The fix the inline comment in ``.github/workflows/ci.yml`` suggested
was to port the API-key-free examples into proper pytest cases — that's
what this file is.

Catches regressions like the v0.1.5 audit's discovery that
``17_streaming_events.py`` was registering executors against magic
strings (``"UserInput"`` / ``"Start"`` / ``"End"``) that didn't match
the actual ``NodeType`` enum values (``"User1"`` / ``"Start1"`` /
``"End1"``) — the script ran to completion but its only LLM node
``error: No executor registered for node type: User1``-ed silently.

Each test ``subprocess``-runs the script with no API-key env vars set
and asserts non-zero exit means failure.  Stdout is captured for
better failure messages.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Repo root: this file lives at <repo>/quartermaster-sdk/tests/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# Examples that genuinely require zero LLM credentials to run end-to-end.
# Add new entries here as you write API-key-free examples; this is the
# allowlist the smoke test iterates over.
_NO_API_KEY_EXAMPLES: tuple[str, ...] = (
    "06_tool_decorator.py",  # tool registry + JSON schema export, no LLM call
    "09_parallel_agents.py",  # SessionManager simulation, no LLM call
    "17_streaming_events.py",  # FlowRunner with a mock executor, no LLM call
)


@pytest.fixture
def clean_env() -> dict[str, str]:
    """Strip every LLM API key env var so the example can't accidentally
    succeed by hitting a real provider.  We want pure no-LLM smoke tests."""
    env = os.environ.copy()
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "GOOGLE_API_KEY",
        "QUARTERMASTER_API_KEY",
    ):
        env.pop(key, None)
    return env


@pytest.mark.parametrize("example_name", _NO_API_KEY_EXAMPLES)
def test_example_runs_without_api_keys(
    example_name: str, clean_env: dict[str, str]
) -> None:
    """Each listed example must exit 0 with no API keys set in the env."""
    script = _EXAMPLES_DIR / example_name
    assert script.exists(), f"example file missing: {script}"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env,
        cwd=_REPO_ROOT,
    )

    if result.returncode != 0:
        # Surface stdout + stderr in the failure message — much easier
        # to diagnose than a bare assertion.
        pytest.fail(
            f"{example_name} exited {result.returncode}.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def test_example_17_no_silent_executor_misses(clean_env: dict[str, str]) -> None:
    """Targeted regression for the v0.1.5 audit finding.

    Example 17 is allowed to print streaming events but its
    ``FlowResult`` summary must NOT contain
    "No executor registered for node type" — that string is the silent-
    failure signature the magic-string registration produced before the
    fix.  This guard makes sure nobody re-introduces the bug by tweaking
    the example to use string literals again.
    """
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES_DIR / "17_streaming_events.py")],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env,
        cwd=_REPO_ROOT,
    )
    combined = result.stdout + result.stderr
    assert "No executor registered" not in combined, (
        "Example 17 silently dropped a node — the magic-string vs. NodeType.X.value "
        "regression is back. Check examples/17_streaming_events.py."
    )
    # Positive assertion: the mock LLM streamed at least one token.
    assert "TOKEN" in combined, "Expected token-streaming events in the output"
