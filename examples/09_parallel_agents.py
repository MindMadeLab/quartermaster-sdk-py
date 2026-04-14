"""Parallel agent sessions.

Demonstrates the SessionManager for running multiple agent tasks
concurrently. Each session runs in its own thread and results are
collected when all complete.

This example uses SessionManager directly (not a Graph), so it does
not require an LLM API key.

Usage:
    uv run examples/09_parallel_agents.py
"""

from __future__ import annotations

import time

from quartermaster_tools.builtin.agents.session import (
    AgentMessage,
    SessionManager,
    SessionStatus,
)

manager = SessionManager()

# Create 3 parallel research sessions
topics = ["AI trends 2026", "EU AI Act impact", "Agent framework comparison"]
sessions = []

for topic in topics:
    session = manager.create_session(name=f"Research: {topic}")

    def make_task(t: str):
        """Create a closure over the topic string."""
        def task(s):
            # Simulate research work with a small delay
            time.sleep(0.1)
            s.messages.append(
                AgentMessage(role="assistant", content=f"Researching: {t}")
            )
            return {"topic": t, "findings": f"Key findings about {t}"}
        return task

    manager.start_session(session.id, make_task(topic))
    sessions.append(session)

print(f"Started {len(sessions)} parallel sessions")

# Wait for all to complete
results = manager.wait_all([s.id for s in sessions], timeout=10)

print("\nResults:")
for session in results:
    print(f"  {session.name}: {session.status.value}")
    if session.result:
        print(f"    Findings: {session.result}")
    if session.error:
        print(f"    Error: {session.error}")

# Cleanup
cleared = manager.clear_completed()
print(f"\nCleared {cleared} completed sessions")
