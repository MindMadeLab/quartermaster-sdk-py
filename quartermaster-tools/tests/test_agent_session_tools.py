"""Tests for parallel agent session management tools."""

from __future__ import annotations

import time

import pytest

from quartermaster_tools.builtin.agents.session import (
    AgentSession,
    SessionManager,
    SessionStatus,
)
from quartermaster_tools.builtin.agents.tools import (
    add_agent_finish_hook,
    cancel_agent_session,
    collect_agent_results,
    create_agent_session,
    get_agent_session_status,
    inject_agent_message,
    list_agent_sessions,
    notify_parent,
    set_manager,
    spawn_agent,
    start_agent_session,
    wait_agent_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quick_task(session: AgentSession):
    """A simple task function that completes quickly."""
    time.sleep(0.05)
    return "done"


def _failing_task(session: AgentSession):
    """A task function that raises an exception."""
    time.sleep(0.05)
    raise RuntimeError("task failed on purpose")


def _slow_task(session: AgentSession):
    """A task that takes longer than typical test timeouts."""
    time.sleep(2.0)
    return "slow done"


@pytest.fixture(autouse=True)
def _fresh_manager():
    """Install a fresh SessionManager for every test, then reset."""
    mgr = SessionManager()
    set_manager(mgr)
    yield mgr
    set_manager(None)


# ---------------------------------------------------------------------------
# SessionManager unit tests
# ---------------------------------------------------------------------------


class TestSessionManager:
    def test_create_session(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session(name="test")
        assert session.name == "test"
        assert session.status == SessionStatus.CREATED
        assert session.id in [s.id for s in mgr.list_sessions()]

    def test_get_session_by_id(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session(name="lookup")
        found = mgr.get_session(session.id)
        assert found is session

    def test_get_session_not_found(self, _fresh_manager):
        mgr = _fresh_manager
        assert mgr.get_session("nonexistent") is None

    def test_list_all_sessions(self, _fresh_manager):
        mgr = _fresh_manager
        mgr.create_session(name="a")
        mgr.create_session(name="b")
        assert len(mgr.list_sessions()) == 2

    def test_list_sessions_filtered_by_status(self, _fresh_manager):
        mgr = _fresh_manager
        s1 = mgr.create_session(name="a")
        s2 = mgr.create_session(name="b")
        s2.status = SessionStatus.RUNNING
        created = mgr.list_sessions(status=SessionStatus.CREATED)
        assert len(created) == 1
        assert created[0].id == s1.id

    def test_inject_message(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        ok = mgr.inject_message(session.id, "user", "hello")
        assert ok is True
        assert len(session.messages) == 1
        assert session.messages[0].content == "hello"
        assert session.messages[0].role == "user"

    def test_inject_message_invalid_session(self, _fresh_manager):
        mgr = _fresh_manager
        assert mgr.inject_message("nope", "user", "hi") is False

    def test_start_session_with_task(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        started = mgr.start_session(session.id, _quick_task)
        assert started is True
        assert session.status == SessionStatus.RUNNING

    def test_session_completes_successfully(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        mgr.start_session(session.id, _quick_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert session.status == SessionStatus.COMPLETED
        assert session.result == "done"

    def test_session_fails(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        mgr.start_session(session.id, _failing_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert session.status == SessionStatus.FAILED
        assert "task failed on purpose" in session.error

    def test_finish_hook_fires_on_completion(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        hook_called = []
        mgr.add_finish_hook(session.id, lambda s: hook_called.append(s.status))
        mgr.start_session(session.id, _quick_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert hook_called == [SessionStatus.COMPLETED]

    def test_finish_hook_fires_on_failure(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        hook_called = []
        mgr.add_finish_hook(session.id, lambda s: hook_called.append(s.status))
        mgr.start_session(session.id, _failing_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert hook_called == [SessionStatus.FAILED]

    def test_cancel_session(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        ok = mgr.cancel_session(session.id)
        assert ok is True
        assert session.status == SessionStatus.CANCELLED

    def test_cancel_nonexistent(self, _fresh_manager):
        mgr = _fresh_manager
        assert mgr.cancel_session("nope") is False

    def test_wait_for_session(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        mgr.start_session(session.id, _quick_task)
        result = mgr.wait_for_session(session.id, timeout=1.0)
        assert result is not None
        assert result.status == SessionStatus.COMPLETED

    def test_wait_all_multiple(self, _fresh_manager):
        mgr = _fresh_manager
        s1 = mgr.create_session(name="w1")
        s2 = mgr.create_session(name="w2")
        mgr.start_session(s1.id, _quick_task)
        mgr.start_session(s2.id, _quick_task)
        results = mgr.wait_all([s1.id, s2.id], timeout=1.0)
        assert len(results) == 2
        assert all(s.status == SessionStatus.COMPLETED for s in results)

    def test_clear_completed(self, _fresh_manager):
        mgr = _fresh_manager
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        mgr.start_session(s1.id, _quick_task)
        mgr.wait_for_session(s1.id, timeout=1.0)
        removed = mgr.clear_completed()
        assert removed == 1
        assert mgr.get_session(s1.id) is None
        assert mgr.get_session(s2.id) is not None


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestCreateSessionTool:
    def test_creates_session(self, _fresh_manager):
        mgr = _fresh_manager
        result = create_agent_session.run(name="my-session")
        assert result.success is True
        assert result.data["name"] == "my-session"
        assert result.data["status"] == "created"
        assert mgr.get_session(result.data["session_id"]) is not None


class TestStartSessionTool:
    def test_starts_session(self, _fresh_manager):
        mgr = _fresh_manager
        r = create_agent_session.run(name="starter")
        sid = r.data["session_id"]
        result = start_agent_session.run(session_id=sid, task="do something")
        assert result.success is True
        assert result.data["status"] == "running"
        # Wait for completion
        mgr.wait_for_session(sid, timeout=1.0)
        session = mgr.get_session(sid)
        assert session.status == SessionStatus.COMPLETED

    def test_start_missing_session(self):
        result = start_agent_session.run(session_id="nonexistent", task="x")
        assert result.success is False


class TestInjectMessageTool:
    def test_injects_message(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        result = inject_agent_message.run(session_id=session.id, content="hello world")
        assert result.success is True
        assert result.data["injected"] is True
        assert result.data["message_count"] == 1

    def test_invalid_session_returns_error(self):
        result = inject_agent_message.run(session_id="bad-id", content="hello")
        assert result.success is False


class TestGetSessionStatusTool:
    def test_returns_status(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session(name="status-check")
        result = get_agent_session_status.run(session_id=session.id)
        assert result.success is True
        assert result.data["status"] == "created"
        assert result.data["name"] == "status-check"

    def test_nonexistent_session(self):
        result = get_agent_session_status.run(session_id="does-not-exist")
        assert result.success is False


class TestListSessionsTool:
    def test_lists_all(self, _fresh_manager):
        mgr = _fresh_manager
        mgr.create_session(name="a")
        mgr.create_session(name="b")
        result = list_agent_sessions.run()
        assert result.success is True
        assert result.data["count"] == 2

    def test_filters_by_status(self, _fresh_manager):
        mgr = _fresh_manager
        s1 = mgr.create_session(name="a")
        s2 = mgr.create_session(name="b")
        s2.status = SessionStatus.RUNNING
        result = list_agent_sessions.run(status="running")
        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["sessions"][0]["name"] == "b"


class TestWaitSessionTool:
    def test_waits_and_returns_result(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        mgr.start_session(session.id, _quick_task)
        result = wait_agent_session.run(session_id=session.id, timeout=1.0)
        assert result.success is True
        assert result.data["status"] == "completed"
        assert result.data["result"] == "done"

    def test_timeout_handling(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        mgr.start_session(session.id, _slow_task)
        result = wait_agent_session.run(session_id=session.id, timeout=0.1)
        assert result.success is True
        # Should still be running after short timeout
        assert result.data["status"] == "running"


class TestCollectResultsTool:
    def test_collects_from_multiple(self, _fresh_manager):
        mgr = _fresh_manager
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        mgr.start_session(s1.id, _quick_task)
        mgr.start_session(s2.id, _quick_task)
        result = collect_agent_results.run(session_ids=f"{s1.id},{s2.id}", timeout=1.0)
        assert result.success is True
        assert result.data["all_completed"] is True
        assert len(result.data["results"]) == 2


class TestCancelSessionTool:
    def test_cancels_session(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        result = cancel_agent_session.run(session_id=session.id)
        assert result.success is True
        assert result.data["cancelled"] is True
        assert session.status == SessionStatus.CANCELLED


class TestAddFinishHookTool:
    def test_adds_log_hook(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        result = add_agent_finish_hook.run(
            session_id=session.id,
            hook_type="log",
            hook_config='{"path": "/tmp/test_agent.log"}',
        )
        assert result.success is True
        assert result.data["hook_added"] is True
        assert result.data["hook_type"] == "log"
        assert len(session._on_finish) == 1

    def test_adds_notify_hook(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        result = add_agent_finish_hook.run(session_id=session.id, hook_type="notify")
        assert result.success is True
        # Run session and verify notification fires
        mgr.start_session(session.id, _quick_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert "notification" in session.metadata
        assert session.metadata["notification"]["status"] == "completed"

    def test_invalid_hook_type(self, _fresh_manager):
        mgr = _fresh_manager
        session = mgr.create_session()
        result = add_agent_finish_hook.run(session_id=session.id, hook_type="bad")
        assert result.success is False


class TestSessionManagerAllowedAgents:
    def test_empty_allowed_agents_allows_all(self, _fresh_manager):
        mgr = _fresh_manager
        assert mgr.is_agent_allowed("any-agent") is True
        assert mgr.is_agent_allowed("another") is True

    def test_set_allowed_agents_restricts(self, _fresh_manager):
        mgr = _fresh_manager
        mgr.set_allowed_agents(["agent-a", "agent-b"])
        assert mgr.is_agent_allowed("agent-a") is True
        assert mgr.is_agent_allowed("agent-b") is True
        assert mgr.is_agent_allowed("agent-c") is False

    def test_create_session_with_allowed_agent_id(self, _fresh_manager):
        mgr = _fresh_manager
        mgr.set_allowed_agents(["agent-a"])
        session = mgr.create_session(name="test", agent_id="agent-a")
        assert session.metadata["agent_id"] == "agent-a"

    def test_create_session_with_disallowed_agent_id(self, _fresh_manager):
        mgr = _fresh_manager
        mgr.set_allowed_agents(["agent-a"])
        with pytest.raises(ValueError, match="not in the allowed agents"):
            mgr.create_session(name="test", agent_id="agent-x")


class TestSpawnAgentTool:
    def test_basic_spawn(self, _fresh_manager):
        mgr = _fresh_manager
        result = spawn_agent.run(agent_id="researcher", task="find information")
        assert result.success is True
        assert result.data["status"] == "running"
        sid = result.data["session_id"]
        mgr.wait_for_session(sid, timeout=1.0)
        session = mgr.get_session(sid)
        assert session.status == SessionStatus.COMPLETED
        assert session.metadata["agent_id"] == "researcher"

    def test_spawn_with_name_and_system_prompt(self, _fresh_manager):
        mgr = _fresh_manager
        result = spawn_agent.run(
            agent_id="writer",
            task="write a report",
            name="my-writer",
            system_prompt="You are a helpful writer.",
        )
        assert result.success is True
        sid = result.data["session_id"]
        mgr.wait_for_session(sid, timeout=1.0)
        session = mgr.get_session(sid)
        assert session.name == "my-writer"
        # system prompt + user task = 2 messages
        assert len(session.messages) == 2
        assert session.messages[0].role == "system"

    def test_spawn_rejects_disallowed_agent(self, _fresh_manager):
        mgr = _fresh_manager
        mgr.set_allowed_agents(["agent-a", "agent-b"])
        result = spawn_agent.run(agent_id="agent-c", task="do stuff")
        assert result.success is False
        assert "not in the allowed agents" in result.error

    def test_spawn_empty_allowed_list_allows_all(self, _fresh_manager):
        mgr = _fresh_manager
        result = spawn_agent.run(agent_id="any-agent", task="do anything")
        assert result.success is True
        assert result.data["status"] == "running"
        mgr.wait_for_session(result.data["session_id"], timeout=1.0)

    def test_spawn_with_allowed_agents_metadata(self, _fresh_manager):
        mgr = _fresh_manager
        result = spawn_agent.run(
            agent_id="orchestrator",
            task="coordinate",
            allowed_agents="worker-a, worker-b",
        )
        assert result.success is True
        sid = result.data["session_id"]
        session = mgr.get_session(sid)
        assert session.metadata["allowed_agents"] == ["worker-a", "worker-b"]

    def test_spawn_missing_agent_id(self):
        result = spawn_agent.run(agent_id="", task="do stuff")
        assert result.success is False
        assert "agent_id" in result.error

    def test_spawn_missing_task(self):
        result = spawn_agent.run(agent_id="agent-a", task="")
        assert result.success is False
        assert "task" in result.error


class TestNotifyParentTool:
    def test_basic_notification(self):
        result = notify_parent.run(message="Task 50% complete", status="progress")
        assert result.success
        assert result.data["status"] == "progress"
        assert result.data["notification"]["message"] == "Task 50% complete"

    def test_missing_message(self):
        result = notify_parent.run()
        assert not result.success

    def test_completed_status(self):
        result = notify_parent.run(message="All done", status="completed", data='{"items": 5}')
        assert result.success
        assert result.data["status"] == "completed"
        assert result.data["notification"]["data"]["items"] == 5

    def test_invalid_json_data(self):
        result = notify_parent.run(message="update", data="not-json")
        assert result.success
        assert result.data["notification"]["data"] == {"raw": "not-json"}


class TestSpawnAgentWithParentSession:
    def test_parent_session_id_stored(self, _fresh_manager):
        mgr = _fresh_manager
        result = spawn_agent.run(
            agent_id="child",
            task="do work",
            parent_session_id="parent-123",
        )
        assert result.success
        sid = result.data["session_id"]
        session = mgr.get_session(sid)
        assert session.metadata["parent_session_id"] == "parent-123"

    def test_no_parent_session_id(self, _fresh_manager):
        mgr = _fresh_manager
        result = spawn_agent.run(agent_id="child", task="do work")
        assert result.success
        sid = result.data["session_id"]
        session = mgr.get_session(sid)
        assert "parent_session_id" not in session.metadata


class TestParallelSessions:
    def test_three_concurrent_sessions(self, _fresh_manager):
        mgr = _fresh_manager
        sessions = [mgr.create_session(name=f"parallel-{i}") for i in range(3)]
        for s in sessions:
            mgr.start_session(s.id, _quick_task)
        results = mgr.wait_all([s.id for s in sessions], timeout=2.0)
        assert len(results) == 3
        assert all(s.status == SessionStatus.COMPLETED for s in results)
        assert all(s.result == "done" for s in results)


class TestOrchestratorPattern:
    """End-to-end test of the orchestrator -> spawn N agents -> collect pattern."""

    def test_spawn_three_and_collect(self, _fresh_manager):
        """Orchestrator spawns researcher, writer, reviewer; collects all."""
        mgr = _fresh_manager
        mgr.set_allowed_agents(["researcher", "writer", "reviewer"])

        # Spawn three agents (simulates three tool_calls in one LLM turn)
        r1 = spawn_agent.run(agent_id="researcher", task="Research AI trends")
        r2 = spawn_agent.run(agent_id="writer", task="Draft blog post on AI")
        r3 = spawn_agent.run(agent_id="reviewer", task="Review checklist for AI post")
        assert r1.success and r2.success and r3.success

        sid1, sid2, sid3 = r1.data["session_id"], r2.data["session_id"], r3.data["session_id"]

        # Collect (simulates second LLM turn)
        result = collect_agent_results.run(session_ids=f"{sid1},{sid2},{sid3}", timeout=5)
        assert result.success
        assert result.data["all_completed"]
        assert len(result.data["results"]) == 3

    def test_spawn_blocked_agent_fails(self, _fresh_manager):
        """Orchestrator cannot spawn agents not in allowed_agents."""
        mgr = _fresh_manager
        mgr.set_allowed_agents(["researcher", "writer"])
        result = spawn_agent.run(agent_id="hacker", task="do bad things")
        assert not result.success
        assert "not in the allowed agents" in result.error

    def test_partial_collect_with_timeout(self, _fresh_manager):
        """collect_agent_results returns partial results on timeout."""
        mgr = _fresh_manager

        # Spawn a fast agent
        r_fast = spawn_agent.run(agent_id="fast", task="quick job")
        assert r_fast.success

        # Create a slow session manually
        slow_session = mgr.create_session(name="slow")

        def slow_task(s):
            time.sleep(10)  # will exceed timeout
            return "late"

        mgr.start_session(slow_session.id, slow_task)

        result = collect_agent_results.run(
            session_ids=f"{r_fast.data['session_id']},{slow_session.id}",
            timeout=1,
        )
        assert result.success
        # Not all completed because the slow one timed out
        assert not result.data["all_completed"]

    def test_spawn_with_parent_and_notify(self, _fresh_manager):
        """Sub-agent notifies parent session after completing."""
        mgr = _fresh_manager

        # Create parent session
        parent = mgr.create_session(name="orchestrator")
        mgr.start_session(parent.id, _quick_task)

        # Spawn child with parent reference
        r = spawn_agent.run(
            agent_id="worker",
            task="do work",
            parent_session_id=parent.id,
        )
        assert r.success
        child_sid = r.data["session_id"]
        child_session = mgr.get_session(child_sid)
        assert child_session.metadata["parent_session_id"] == parent.id

        # Child notifies parent
        notify_result = notify_parent.run(
            message="work complete",
            status="completed",
        )
        assert notify_result.success

    def test_recursive_allowed_agents(self, _fresh_manager):
        """Spawned agent can restrict its own children's allowed agents."""
        mgr = _fresh_manager
        mgr.set_allowed_agents(["level1"])
        result = spawn_agent.run(
            agent_id="level1",
            task="manage sub-tasks",
            allowed_agents="level2a,level2b",
        )
        assert result.success
        session = mgr.get_session(result.data["session_id"])
        assert session.metadata["allowed_agents"] == ["level2a", "level2b"]
