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
    AddFinishHookTool,
    CancelSessionTool,
    CollectResultsTool,
    CreateSessionTool,
    GetSessionStatusTool,
    InjectMessageTool,
    ListSessionsTool,
    StartSessionTool,
    WaitSessionTool,
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


# ---------------------------------------------------------------------------
# SessionManager unit tests
# ---------------------------------------------------------------------------

class TestSessionManager:
    def test_create_session(self):
        mgr = SessionManager()
        session = mgr.create_session(name="test")
        assert session.name == "test"
        assert session.status == SessionStatus.CREATED
        assert session.id in [s.id for s in mgr.list_sessions()]

    def test_get_session_by_id(self):
        mgr = SessionManager()
        session = mgr.create_session(name="lookup")
        found = mgr.get_session(session.id)
        assert found is session

    def test_get_session_not_found(self):
        mgr = SessionManager()
        assert mgr.get_session("nonexistent") is None

    def test_list_all_sessions(self):
        mgr = SessionManager()
        mgr.create_session(name="a")
        mgr.create_session(name="b")
        assert len(mgr.list_sessions()) == 2

    def test_list_sessions_filtered_by_status(self):
        mgr = SessionManager()
        s1 = mgr.create_session(name="a")
        s2 = mgr.create_session(name="b")
        s2.status = SessionStatus.RUNNING
        created = mgr.list_sessions(status=SessionStatus.CREATED)
        assert len(created) == 1
        assert created[0].id == s1.id

    def test_inject_message(self):
        mgr = SessionManager()
        session = mgr.create_session()
        ok = mgr.inject_message(session.id, "user", "hello")
        assert ok is True
        assert len(session.messages) == 1
        assert session.messages[0].content == "hello"
        assert session.messages[0].role == "user"

    def test_inject_message_invalid_session(self):
        mgr = SessionManager()
        assert mgr.inject_message("nope", "user", "hi") is False

    def test_start_session_with_task(self):
        mgr = SessionManager()
        session = mgr.create_session()
        started = mgr.start_session(session.id, _quick_task)
        assert started is True
        assert session.status == SessionStatus.RUNNING

    def test_session_completes_successfully(self):
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.start_session(session.id, _quick_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert session.status == SessionStatus.COMPLETED
        assert session.result == "done"

    def test_session_fails(self):
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.start_session(session.id, _failing_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert session.status == SessionStatus.FAILED
        assert "task failed on purpose" in session.error

    def test_finish_hook_fires_on_completion(self):
        mgr = SessionManager()
        session = mgr.create_session()
        hook_called = []
        mgr.add_finish_hook(session.id, lambda s: hook_called.append(s.status))
        mgr.start_session(session.id, _quick_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert hook_called == [SessionStatus.COMPLETED]

    def test_finish_hook_fires_on_failure(self):
        mgr = SessionManager()
        session = mgr.create_session()
        hook_called = []
        mgr.add_finish_hook(session.id, lambda s: hook_called.append(s.status))
        mgr.start_session(session.id, _failing_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert hook_called == [SessionStatus.FAILED]

    def test_cancel_session(self):
        mgr = SessionManager()
        session = mgr.create_session()
        ok = mgr.cancel_session(session.id)
        assert ok is True
        assert session.status == SessionStatus.CANCELLED

    def test_cancel_nonexistent(self):
        mgr = SessionManager()
        assert mgr.cancel_session("nope") is False

    def test_wait_for_session(self):
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.start_session(session.id, _quick_task)
        result = mgr.wait_for_session(session.id, timeout=1.0)
        assert result is not None
        assert result.status == SessionStatus.COMPLETED

    def test_wait_all_multiple(self):
        mgr = SessionManager()
        s1 = mgr.create_session(name="w1")
        s2 = mgr.create_session(name="w2")
        mgr.start_session(s1.id, _quick_task)
        mgr.start_session(s2.id, _quick_task)
        results = mgr.wait_all([s1.id, s2.id], timeout=1.0)
        assert len(results) == 2
        assert all(s.status == SessionStatus.COMPLETED for s in results)

    def test_clear_completed(self):
        mgr = SessionManager()
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
    def test_creates_session(self):
        mgr = SessionManager()
        tool = CreateSessionTool(manager=mgr)
        result = tool.run(name="my-session")
        assert result.success is True
        assert result.data["name"] == "my-session"
        assert result.data["status"] == "created"
        assert mgr.get_session(result.data["session_id"]) is not None


class TestStartSessionTool:
    def test_starts_session(self):
        mgr = SessionManager()
        tool_create = CreateSessionTool(manager=mgr)
        tool_start = StartSessionTool(manager=mgr)
        r = tool_create.run(name="starter")
        sid = r.data["session_id"]
        result = tool_start.run(session_id=sid, task="do something")
        assert result.success is True
        assert result.data["status"] == "running"
        # Wait for completion
        mgr.wait_for_session(sid, timeout=1.0)
        session = mgr.get_session(sid)
        assert session.status == SessionStatus.COMPLETED

    def test_start_missing_session(self):
        mgr = SessionManager()
        tool = StartSessionTool(manager=mgr)
        result = tool.run(session_id="nonexistent", task="x")
        assert result.success is False


class TestInjectMessageTool:
    def test_injects_message(self):
        mgr = SessionManager()
        session = mgr.create_session()
        tool = InjectMessageTool(manager=mgr)
        result = tool.run(session_id=session.id, content="hello world")
        assert result.success is True
        assert result.data["injected"] is True
        assert result.data["message_count"] == 1

    def test_invalid_session_returns_error(self):
        mgr = SessionManager()
        tool = InjectMessageTool(manager=mgr)
        result = tool.run(session_id="bad-id", content="hello")
        assert result.success is False


class TestGetSessionStatusTool:
    def test_returns_status(self):
        mgr = SessionManager()
        session = mgr.create_session(name="status-check")
        tool = GetSessionStatusTool(manager=mgr)
        result = tool.run(session_id=session.id)
        assert result.success is True
        assert result.data["status"] == "created"
        assert result.data["name"] == "status-check"

    def test_nonexistent_session(self):
        mgr = SessionManager()
        tool = GetSessionStatusTool(manager=mgr)
        result = tool.run(session_id="does-not-exist")
        assert result.success is False


class TestListSessionsTool:
    def test_lists_all(self):
        mgr = SessionManager()
        mgr.create_session(name="a")
        mgr.create_session(name="b")
        tool = ListSessionsTool(manager=mgr)
        result = tool.run()
        assert result.success is True
        assert result.data["count"] == 2

    def test_filters_by_status(self):
        mgr = SessionManager()
        s1 = mgr.create_session(name="a")
        s2 = mgr.create_session(name="b")
        s2.status = SessionStatus.RUNNING
        tool = ListSessionsTool(manager=mgr)
        result = tool.run(status="running")
        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["sessions"][0]["name"] == "b"


class TestWaitSessionTool:
    def test_waits_and_returns_result(self):
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.start_session(session.id, _quick_task)
        tool = WaitSessionTool(manager=mgr)
        result = tool.run(session_id=session.id, timeout=1.0)
        assert result.success is True
        assert result.data["status"] == "completed"
        assert result.data["result"] == "done"

    def test_timeout_handling(self):
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.start_session(session.id, _slow_task)
        tool = WaitSessionTool(manager=mgr)
        result = tool.run(session_id=session.id, timeout=0.1)
        assert result.success is True
        # Should still be running after short timeout
        assert result.data["status"] == "running"


class TestCollectResultsTool:
    def test_collects_from_multiple(self):
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        mgr.start_session(s1.id, _quick_task)
        mgr.start_session(s2.id, _quick_task)
        tool = CollectResultsTool(manager=mgr)
        result = tool.run(session_ids=f"{s1.id},{s2.id}", timeout=1.0)
        assert result.success is True
        assert result.data["all_completed"] is True
        assert len(result.data["results"]) == 2


class TestCancelSessionTool:
    def test_cancels_session(self):
        mgr = SessionManager()
        session = mgr.create_session()
        tool = CancelSessionTool(manager=mgr)
        result = tool.run(session_id=session.id)
        assert result.success is True
        assert result.data["cancelled"] is True
        assert session.status == SessionStatus.CANCELLED


class TestAddFinishHookTool:
    def test_adds_log_hook(self):
        mgr = SessionManager()
        session = mgr.create_session()
        tool = AddFinishHookTool(manager=mgr)
        result = tool.run(
            session_id=session.id,
            hook_type="log",
            hook_config='{"path": "/tmp/test_agent.log"}',
        )
        assert result.success is True
        assert result.data["hook_added"] is True
        assert result.data["hook_type"] == "log"
        assert len(session._on_finish) == 1

    def test_adds_notify_hook(self):
        mgr = SessionManager()
        session = mgr.create_session()
        tool = AddFinishHookTool(manager=mgr)
        result = tool.run(session_id=session.id, hook_type="notify")
        assert result.success is True
        # Run session and verify notification fires
        mgr.start_session(session.id, _quick_task)
        mgr.wait_for_session(session.id, timeout=1.0)
        assert "notification" in session.metadata
        assert session.metadata["notification"]["status"] == "completed"

    def test_invalid_hook_type(self):
        mgr = SessionManager()
        session = mgr.create_session()
        tool = AddFinishHookTool(manager=mgr)
        result = tool.run(session_id=session.id, hook_type="bad")
        assert result.success is False


class TestParallelSessions:
    def test_three_concurrent_sessions(self):
        mgr = SessionManager()
        sessions = [mgr.create_session(name=f"parallel-{i}") for i in range(3)]
        for s in sessions:
            mgr.start_session(s.id, _quick_task)
        results = mgr.wait_all([s.id for s in sessions], timeout=2.0)
        assert len(results) == 3
        assert all(s.status == SessionStatus.COMPLETED for s in results)
        assert all(s.result == "done" for s in results)
