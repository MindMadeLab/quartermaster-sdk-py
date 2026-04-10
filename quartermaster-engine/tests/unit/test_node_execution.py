"""Tests for NodeExecution state tracking."""

from uuid import uuid4

from quartermaster_engine.context.node_execution import NodeExecution, NodeStatus


class TestNodeStatus:
    def test_terminal_states(self):
        assert NodeStatus.FINISHED.is_terminal
        assert NodeStatus.FAILED.is_terminal
        assert NodeStatus.SKIPPED.is_terminal

    def test_non_terminal_states(self):
        assert not NodeStatus.PENDING.is_terminal
        assert not NodeStatus.RUNNING.is_terminal
        assert not NodeStatus.WAITING_USER.is_terminal
        assert not NodeStatus.WAITING_TOOL.is_terminal

    def test_active_states(self):
        assert NodeStatus.RUNNING.is_active
        assert NodeStatus.WAITING_USER.is_active
        assert NodeStatus.WAITING_TOOL.is_active
        assert not NodeStatus.PENDING.is_active
        assert not NodeStatus.FINISHED.is_active


class TestNodeExecution:
    def test_initial_state(self):
        nid = uuid4()
        execution = NodeExecution(node_id=nid)
        assert execution.status == NodeStatus.PENDING
        assert execution.started_at is None
        assert execution.finished_at is None
        assert execution.result is None
        assert execution.error is None
        assert execution.retry_count == 0

    def test_start(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        assert execution.status == NodeStatus.RUNNING
        assert execution.started_at is not None

    def test_finish(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        execution.finish(result="done", output_data={"key": "value"})
        assert execution.status == NodeStatus.FINISHED
        assert execution.finished_at is not None
        assert execution.result == "done"
        assert execution.output_data == {"key": "value"}

    def test_fail(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        execution.fail("something broke")
        assert execution.status == NodeStatus.FAILED
        assert execution.error == "something broke"
        assert execution.finished_at is not None

    def test_skip(self):
        execution = NodeExecution(node_id=uuid4())
        execution.skip()
        assert execution.status == NodeStatus.SKIPPED
        assert execution.finished_at is not None

    def test_wait_for_user(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        execution.wait_for_user()
        assert execution.status == NodeStatus.WAITING_USER

    def test_wait_for_tool(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        execution.wait_for_tool()
        assert execution.status == NodeStatus.WAITING_TOOL

    def test_duration_seconds(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        execution.finish(result="done")
        duration = execution.duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_duration_none_when_not_finished(self):
        execution = NodeExecution(node_id=uuid4())
        execution.start()
        assert execution.duration_seconds is None

    def test_duration_none_when_not_started(self):
        execution = NodeExecution(node_id=uuid4())
        assert execution.duration_seconds is None
