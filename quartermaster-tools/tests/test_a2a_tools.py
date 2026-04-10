"""Tests for A2A (Agent-to-Agent) protocol tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from quartermaster_tools.builtin.a2a.discover import A2ADiscoverTool
from quartermaster_tools.builtin.a2a.register import A2ARegisterTool
from quartermaster_tools.builtin.a2a.task import (
    A2ACheckStatusTool,
    A2ACollectResultTool,
    A2ASendTaskTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "TestAgent",
    "description": "A test agent",
    "url": "https://agent.example.com",
    "version": "2.0.0",
    "skills": [
        {"id": "s1", "name": "Summarise", "description": "Summarise text"},
        {"id": "s2", "name": "Translate", "description": "Translate text"},
    ],
    "capabilities": {"streaming": True, "pushNotifications": False},
}


def _mock_response(*, status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _mock_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get.return_value = response
    client.post.return_value = response
    return client


# All network tests bypass SSRF checks (we're testing protocol logic, not DNS)
SSRF_PATCH = "quartermaster_tools.builtin.a2a.discover._is_private_url"


# ---------------------------------------------------------------------------
# A2ADiscoverTool
# ---------------------------------------------------------------------------


class TestA2ADiscoverTool:
    def setup_method(self) -> None:
        self.tool = A2ADiscoverTool()

    def test_name_and_version(self) -> None:
        assert self.tool.name() == "a2a_discover"
        assert self.tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        info = self.tool.info()
        assert info.name == "a2a_discover"
        assert len(info.parameters) == 1

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_discover_success(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        resp = _mock_response(json_data=AGENT_CARD)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com")
        assert result.success
        assert result.data["agent_name"] == "TestAgent"
        assert result.data["version"] == "2.0.0"
        assert len(result.data["skills"]) == 2
        assert result.data["capabilities"]["streaming"] is True

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_discover_404(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        resp = _mock_response(status_code=404)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com")
        assert not result.success
        assert "404" in result.error

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_discover_connection_error(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = client

        result = self.tool.run(agent_url="https://agent.example.com")
        assert not result.success
        assert "Connection error" in result.error

    def test_discover_missing_url(self) -> None:
        result = self.tool.run()
        assert not result.success
        assert "required" in result.error.lower()

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_discover_parses_fields(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        card = {"name": "X", "description": "D", "version": "3.0", "skills": [], "capabilities": {}}
        resp = _mock_response(json_data=card)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com")
        assert result.success
        assert result.data["agent_name"] == "X"
        assert result.data["description"] == "D"
        assert result.data["skills"] == []

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_discover_timeout(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = client

        result = self.tool.run(agent_url="https://agent.example.com")
        assert not result.success
        assert "timed out" in result.error.lower()


# ---------------------------------------------------------------------------
# A2ASendTaskTool
# ---------------------------------------------------------------------------


class TestA2ASendTaskTool:
    def setup_method(self) -> None:
        self.tool = A2ASendTaskTool()

    def test_name_and_version(self) -> None:
        assert self.tool.name() == "a2a_send_task"

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_send_success(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc_result = {
            "jsonrpc": "2.0",
            "result": {
                "id": "task-1",
                "status": {"state": "working"},
                "artifacts": [],
            },
        }
        resp = _mock_response(json_data=rpc_result)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(
            agent_url="https://agent.example.com",
            task_message="Do something",
            task_id="task-1",
        )
        assert result.success
        assert result.data["task_id"] == "task-1"
        assert result.data["status"] == "working"

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_send_error_response(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc_err = {
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid request"},
        }
        resp = _mock_response(json_data=rpc_err)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(
            agent_url="https://agent.example.com",
            task_message="Do something",
        )
        assert not result.success
        assert "Invalid request" in result.error

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_send_timeout(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = client

        result = self.tool.run(
            agent_url="https://agent.example.com",
            task_message="Do something",
        )
        assert not result.success
        assert "timed out" in result.error.lower()

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_send_jsonrpc_format(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc_result = {
            "jsonrpc": "2.0",
            "result": {"id": "t1", "status": {"state": "completed"}, "artifacts": []},
        }
        resp = _mock_response(json_data=rpc_result)
        client = _mock_client(resp)
        mock_client_cls.return_value = client

        self.tool.run(
            agent_url="https://agent.example.com",
            task_message="test",
            task_id="t1",
        )

        # Verify JSON-RPC payload structure
        call_args = client.post.call_args
        payload = call_args[1]["json"]
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "tasks/send"
        assert "id" in payload
        assert payload["params"]["message"]["role"] == "user"
        assert payload["params"]["message"]["parts"][0]["type"] == "text"

    def test_send_missing_message(self) -> None:
        result = self.tool.run(agent_url="https://agent.example.com")
        assert not result.success
        assert "task_message" in result.error.lower()

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_send_auto_generates_task_id(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc_result = {
            "jsonrpc": "2.0",
            "result": {"id": "auto", "status": {"state": "working"}, "artifacts": []},
        }
        resp = _mock_response(json_data=rpc_result)
        client = _mock_client(resp)
        mock_client_cls.return_value = client

        self.tool.run(agent_url="https://agent.example.com", task_message="test")

        payload = client.post.call_args[1]["json"]
        # task_id should be a uuid string (36 chars with hyphens)
        assert len(payload["params"]["id"]) == 36


# ---------------------------------------------------------------------------
# A2ACheckStatusTool
# ---------------------------------------------------------------------------


class TestA2ACheckStatusTool:
    def setup_method(self) -> None:
        self.tool = A2ACheckStatusTool()

    def test_name(self) -> None:
        assert self.tool.name() == "a2a_check_status"

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_status_working(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {
            "jsonrpc": "2.0",
            "result": {"id": "t1", "status": {"state": "working"}, "artifacts": []},
        }
        resp = _mock_response(json_data=rpc)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com", task_id="t1")
        assert result.success
        assert result.data["status"]["state"] == "working"

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_status_completed(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {
            "jsonrpc": "2.0",
            "result": {
                "id": "t1",
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": "done"}]}],
            },
        }
        resp = _mock_response(json_data=rpc)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com", task_id="t1")
        assert result.success
        assert result.data["status"]["state"] == "completed"
        assert len(result.data["artifacts"]) == 1

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_status_failed(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {
            "jsonrpc": "2.0",
            "result": {
                "id": "t1",
                "status": {"state": "failed", "message": "something broke"},
                "artifacts": [],
            },
        }
        resp = _mock_response(json_data=rpc)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com", task_id="t1")
        assert result.success
        assert result.data["status"]["state"] == "failed"

    def test_status_missing_task_id(self) -> None:
        result = self.tool.run(agent_url="https://agent.example.com")
        assert not result.success
        assert "task_id" in result.error.lower()

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_status_jsonrpc_method(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {
            "jsonrpc": "2.0",
            "result": {"id": "t1", "status": {"state": "working"}, "artifacts": []},
        }
        resp = _mock_response(json_data=rpc)
        client = _mock_client(resp)
        mock_client_cls.return_value = client

        self.tool.run(agent_url="https://agent.example.com", task_id="t1")

        payload = client.post.call_args[1]["json"]
        assert payload["method"] == "tasks/get"
        assert payload["params"]["id"] == "t1"


# ---------------------------------------------------------------------------
# A2ACollectResultTool
# ---------------------------------------------------------------------------


class TestA2ACollectResultTool:
    def setup_method(self) -> None:
        self.tool = A2ACollectResultTool()

    def test_name(self) -> None:
        assert self.tool.name() == "a2a_collect_result"

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_collect_completed(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {
            "jsonrpc": "2.0",
            "result": {
                "id": "t1",
                "status": {"state": "completed"},
                "artifacts": [
                    {"parts": [{"type": "text", "text": "Result A"}]},
                    {"parts": [{"type": "text", "text": "Result B"}]},
                ],
            },
        }
        resp = _mock_response(json_data=rpc)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com", task_id="t1")
        assert result.success
        assert result.data["completed"] is True
        assert result.data["status"] == "completed"
        assert len(result.data["results"]) == 2
        assert result.data["results"][0]["content"] == "Result A"

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_collect_incomplete(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {
            "jsonrpc": "2.0",
            "result": {
                "id": "t1",
                "status": {"state": "working"},
                "artifacts": [],
            },
        }
        resp = _mock_response(json_data=rpc)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com", task_id="t1")
        assert result.success
        assert result.data["completed"] is False
        assert result.data["results"] == []

    @patch(SSRF_PATCH, return_value=False)
    @patch("httpx.Client")
    def test_collect_error_response(self, mock_client_cls: MagicMock, _: MagicMock) -> None:
        rpc = {"jsonrpc": "2.0", "error": {"code": -32001, "message": "Not found"}}
        resp = _mock_response(json_data=rpc)
        mock_client_cls.return_value = _mock_client(resp)

        result = self.tool.run(agent_url="https://agent.example.com", task_id="t1")
        assert not result.success
        assert "Not found" in result.error


# ---------------------------------------------------------------------------
# A2ARegisterTool
# ---------------------------------------------------------------------------


class TestA2ARegisterTool:
    def setup_method(self) -> None:
        self.tool = A2ARegisterTool()

    def test_name_and_version(self) -> None:
        assert self.tool.name() == "a2a_register"
        assert self.tool.version() == "1.0.0"

    def test_generate_valid_card(self) -> None:
        result = self.tool.run(
            name="MyAgent",
            description="Does things",
            url="https://my-agent.example.com",
            skills=[{"id": "s1", "name": "Skill1", "description": "Desc1"}],
        )
        assert result.success
        card = result.data["agent_card"]
        assert card["name"] == "MyAgent"
        assert card["description"] == "Does things"
        assert card["url"] == "https://my-agent.example.com"
        assert card["version"] == "1.0.0"
        assert len(card["skills"]) == 1
        assert card["capabilities"]["streaming"] is False
        assert card["capabilities"]["pushNotifications"] is False

    def test_save_to_file(self, tmp_path: object) -> None:
        out = str(tmp_path) + "/agent.json"  # type: ignore[operator]
        result = self.tool.run(
            name="MyAgent",
            description="Does things",
            url="https://my-agent.example.com",
            skills=[{"id": "s1", "name": "Skill1", "description": "Desc1"}],
            output_path=out,
        )
        assert result.success
        assert result.data["saved_to"] == out

        with open(out) as f:
            saved = json.load(f)
        assert saved["name"] == "MyAgent"

    def test_validate_structure(self) -> None:
        result = self.tool.run(
            name="Agent",
            description="Desc",
            url="https://agent.example.com",
            skills=[
                {"id": "a", "name": "A", "description": "AA"},
                {"id": "b", "name": "B", "description": "BB"},
            ],
            version="2.5.0",
            streaming=True,
            push_notifications=True,
        )
        assert result.success
        card = result.data["agent_card"]
        assert card["version"] == "2.5.0"
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"]["pushNotifications"] is True
        assert len(card["skills"]) == 2

    def test_without_optional_fields(self) -> None:
        result = self.tool.run(
            name="Agent",
            description="Desc",
            url="https://agent.example.com",
            skills=[{"id": "a", "name": "A", "description": "AA"}],
        )
        assert result.success
        assert result.data["saved_to"] is None
        assert result.data["agent_card"]["version"] == "1.0.0"

    def test_missing_required_name(self) -> None:
        result = self.tool.run(
            description="Desc",
            url="https://x.com",
            skills=[{"id": "a", "name": "A", "description": "AA"}],
        )
        assert not result.success

    def test_missing_required_skills(self) -> None:
        result = self.tool.run(
            name="Agent",
            description="Desc",
            url="https://x.com",
        )
        assert not result.success

    def test_info_descriptor(self) -> None:
        info = self.tool.info()
        assert info.is_local is True
        assert len(info.parameters) == 8


# ---------------------------------------------------------------------------
# SSRF Protection
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    """Verify that private/internal IP URLs are blocked."""

    def setup_method(self) -> None:
        self.discover = A2ADiscoverTool()
        self.send = A2ASendTaskTool()

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1",
            "http://10.0.0.1",
            "http://172.16.0.1",
            "http://192.168.1.1",
            "http://169.254.169.254",
            "http://[::1]",
        ],
    )
    def test_discover_blocks_private_ips(self, url: str) -> None:
        result = self.discover.run(agent_url=url)
        assert not result.success
        assert "private" in result.error.lower() or "denied" in result.error.lower()

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1",
            "http://10.0.0.1",
            "http://192.168.0.1",
        ],
    )
    def test_send_blocks_private_ips(self, url: str) -> None:
        result = self.send.run(agent_url=url, task_message="test")
        assert not result.success
        assert "private" in result.error.lower() or "denied" in result.error.lower()

    def test_discover_blocks_invalid_scheme(self) -> None:
        result = self.discover.run(agent_url="ftp://agent.example.com")
        assert not result.success
        assert "scheme" in result.error.lower()
