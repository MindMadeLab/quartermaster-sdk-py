"""Tests for the QuartermasterCloud upload client."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

from quartermaster_graph import Graph, QuartermasterCloud, CloudError


class TestCloudInit:
    """Test QuartermasterCloud initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                QuartermasterCloud(api_key="")

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-test"}):
            cloud = QuartermasterCloud()
            assert cloud.api_key == "qm-test"

    def test_api_key_from_param(self):
        cloud = QuartermasterCloud(api_key="qm-param")
        assert cloud.api_key == "qm-param"

    def test_default_base_url(self):
        cloud = QuartermasterCloud(api_key="qm-test")
        assert "quartermaster" in cloud.base_url

    def test_custom_base_url(self):
        cloud = QuartermasterCloud(api_key="qm-test", base_url="http://localhost:8000")
        assert cloud.base_url == "http://localhost:8000"


class _MockHandler(BaseHTTPRequestHandler):
    """HTTP handler for testing upload."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/v1/agents":
            response = {"id": "agt_test123", "name": body.get("name", "")}
        elif "/versions" in self.path:
            response = {"id": "ver_test456", "version": body.get("version", "0.1.0")}
        else:
            response = {"ok": True}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):  # noqa: A002
        pass  # suppress logging


class TestCloudUpload:
    """Test graph upload to mock server."""

    @pytest.fixture(autouse=True)
    def _server(self):
        server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.base_url = f"http://127.0.0.1:{port}"
        yield
        server.shutdown()

    def test_upload_creates_agent_and_version(self):
        cloud = QuartermasterCloud(api_key="qm-test", base_url=self.base_url)
        graph = Graph("Test Agent").start().instruction("Work").end()

        result = cloud.upload(graph, version="1.0.0")
        assert result["id"] == "ver_test456"
        assert result["agent_id"] == "agt_test123"

    def test_upload_with_existing_agent_id(self):
        cloud = QuartermasterCloud(api_key="qm-test", base_url=self.base_url)
        graph = Graph("Test").start().instruction("Work").end()

        result = cloud.upload(graph, agent_id="agt_existing", version="2.0.0")
        assert result["version"] == "2.0.0"

    def test_upload_validates_graph(self):
        cloud = QuartermasterCloud(api_key="qm-test", base_url=self.base_url)
        graph = Graph("Bad Graph")  # No start node

        with pytest.raises(ValueError, match="Start node"):
            cloud.upload(graph)


class TestAllowedAgents:
    """Test allowed_agents on GraphBuilder."""

    def test_set_allowed_agents(self):
        graph = (
            Graph("Coordinator")
            .allowed_agents("researcher", "writer")
            .start()
            .user("Task")
            .sub_agent("Research", graph_id="researcher")
            .end()
        )
        assert graph._allowed_agents == ["researcher", "writer"]

    def test_default_empty(self):
        graph = Graph("Test").start().end()
        assert graph._allowed_agents == []
