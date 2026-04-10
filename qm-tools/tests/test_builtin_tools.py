"""Tests for built-in tools: ReadFileTool, WriteFileTool, WebRequestTool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qm_tools.builtin.file_read import ReadFileTool
from qm_tools.builtin.file_write import WriteFileTool
from qm_tools.builtin.web_request import WebRequestTool
from qm_tools.types import ToolResult


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------


class TestReadFileTool:
    """Tests for ReadFileTool."""

    def test_name_and_version(self) -> None:
        tool = ReadFileTool()
        assert tool.name() == "read_file"
        assert tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        tool = ReadFileTool()
        info = tool.info()
        assert info.name == "read_file"
        assert info.is_local is True

    def test_parameters_defined(self) -> None:
        tool = ReadFileTool()
        params = tool.parameters()
        names = [p.name for p in params]
        assert "path" in names
        assert "encoding" in names

    def test_read_existing_file(self, tmp_path: str) -> None:
        file = tmp_path / "test.txt"
        file.write_text("hello world", encoding="utf-8")

        tool = ReadFileTool()
        result = tool.run(path=str(file))

        assert result.success is True
        assert result.data["content"] == "hello world"

    def test_read_nonexistent_file(self) -> None:
        tool = ReadFileTool()
        result = tool.run(path="/tmp/nonexistent_file_abc123xyz.txt")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_directory_fails(self, tmp_path: str) -> None:
        tool = ReadFileTool()
        result = tool.run(path=str(tmp_path))

        assert result.success is False
        assert "not a file" in result.error.lower()

    def test_file_too_large(self, tmp_path: str) -> None:
        file = tmp_path / "big.txt"
        file.write_text("x" * 100, encoding="utf-8")

        tool = ReadFileTool(max_file_size=50)
        result = tool.run(path=str(file))

        assert result.success is False
        assert "too large" in result.error.lower()

    def test_blocked_path_etc_shadow(self) -> None:
        tool = ReadFileTool()
        result = tool.run(path="/etc/shadow")

        # On macOS /etc/shadow doesn't exist; on Linux it's blocked.
        # Either way the result should be a failure.
        assert result.success is False

    def test_blocked_path_proc(self) -> None:
        tool = ReadFileTool()
        result = tool.run(path="/proc/cpuinfo")

        assert result.success is False
        assert "access denied" in result.error.lower()

    def test_allowed_base_dir_restriction(self, tmp_path: str) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        outside_file = outside / "secret.txt"
        outside_file.write_text("secret", encoding="utf-8")

        tool = ReadFileTool(allowed_base_dir=str(allowed))
        result = tool.run(path=str(outside_file))

        assert result.success is False
        assert "access denied" in result.error.lower()

    def test_allowed_base_dir_permits_inside(self, tmp_path: str) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        inside_file = allowed / "ok.txt"
        inside_file.write_text("allowed content", encoding="utf-8")

        tool = ReadFileTool(allowed_base_dir=str(allowed))
        result = tool.run(path=str(inside_file))

        assert result.success is True
        assert result.data["content"] == "allowed content"

    def test_missing_path_param(self) -> None:
        tool = ReadFileTool()
        result = tool.run()

        assert result.success is False
        assert "required" in result.error.lower()

    def test_safe_run_validates_params(self) -> None:
        tool = ReadFileTool()
        result = tool.safe_run()  # missing required 'path'

        assert result.success is False


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    def test_name_and_version(self) -> None:
        tool = WriteFileTool()
        assert tool.name() == "write_file"
        assert tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        tool = WriteFileTool()
        info = tool.info()
        assert info.name == "write_file"

    def test_write_new_file(self, tmp_path: str) -> None:
        file = tmp_path / "output.txt"

        tool = WriteFileTool()
        result = tool.run(path=str(file), content="hello")

        assert result.success is True
        assert file.read_text(encoding="utf-8") == "hello"
        assert result.data["mode"] == "overwrite"

    def test_overwrite_existing_file(self, tmp_path: str) -> None:
        file = tmp_path / "output.txt"
        file.write_text("old content", encoding="utf-8")

        tool = WriteFileTool()
        result = tool.run(path=str(file), content="new content")

        assert result.success is True
        assert file.read_text(encoding="utf-8") == "new content"

    def test_append_mode(self, tmp_path: str) -> None:
        file = tmp_path / "output.txt"
        file.write_text("line1\n", encoding="utf-8")

        tool = WriteFileTool()
        result = tool.run(path=str(file), content="line2\n", append=True)

        assert result.success is True
        assert file.read_text(encoding="utf-8") == "line1\nline2\n"
        assert result.data["mode"] == "append"

    def test_creates_parent_directories(self, tmp_path: str) -> None:
        file = tmp_path / "a" / "b" / "c" / "deep.txt"

        tool = WriteFileTool(create_dirs=True)
        result = tool.run(path=str(file), content="deep")

        assert result.success is True
        assert file.read_text(encoding="utf-8") == "deep"

    def test_content_too_large(self, tmp_path: str) -> None:
        file = tmp_path / "big.txt"

        tool = WriteFileTool(max_content_size=10)
        result = tool.run(path=str(file), content="x" * 100)

        assert result.success is False
        assert "too large" in result.error.lower()

    def test_blocked_path(self) -> None:
        tool = WriteFileTool()
        result = tool.run(path="/etc/evil.conf", content="bad")

        # On macOS /etc -> /private/etc, both are blocked. Result is always failure.
        assert result.success is False

    def test_allowed_base_dir_restriction(self, tmp_path: str) -> None:
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside_file = tmp_path / "outside.txt"

        tool = WriteFileTool(allowed_base_dir=str(allowed))
        result = tool.run(path=str(outside_file), content="nope")

        assert result.success is False
        assert "access denied" in result.error.lower()

    def test_missing_path_param(self) -> None:
        tool = WriteFileTool()
        result = tool.run(content="hello")

        assert result.success is False
        assert "required" in result.error.lower()

    def test_bytes_written_reported(self, tmp_path: str) -> None:
        file = tmp_path / "out.txt"

        tool = WriteFileTool()
        result = tool.run(path=str(file), content="abc")

        assert result.success is True
        assert result.data["bytes_written"] == 3


# ---------------------------------------------------------------------------
# WebRequestTool
# ---------------------------------------------------------------------------


class TestWebRequestTool:
    """Tests for WebRequestTool."""

    def test_name_and_version(self) -> None:
        tool = WebRequestTool()
        assert tool.name() == "web_request"
        assert tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        tool = WebRequestTool()
        info = tool.info()
        assert info.name == "web_request"
        assert info.is_local is False

    def test_parameters_defined(self) -> None:
        tool = WebRequestTool()
        params = tool.parameters()
        names = [p.name for p in params]
        assert "url" in names
        assert "method" in names
        assert "headers" in names
        assert "body" in names

    def test_missing_url(self) -> None:
        tool = WebRequestTool()
        result = tool.run()

        assert result.success is False
        assert "required" in result.error.lower()

    def test_unsupported_method(self) -> None:
        tool = WebRequestTool()
        result = tool.run(url="https://example.com", method="DELETE")

        assert result.success is False
        assert "unsupported" in result.error.lower()

    def _make_mock_httpx(self) -> MagicMock:
        """Create a mock httpx module with required exception classes."""
        mock_httpx = MagicMock()
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
        mock_httpx.ConnectError = type("ConnectError", (Exception,), {})
        mock_httpx.HTTPError = type("HTTPError", (Exception,), {})
        return mock_httpx

    def _make_mock_response(
        self, text: str = "", status_code: int = 200, headers: dict | None = None
    ) -> MagicMock:
        """Create a mock httpx response."""
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        resp.headers = headers or {}
        resp.content = text.encode()
        resp.url = "https://example.com/api"
        return resp

    def test_get_request_success(self) -> None:
        mock_httpx = self._make_mock_httpx()
        mock_response = self._make_mock_response(
            text='{"ok": true}',
            status_code=200,
            headers={"content-type": "application/json"},
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        import sys
        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebRequestTool()
            result = tool.run(url="https://example.com/api")

            assert result.success is True
            assert result.data["status_code"] == 200
            assert result.data["body"] == '{"ok": true}'
            mock_client.get.assert_called_once_with("https://example.com/api", headers=None)
        finally:
            sys.modules.pop("httpx", None)

    def test_post_request_success(self) -> None:
        mock_httpx = self._make_mock_httpx()
        mock_response = self._make_mock_response(text="created", status_code=201)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        import sys
        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebRequestTool()
            result = tool.run(
                url="https://example.com/api",
                method="POST",
                body='{"data": 1}',
            )

            assert result.success is True
            assert result.data["status_code"] == 201
            mock_client.post.assert_called_once()
        finally:
            sys.modules.pop("httpx", None)

    def test_timeout_error(self) -> None:
        mock_httpx = self._make_mock_httpx()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = mock_httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = mock_client

        import sys
        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebRequestTool()
            result = tool.run(url="https://example.com/slow")

            assert result.success is False
            assert "timed out" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)

    def test_response_too_large(self) -> None:
        mock_httpx = self._make_mock_httpx()
        mock_response = self._make_mock_response(text="x" * 200)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        import sys
        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebRequestTool(max_response_size=100)
            result = tool.run(url="https://example.com")

            assert result.success is False
            assert "too large" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)

    def test_httpx_not_installed(self) -> None:
        import sys

        # Temporarily make httpx unimportable
        httpx_mod = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]
        try:
            tool = WebRequestTool()
            result = tool.run(url="https://example.com")
            assert result.success is False
            assert "httpx is required" in result.error
        finally:
            if httpx_mod is not None:
                sys.modules["httpx"] = httpx_mod
            else:
                sys.modules.pop("httpx", None)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestBuiltinRegistration:
    """Test that built-in tools can be registered in a ToolRegistry."""

    def test_register_all_builtins(self) -> None:
        from qm_tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(WebRequestTool())

        assert len(registry) == 3
        assert "read_file" in registry
        assert "write_file" in registry
        assert "web_request" in registry

    def test_export_json_schema(self) -> None:
        from qm_tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(ReadFileTool())

        schemas = registry.to_json_schema()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "read_file"
        assert "parameters" in schemas[0]

    def test_imports_from_top_level(self) -> None:
        from qm_tools import ReadFileTool, WriteFileTool, WebRequestTool

        assert ReadFileTool().name() == "read_file"
        assert WriteFileTool().name() == "write_file"
        assert WebRequestTool().name() == "web_request"
