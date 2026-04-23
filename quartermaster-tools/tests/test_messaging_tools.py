"""Tests for slack_message, slack_read, webhook_notify, and discord_message tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quartermaster_tools.builtin.messaging.tools import (
    discord_message,
    slack_message,
    slack_read,
    webhook_notify,
)


class TestSlackMessageTool:
    def setup_method(self) -> None:
        self.tool = slack_message

    def test_name(self) -> None:
        assert self.tool.name() == "slack_message"

    def test_version(self) -> None:
        assert self.tool.version() == "1.0.0"

    def test_parameters_list(self) -> None:
        params = self.tool.parameters()
        names = [p.name for p in params]
        assert "channel" in names
        assert "text" in names
        assert "thread_ts" in names
        assert "token" in names

    def test_info_descriptor(self) -> None:
        info = self.tool.info()
        assert info.name == "slack_message"

    def test_missing_channel(self) -> None:
        result = self.tool.run(channel="", text="hi", token="tok")
        assert result.success is False
        assert "channel" in result.error.lower()

    def test_missing_text(self) -> None:
        result = self.tool.run(channel="C123", text="", token="tok")
        assert result.success is False
        assert "text" in result.error.lower()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_token(self) -> None:
        result = self.tool.run(channel="C123", text="hello")
        assert result.success is False
        assert "token" in result.error.lower()

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    @patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"})
    def test_successful_send(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "channel": "C123", "ts": "123.456"}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(channel="C123", text="Hello Slack!")
        assert result.success is True
        assert result.data["ts"] == "123.456"

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    @patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"})
    def test_slack_api_error(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(channel="C123", text="Hello")
        assert result.success is False
        assert "channel_not_found" in result.error

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    @patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"})
    def test_timeout_error(self, mock_httpx: MagicMock) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = mock_client
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPError = httpx.HTTPError

        result = self.tool.run(channel="C123", text="Hello")
        assert result.success is False
        assert "timed out" in result.error.lower()


class TestSlackReadTool:
    def setup_method(self) -> None:
        self.tool = slack_read

    def test_name(self) -> None:
        assert self.tool.name() == "slack_read"

    def test_missing_channel(self) -> None:
        result = self.tool.run(channel="", token="tok")
        assert result.success is False

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_token(self) -> None:
        result = self.tool.run(channel="C123")
        assert result.success is False

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    @patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"})
    def test_successful_read(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "messages": [
                {"user": "U123", "text": "Hello!", "ts": "111.222", "type": "message"},
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(channel="C123")
        assert result.success is True
        assert result.data["message_count"] == 1
        assert result.data["messages"][0]["text"] == "Hello!"

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    @patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"})
    def test_slack_read_api_error(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "not_authed"}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(channel="C123")
        assert result.success is False
        assert "not_authed" in result.error


class TestWebhookNotifyTool:
    def setup_method(self) -> None:
        self.tool = webhook_notify

    def test_name(self) -> None:
        assert self.tool.name() == "webhook_notify"

    def test_missing_url(self) -> None:
        result = self.tool.run(url="", payload={"key": "val"})
        assert result.success is False
        assert "url" in result.error.lower()

    def test_missing_payload(self) -> None:
        result = self.tool.run(url="https://hooks.example.com/hook", payload={})
        assert result.success is False
        assert "payload" in result.error.lower()

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    def test_successful_webhook(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(url="https://hooks.example.com/hook", payload={"event": "deploy"})
        assert result.success is True
        assert result.data["status_code"] == 200

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    def test_webhook_with_custom_headers(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(
            url="https://hooks.example.com/hook",
            payload={"data": 1},
            headers={"X-Custom": "value"},
        )
        assert result.success is True
        call_kwargs = mock_client.post.call_args
        headers = (
            call_kwargs[1]["headers"]
            if "headers" in call_kwargs[1]
            else call_kwargs.kwargs["headers"]
        )
        assert headers["X-Custom"] == "value"

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    def test_webhook_http_error(self, mock_httpx: MagicMock) -> None:
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )
        mock_httpx.Client.return_value = mock_client
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.HTTPError = httpx.HTTPError

        result = self.tool.run(url="https://hooks.example.com/hook", payload={"a": 1})
        assert result.success is False
        assert "500" in result.error


class TestDiscordMessageTool:
    def setup_method(self) -> None:
        self.tool = discord_message

    def test_name(self) -> None:
        assert self.tool.name() == "discord_message"

    def test_version(self) -> None:
        assert self.tool.version() == "1.0.0"

    def test_missing_webhook_url(self) -> None:
        result = self.tool.run(webhook_url="", content="hi")
        assert result.success is False
        assert "webhook_url" in result.error.lower()

    def test_missing_content(self) -> None:
        result = self.tool.run(webhook_url="https://discord.com/api/webhooks/123/abc", content="")
        assert result.success is False
        assert "content" in result.error.lower()

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    def test_successful_send(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            content="Hello Discord!",
        )
        assert result.success is True

    @patch("quartermaster_tools.builtin.messaging.tools.httpx")
    def test_discord_timeout(self, mock_httpx: MagicMock) -> None:
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = mock_client
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.HTTPError = httpx.HTTPError

        result = self.tool.run(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            content="Hello",
        )
        assert result.success is False
        assert "timed out" in result.error.lower()
