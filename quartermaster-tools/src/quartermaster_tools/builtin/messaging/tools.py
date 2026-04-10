"""
Messaging tools: Slack, Discord, and generic webhooks.

Uses httpx for HTTP requests.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class SlackMessageTool(AbstractTool):
    """Send a message to a Slack channel via the Slack Web API.

    Requires a bot token (SLACK_BOT_TOKEN env var or token parameter).
    """

    def name(self) -> str:
        return "slack_message"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="channel", description="Slack channel ID or name.", type="string", required=True),
            ToolParameter(name="text", description="Message text to send.", type="string", required=True),
            ToolParameter(name="thread_ts", description="Thread timestamp to reply in a thread.", type="string", required=False),
            ToolParameter(name="token", description="Slack bot token (env: SLACK_BOT_TOKEN).", type="string", required=False),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Send a message to a Slack channel.",
            long_description=(
                "Posts a message to a Slack channel using the Slack Web API "
                "(chat.postMessage). Supports threading. Requires a bot token."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        channel: str = kwargs.get("channel", "").strip()
        text: str = kwargs.get("text", "").strip()
        thread_ts: str | None = kwargs.get("thread_ts")
        token = kwargs.get("token") or os.environ.get("SLACK_BOT_TOKEN", "")

        if not channel:
            return ToolResult(success=False, error="Parameter 'channel' is required.")
        if not text:
            return ToolResult(success=False, error="Parameter 'text' is required.")
        if not token:
            return ToolResult(
                success=False,
                error="Slack bot token not provided. Set token parameter or SLACK_BOT_TOKEN env var.",
            )

        if httpx is None:
            return ToolResult(
                success=False,
                error="httpx is required for SlackMessageTool. Install with: pip install quartermaster-tools[web]",
            )

        payload: dict[str, Any] = {"channel": channel, "text": text}
        if thread_ts:
            payload["thread_ts"] = thread_ts

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Slack API request timed out.")
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error: {e}")

        if not data.get("ok"):
            return ToolResult(
                success=False,
                error=f"Slack API error: {data.get('error', 'unknown')}",
            )

        return ToolResult(
            success=True,
            data={
                "channel": data.get("channel", channel),
                "ts": data.get("ts", ""),
                "message": "Message sent successfully.",
            },
        )


class SlackReadTool(AbstractTool):
    """Read messages from a Slack channel via the conversations.history API."""

    def name(self) -> str:
        return "slack_read"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="channel", description="Slack channel ID.", type="string", required=True),
            ToolParameter(name="count", description="Number of messages to fetch (default 10).", type="number", required=False, default=10),
            ToolParameter(name="token", description="Slack bot token (env: SLACK_BOT_TOKEN).", type="string", required=False),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Read messages from a Slack channel.",
            long_description=(
                "Fetches recent messages from a Slack channel using the "
                "conversations.history API endpoint."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        channel: str = kwargs.get("channel", "").strip()
        count: int = int(kwargs.get("count", 10) or 10)
        token = kwargs.get("token") or os.environ.get("SLACK_BOT_TOKEN", "")

        if not channel:
            return ToolResult(success=False, error="Parameter 'channel' is required.")
        if not token:
            return ToolResult(
                success=False,
                error="Slack bot token not provided. Set token parameter or SLACK_BOT_TOKEN env var.",
            )

        if httpx is None:
            return ToolResult(
                success=False,
                error="httpx is required for SlackReadTool. Install with: pip install quartermaster-tools[web]",
            )

        headers = {
            "Authorization": f"Bearer {token}",
        }
        params = {"channel": channel, "limit": count}

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    "https://slack.com/api/conversations.history",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Slack API request timed out.")
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error: {e}")

        if not data.get("ok"):
            return ToolResult(
                success=False,
                error=f"Slack API error: {data.get('error', 'unknown')}",
            )

        messages = []
        for msg in data.get("messages", []):
            messages.append({
                "user": msg.get("user", ""),
                "text": msg.get("text", ""),
                "ts": msg.get("ts", ""),
                "type": msg.get("type", ""),
            })

        return ToolResult(
            success=True,
            data={"channel": channel, "messages": messages, "message_count": len(messages)},
        )


class WebhookNotifyTool(AbstractTool):
    """Send a POST request to a generic webhook URL."""

    def name(self) -> str:
        return "webhook_notify"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="url", description="Webhook URL to POST to.", type="string", required=True),
            ToolParameter(name="payload", description="JSON payload dict to send.", type="object", required=True),
            ToolParameter(name="headers", description="Optional HTTP headers dict.", type="object", required=False),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Send a POST request to a webhook URL.",
            long_description=(
                "Posts a JSON payload to any webhook URL. Useful for notifications "
                "to services like Zapier, IFTTT, or custom endpoints."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        url: str = kwargs.get("url", "").strip()
        payload: dict[str, Any] = kwargs.get("payload", {})
        custom_headers: dict[str, str] = kwargs.get("headers") or {}

        if not url:
            return ToolResult(success=False, error="Parameter 'url' is required.")
        if not payload:
            return ToolResult(success=False, error="Parameter 'payload' is required.")

        if httpx is None:
            return ToolResult(
                success=False,
                error="httpx is required for WebhookNotifyTool. Install with: pip install quartermaster-tools[web]",
            )

        headers = {"Content-Type": "application/json"}
        headers.update(custom_headers)

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return ToolResult(
                    success=True,
                    data={
                        "status_code": response.status_code,
                        "message": "Webhook notification sent successfully.",
                    },
                )
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Webhook request timed out.")
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"Webhook HTTP error {e.response.status_code}: {e.response.text}",
            )
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error: {e}")


class DiscordMessageTool(AbstractTool):
    """Send a message to Discord via webhook URL."""

    def name(self) -> str:
        return "discord_message"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="webhook_url", description="Discord webhook URL.", type="string", required=True),
            ToolParameter(name="content", description="Message content to send.", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Send a message to Discord via webhook.",
            long_description=(
                "Posts a message to a Discord channel using a webhook URL. "
                "Simple text messages only."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        webhook_url: str = kwargs.get("webhook_url", "").strip()
        content: str = kwargs.get("content", "").strip()

        if not webhook_url:
            return ToolResult(success=False, error="Parameter 'webhook_url' is required.")
        if not content:
            return ToolResult(success=False, error="Parameter 'content' is required.")

        if httpx is None:
            return ToolResult(
                success=False,
                error="httpx is required for DiscordMessageTool. Install with: pip install quartermaster-tools[web]",
            )

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    webhook_url,
                    json={"content": content},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return ToolResult(
                    success=True,
                    data={"message": "Discord message sent successfully."},
                )
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Discord webhook request timed out.")
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"Discord HTTP error {e.response.status_code}: {e.response.text}",
            )
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error: {e}")
