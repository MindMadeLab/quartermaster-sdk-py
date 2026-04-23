"""
Messaging tools: Slack, Discord, and generic webhooks.

Uses httpx for HTTP requests.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.decorator import tool

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


@tool()
def slack_message(
    channel: str,
    text: str,
    thread_ts: str = None,
    token: str = None,
) -> dict:
    """Send a message to a Slack channel.

    Posts a message to a Slack channel using the Slack Web API
    (chat.postMessage). Supports threading. Requires a bot token.

    Args:
        channel: Slack channel ID or name.
        text: Message text to send.
        thread_ts: Thread timestamp to reply in a thread.
        token: Slack bot token (env: SLACK_BOT_TOKEN).
    """
    channel = channel.strip() if channel else ""
    text = text.strip() if text else ""
    token = token or os.environ.get("SLACK_BOT_TOKEN", "")

    if not channel:
        raise ValueError("Parameter 'channel' is required.")
    if not text:
        raise ValueError("Parameter 'text' is required.")
    if not token:
        raise ValueError(
            "Slack bot token not provided. Set token parameter or SLACK_BOT_TOKEN env var."
        )

    if httpx is None:
        raise ImportError(
            "httpx is required for SlackMessageTool. Install with: pip install quartermaster-tools[web]"
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
        raise TimeoutError("Slack API request timed out.")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")

    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: {data.get('error', 'unknown')}")

    return {
        "channel": data.get("channel", channel),
        "ts": data.get("ts", ""),
        "message": "Message sent successfully.",
    }


@tool()
def slack_read(
    channel: str,
    count: int = 10,
    token: str = None,
) -> dict:
    """Read messages from a Slack channel.

    Fetches recent messages from a Slack channel using the
    conversations.history API endpoint.

    Args:
        channel: Slack channel ID.
        count: Number of messages to fetch (default 10).
        token: Slack bot token (env: SLACK_BOT_TOKEN).
    """
    channel = channel.strip() if channel else ""
    count = int(count or 10)
    token = token or os.environ.get("SLACK_BOT_TOKEN", "")

    if not channel:
        raise ValueError("Parameter 'channel' is required.")
    if not token:
        raise ValueError(
            "Slack bot token not provided. Set token parameter or SLACK_BOT_TOKEN env var."
        )

    if httpx is None:
        raise ImportError(
            "httpx is required for SlackReadTool. Install with: pip install quartermaster-tools[web]"
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
        raise TimeoutError("Slack API request timed out.")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")

    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: {data.get('error', 'unknown')}")

    messages = []
    for msg in data.get("messages", []):
        messages.append(
            {
                "user": msg.get("user", ""),
                "text": msg.get("text", ""),
                "ts": msg.get("ts", ""),
                "type": msg.get("type", ""),
            }
        )

    return {"channel": channel, "messages": messages, "message_count": len(messages)}


@tool()
def webhook_notify(
    url: str,
    payload: dict = None,
    headers: dict = None,
) -> dict:
    """Send a POST request to a webhook URL.

    Posts a JSON payload to any webhook URL. Useful for notifications
    to services like Zapier, IFTTT, or custom endpoints.

    Args:
        url: Webhook URL to POST to.
        payload: JSON payload dict to send.
        headers: Optional HTTP headers dict.
    """
    url = url.strip() if url else ""
    custom_headers: dict[str, str] = headers or {}

    if not url:
        raise ValueError("Parameter 'url' is required.")
    if not payload:
        raise ValueError("Parameter 'payload' is required.")

    if httpx is None:
        raise ImportError(
            "httpx is required for WebhookNotifyTool. Install with: pip install quartermaster-tools[web]"
        )

    req_headers = {"Content-Type": "application/json"}
    req_headers.update(custom_headers)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload, headers=req_headers)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "message": "Webhook notification sent successfully.",
            }
    except httpx.TimeoutException:
        raise TimeoutError("Webhook request timed out.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Webhook HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")


@tool()
def discord_message(webhook_url: str, content: str) -> dict:
    """Send a message to Discord via webhook.

    Posts a message to a Discord channel using a webhook URL.
    Simple text messages only.

    Args:
        webhook_url: Discord webhook URL.
        content: Message content to send.
    """
    webhook_url = webhook_url.strip() if webhook_url else ""
    content = content.strip() if content else ""

    if not webhook_url:
        raise ValueError("Parameter 'webhook_url' is required.")
    if not content:
        raise ValueError("Parameter 'content' is required.")

    if httpx is None:
        raise ImportError(
            "httpx is required for DiscordMessageTool. Install with: pip install quartermaster-tools[web]"
        )

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                webhook_url,
                json={"content": content},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return {"message": "Discord message sent successfully."}
    except httpx.TimeoutException:
        raise TimeoutError("Discord webhook request timed out.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Discord HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")
