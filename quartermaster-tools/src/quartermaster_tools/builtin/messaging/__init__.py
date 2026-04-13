"""
Messaging tools for quartermaster-tools.

Provides:
- slack_message: Send messages to Slack channels
- slack_read: Read Slack channel history
- webhook_notify: Send POST to any webhook URL
- discord_message: Send messages to Discord via webhook
"""

from quartermaster_tools.builtin.messaging.tools import (
    DiscordMessageTool,
    SlackMessageTool,
    SlackReadTool,
    WebhookNotifyTool,
    discord_message,
    slack_message,
    slack_read,
    webhook_notify,
)

__all__ = [
    "discord_message",
    "DiscordMessageTool",
    "slack_message",
    "SlackMessageTool",
    "slack_read",
    "SlackReadTool",
    "webhook_notify",
    "WebhookNotifyTool",
]
