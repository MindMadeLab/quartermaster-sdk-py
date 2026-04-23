"""
Messaging tools for quartermaster-tools.

Provides:
- slack_message: Send messages to Slack channels
- slack_read: Read Slack channel history
- webhook_notify: Send POST to any webhook URL
- discord_message: Send messages to Discord via webhook
"""

from quartermaster_tools.builtin.messaging.tools import (
    discord_message,
    slack_message,
    slack_read,
    webhook_notify,
)

__all__ = [
    "discord_message",
    "slack_message",
    "slack_read",
    "webhook_notify",
]
