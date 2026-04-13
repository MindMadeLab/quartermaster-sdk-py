"""
Email tools for quartermaster-tools.

Provides:
- send_email: Send email via SMTP
- read_email: Read email via IMAP
- search_email: Search email via IMAP SEARCH
"""

from quartermaster_tools.builtin.email.tools import (
    send_email,
    read_email,
    search_email,
    SendEmailTool,
    ReadEmailTool,
    SearchEmailTool,
    _send_timestamps,
)

__all__ = [
    "send_email",
    "read_email",
    "search_email",
    # Backward-compatible aliases
    "SendEmailTool",
    "ReadEmailTool",
    "SearchEmailTool",
    "_send_timestamps",
]
