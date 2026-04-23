"""
Email tools for quartermaster-tools.

Provides:
- send_email: Send email via SMTP
- read_email: Read email via IMAP
- search_email: Search email via IMAP SEARCH
"""

from quartermaster_tools.builtin.email.tools import (
    _send_timestamps,
    read_email,
    search_email,
    send_email,
)

__all__ = [
    "_send_timestamps",
    "read_email",
    "search_email",
    "send_email",
]
