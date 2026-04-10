"""
Email tools for quartermaster-tools.

Provides:
- SendEmailTool: Send email via SMTP
- ReadEmailTool: Read email via IMAP
- SearchEmailTool: Search email via IMAP SEARCH
"""

from quartermaster_tools.builtin.email.tools import (
    ReadEmailTool,
    SearchEmailTool,
    SendEmailTool,
)

__all__ = [
    "ReadEmailTool",
    "SearchEmailTool",
    "SendEmailTool",
]
